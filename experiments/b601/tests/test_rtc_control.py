"""Exercise the RTC actor with the same delayed worker/hardware fixture as sync."""

import numpy as np
import pytest

from experiments.b601.runtime import rtc_controller
from .test_policy_recovery import simulate


@pytest.mark.parametrize(
    "enabled,flag,expected",
    [
        (False, None, False),
        (True, None, True),
        (False, "true", True),
        (True, "false", False),
        (False, "false", False),
        (True, "true", True),
    ],
)
def test_cli_rtc_is_opt_in_and_sync_can_be_forced(
    tmp_path, capsys, enabled, flag, expected
):
    import json
    import yaml
    from dataclasses import asdict
    from experiments.b601.local import main
    from experiments.b601.runtime.config import Config

    config = asdict(Config())
    config.pop("policy")
    config["robot"]["camera_devices"] = {"front": "front", "side": "side"}
    config["robot"]["lock_paths"] = [str(tmp_path / "shared.lock")]
    config["model"]["checkpoints"] = {"5epoch": str(tmp_path / "checkpoint.pt")}
    config["rtc"]["enabled"] = enabled
    if not enabled:
        config.pop("rtc")  # Existing installed configurations need no migration.
    path = tmp_path / "deploy.yml"
    path.write_text(yaml.safe_dump(config))
    args = ["config", "--config", str(path)]
    if flag is not None:
        args.extend(["--rtc", flag])
    assert main(args) == 0
    effective = json.loads(capsys.readouterr().out)
    assert effective["rtc"]["enabled"] is expected
    assert effective["control"]["execution_horizon"] == 16


@pytest.mark.parametrize("supported,dt", [(False, 1 / 30), (True, 1 / 20)])
def test_incompatible_backend_or_timebase_rejected(supported, dt):
    from types import SimpleNamespace
    from experiments.b601.runtime.config import Config
    from experiments.b601.runtime.contracts import ProtocolError

    connection = SimpleNamespace(
        rtc_capability=lambda: dict(
            supported=supported,
            guidance_modes=("lerobot", "pi_vjp"),
            representation="base_frame_ee_delta_v2",
            horizon=32,
            model_dt_s=dt,
        )
    )
    with pytest.raises(ProtocolError, match="RTC contract"):
        rtc_controller.validate_capability(connection, Config())


def test_rtc_bootstrap_guided_execution_and_hold_continuity(tmp_path, monkeypatch):
    r = simulate(
        tmp_path, monkeypatch, engine=rtc_controller, execution_horizon=16, duration=4
    )
    assert r.report["outcome"] == "timeout", r.report
    assert r.report["guided_queries"] >= 2 and r.report["policy_commands"] > 60
    commands = [c for c in r.commands if c["phase"] == "policy"]
    assert commands[0]["action_index"] == 0
    assert any(c["action_index"] > 0 for c in commands if c["query_id"] != "1")
    last = None
    for c in r.commands:
        if c["phase"] == "policy":
            last = c["executed_recorded_command"]
        elif last is not None:
            np.testing.assert_array_equal(c["executed_recorded_command"], last)


def test_empty_guidance_window_rejected_before_hardware():
    from dataclasses import replace
    from types import SimpleNamespace
    from experiments.b601.runtime.config import Config
    from experiments.b601.runtime.contracts import ProtocolError
    connection = SimpleNamespace(rtc_capability=lambda: dict(
        supported=True, guidance_modes=["pi_vjp"],
        representation="base_frame_ee_delta_v2", horizon=32,
        model_dt_s=1 / 30, flow_times=[0.0, 0.5],
    ))
    with pytest.raises(ProtocolError, match="no model sampling steps"):
        rtc_controller.validate_capability(
            connection, Config(rtc=replace(Config().rtc, guidance_mode="pi_vjp", guidance_start=0.8))
        )


@pytest.mark.parametrize("interrupt", ["pause", "camera", "stop"])
def test_rtc_inflight_invalidation_and_exit(tmp_path, monkeypatch, interrupt):
    r = simulate(
        tmp_path, monkeypatch, engine=rtc_controller, interrupt=interrupt, duration=3
    )
    assert r.report["outcome"] == (
        "aborted" if interrupt == "stop" else "timeout"
    ), r.report
    assert r.closed == ["hardware", "model"]
    if interrupt != "stop":
        assert r.report["discarded_responses"] >= 1


def test_rtc_unreachable_discards_whole_plan_and_keeps_last_target(
    tmp_path, monkeypatch
):
    r = simulate(
        tmp_path, monkeypatch, engine=rtc_controller, first="ik_unreachable", duration=3
    )
    assert r.report["outcome"] == "timeout", r.report
    assert r.report["rejected_policy_actions"] == 1
    assert r.report["policy_replans"] == 1
    assert r.report["policy_commands"] > 0


@pytest.mark.parametrize("fatal", ["decode", "hardware"])
def test_rtc_programming_or_hardware_error_is_not_hidden(tmp_path, monkeypatch, fatal):
    r = simulate(tmp_path, monkeypatch, engine=rtc_controller, fatal=fatal, duration=3)
    assert r.report["outcome"] == "fault" and r.closed == ["hardware", "model"]
