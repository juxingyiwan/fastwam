"""Real B601 nominal geometry, dense replay and fake-motor lifecycle tests."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from fastwam.datasets.lerobot3.ee_delta import EEDeltaCodec
from experiments.b601.runtime.contracts import HardwareError
from experiments.b601.runtime.fastwam_hardware import FastWAMHardware
from experiments.b601.runtime.hardware import b601, front_home
from .test_shutdown import fake_arm


# Nominal transform chain from the deployed B601 robot.urdf; no mesh dependency.
CHAIN = [
    ('-0.00034283 -0.00098683 0.075', '0 0 0', '0 0 -1'),
    ('0.020343 0.027237 0.07', '-1.5708 0 0', '0 0 1'),
    ('-0.236 0 0', '0 0 0', '0 0 -1'),
    ('0.228 -0.072746 0.0045', '0 0 0', '0 0 -1'),
    ('0.087 -0.048 -0.03075', '-1.5708 0 0', '0 0 -1'),
    ('0.0365 0 0.048', '0 1.5708 0', '0 0 -1'),
    ('0 0 0.16621', '3.1416 -1.5708 0', None),
]
LAST_RUN = np.array([1.9117069, 105.2961502, 86.1794205, -62.3822632,
                     18.0180931, 19.3804836, 112.9868317])


@pytest.fixture
def codec(tmp_path):
    config = json.loads((Path(__file__).resolve().parents[2] / 'configs/ee_delta.json').read_text())
    links = ['base_link', *[f'link{i}' for i in range(1, 7)], 'gripper_end']
    root = ET.Element('robot', name='b601_geometry_test')
    for name in links:
        ET.SubElement(root, 'link', name=name)
    for i, (xyz, rpy, axis) in enumerate(CHAIN):
        joint = ET.SubElement(root, 'joint', name=f'joint{i+1}' if axis else 'j_gripper_end',
                              type='revolute' if axis else 'fixed')
        ET.SubElement(joint, 'origin', xyz=xyz, rpy=rpy)
        ET.SubElement(joint, 'parent', link=links[i])
        ET.SubElement(joint, 'child', link=links[i+1])
        if axis:
            ET.SubElement(joint, 'axis', xyz=axis)
    path = tmp_path / 'robot.urdf'
    ET.ElementTree(root).write(path)
    return EEDeltaCodec(path, config)


def limits(codec):
    return np.asarray(codec.config['driver_motor_limits_deg'], float)


def test_last_run_avoids_595mm_native_sweep_without_joint_or_velocity_jumps(codec):
    plan = front_home.plan_home(codec, LAST_RUN, limits(codec))
    assert not plan.native
    assert 0.594 < plan.native_max_x_m < 0.600  # Includes the inter-sample upper bound.
    u = np.linspace(0, 1, 10001)
    progress = u**3 * (10 + u * (-15 + 6*u))
    q = plan.joints(progress)
    xyz = front_home.tool_positions(codec, q)
    assert np.max(xyz[:, 0]) < 0.366
    assert np.max(xyz[:, 0]) <= plan.planned_max_x_m <= 0.47
    assert np.min(xyz[:, 2]) >= xyz[0, 2] - 1e-9
    assert np.allclose(q[0], LAST_RUN[:6]) and np.array_equal(q[-1], np.zeros(6))
    assert np.all(q >= limits(codec)[:6, 0]) and np.all(q <= limits(codec)[:6, 1])
    dt = plan.duration_s / (len(u)-1)
    velocity = np.diff(q, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    assert np.max(abs(velocity)) <= front_home.MAX_SPEED_DEG_S + 1e-6
    assert np.max(abs(acceleration)) <= front_home.MAX_ACCEL_DEG_S2 + 1e-3
    assert np.max(abs(velocity[[0, -1]])) < 0.001


def test_already_home_preserves_native_path_and_gripper_position(codec):
    measured = np.array([.28, -.0879, .13, -.0879, -.04, .77, 200.])
    plan = front_home.plan_home(codec, measured, limits(codec))
    assert plan.native
    assert plan.planned_max_x_m < .31
    assert measured[6] == 200


def test_outside_start_retreats_instead_of_rejecting_homing(codec):
    measured = np.array([0, 105, 86, 0, 0, 0, 100.])
    plan = front_home.plan_home(codec, measured, limits(codec))
    assert plan.start_x_m > .5
    points = front_home.native_path(measured[:6]) if plan.native else plan.joints(np.linspace(0, 1, 10001))
    xyz = front_home.tool_positions(codec, points)
    assert np.max(xyz[:, 0]) <= xyz[0, 0] + 1e-9
    assert xyz[-1, 0] < .47
    assert np.all(np.diff(xyz[:, 0])[xyz[:-1, 0] > .47] <= 1e-8)


@pytest.mark.parametrize('q', [
    [41.8836584, 161.4004051, 160.9427210, -59.3668149, -14.8756184, -55.6944510, 0],
    [-39.6038345, 165.2422165, 142.0805287, 25.6677936, -68.6736037, -30.5507862, 0],
])
def test_coordinated_home_has_no_hidden_forward_peak(codec, q):
    plan = front_home.plan_home(codec, q, limits(codec))
    assert not plan.native
    points = plan.joints(np.linspace(0, 1, 20001))
    x = front_home.tool_positions(codec, points)[:, 0]
    assert max(x) <= max(.47, x[0]) + 1e-7
    assert max(x) <= plan.planned_max_x_m
    if x[0] > .47:
        assert np.all(np.diff(x)[x[:-1] > .47] <= 1e-7)


def test_start_in_clearance_band_can_return_continuously(codec):
    measured = np.array([0., 77, 77, 0, 0, 0, 0])
    plan = front_home.plan_home(codec, measured, limits(codec))
    assert .47 < plan.start_x_m < .5 and plan.native


def test_start_just_inside_clearance_line_is_not_rejected_by_sample_padding(codec):
    angle = np.rad2deg(np.arccos(1 - (.4699 - .30171017) / .236))
    measured = np.array([0., angle, angle, 0, 0, 0, 0])
    plan = front_home.plan_home(codec, measured, limits(codec))
    assert .469 < plan.start_x_m < .47
    assert plan.native and plan.planned_max_x_m <= .5


@pytest.mark.parametrize('bad', ['nan', 'frame', 'unreachable_zero', 'limits'])
def test_invalid_geometry_or_input_fails_before_motion(codec, bad):
    measured = LAST_RUN.copy()
    bounds = limits(codec)
    if bad == 'nan':
        measured[0] = np.nan
    elif bad == 'frame':
        codec.config['base_link'] = 'wrong'
    elif bad == 'limits':
        bounds[0] = [1, 100]
    else:
        codec.fk.chain[0][0][0, 3] += 2
    with pytest.raises(HardwareError):
        front_home.plan_home(codec, measured, bounds)


def set_up_hardware(cfg, codec, monkeypatch):
    hw, history = fake_arm(cfg, position=LAST_RUN.copy())
    hw.__class__ = FastWAMHardware
    hw.decoder = SimpleNamespace(codec=codec)
    hw.limits = limits(codec)
    hw.last_target = LAST_RUN.copy()
    hw.last_target[6] = 111.0
    hw.shutdown_cfg = replace(hw.shutdown_cfg, home_timeout_s=65., settle_s=.25)
    clock = SimpleNamespace(now=100.)
    def sleep(dt):
        clock.now += dt
    timer = SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now, sleep=sleep)
    monkeypatch.setattr(front_home, 'time', timer)
    monkeypatch.setattr(b601, 'time', timer)
    return hw, history, clock


def test_full_shutdown_reaches_native_home_and_gripper_zero_with_no_boundary_abort(cfg, codec, monkeypatch):
    hw, history, clock = set_up_hardware(cfg, codec, monkeypatch)
    events = []
    hw.event = lambda kind, **kw: events.append((kind, kw))
    result = hw.close()
    assert result['homing'] == 'completed'
    assert result['disable_reason'] == 'home_verified'
    assert history.count('home') == 1
    assert sum(s.startswith('disable:') for s in history) == 7
    targets = np.asarray(hw.test_targets)
    assert np.max(front_home.tool_positions(codec, targets[:, :6])[:, 0]) < .47
    assert np.max(abs(np.diff(targets[:, :6], axis=0))) <= front_home.MAX_SPEED_DEG_S * front_home.PERIOD_S + 1e-6
    assert np.allclose(targets[-1], np.zeros(7))
    assert targets[0, 6] == 111.0  # Preserve the policy's existing gripper hold.
    assert any(kind == 'home_forward_complete' for kind, _ in events)
    assert not hw._home_active
    assert clock.now < 130


@pytest.mark.parametrize('fault', ['feedback', 'command'])
def test_forward_path_fault_retains_disable_and_never_restarts_native_home(cfg, codec, monkeypatch, fault):
    hw, history, _ = set_up_hardware(cfg, codec, monkeypatch)
    original = hw.robot.send_action
    count = 0
    def send(action):
        nonlocal count
        count += 1
        if count == 4:
            if fault == 'feedback':
                def fail():
                    raise HardwareError('injected feedback fault')
                hw.feedback.refresh = fail
            else:
                raise HardwareError('injected motor command fault')
        return original(action)
    hw.robot.send_action = send
    with pytest.raises(HardwareError, match='injected'):
        hw.close()
    assert 'home' not in history
    assert sum(s.startswith('disable:') for s in history) == 7
    assert not hw._home_active


def test_initial_feedback_exception_restores_homing_guard(cfg, codec, monkeypatch):
    hw, _, _ = set_up_hardware(cfg, codec, monkeypatch)
    path = front_home.plan_home(codec, LAST_RUN, hw.limits)
    def fail(**kw):
        raise HardwareError('initial feedback failed')
    hw._home_feedback = fail
    with pytest.raises(HardwareError, match='initial feedback'):
        front_home.execute_home(hw, codec, path)
    assert not hw._home_active


def test_native_safe_route_does_not_enter_coordinated_executor(cfg, codec, monkeypatch):
    hw, history, _ = set_up_hardware(cfg, codec, monkeypatch)
    hw.robot.send_action({name+'.pos': 0. for name in b601.JOINTS})
    import experiments.b601.runtime.fastwam_hardware as bridge
    monkeypatch.setattr(bridge, 'execute_home', lambda *a: pytest.fail('unexpected extra arm path'))
    hw._native_home()
    assert history.count('home') == 1


def test_no_feasible_path_never_dispatches_outward_native_fallback(cfg, codec, monkeypatch):
    class BlockedFK:
        chain = codec.fk.chain
        def __call__(self, joints):
            joints = np.asarray(joints)
            pose = np.broadcast_to(np.eye(4), (*joints.shape[:-1], 4, 4)).copy()
            pose[..., 0, 3] = .30 + .25 * np.sin(2 * joints[..., 0])
            pose[..., 2, 3] = .2
            return pose
    hw, history, _ = set_up_hardware(cfg, codec, monkeypatch)
    hw.robot.send_action({name+'.pos': (90. if i == 0 else 0.) for i,name in enumerate(b601.JOINTS)})
    hw.test_targets.clear()
    codec.fk = BlockedFK()
    with pytest.raises(HardwareError, match='No continuous inward home'):
        hw.close()
    assert not hw.test_targets and 'home' not in history
    assert sum(s.startswith('disable:') for s in history) == 7


def test_unequal_motor_lag_slows_progress_and_still_finishes(cfg, codec, monkeypatch):
    hw, history, clock = set_up_hardware(cfg, codec, monkeypatch)
    position = LAST_RUN.copy()
    observed = []
    hw.feedback.refresh = hw.feedback.snapshot = lambda: (position.copy(), {})
    def send(action):
        target = np.array([action[j+'.pos'] for j in b601.JOINTS]) * hw.directions
        hw.test_targets.append(target.copy())
        position[:] += (target - position) * [.5, .08, .2, .6, .6, .6, 1.]
        observed.append(position.copy())
        return dict(zip((j+'.pos' for j in b601.JOINTS), target, strict=True))
    hw.robot.send_action = send
    result = hw.close()
    assert result['homing'] == 'completed' and history.count('home') == 1
    assert np.max(front_home.tool_positions(codec, np.array(observed)[:, :6])[:, 0]) < .5
    assert clock.now < 165
    assert not hw._home_active
