"""Dynamical and fault tests; no CAN, camera, or real robot construction."""
import math
from types import SimpleNamespace

import pytest

from experiments.b601.runtime.hardware.zero_servo import (ZeroGripperServo, corrected_gripper_torque,
                         zero_gripper_session, ZERO_PRECISION_DEG)


def simulate(use_servo, *, stop_deg=None, dt=.02):
    # Constant opposing load makes native PD settle at the observed 2.483 deg.
    initial = math.radians(2.483)
    load = 12 * initial
    position, velocity = initial, 0.0
    servo, commands = ZeroGripperServo(), []
    for step in range(round(8 / dt)):
        now = step * dt
        native = max(-1., min(1., -12 * position - .05 * velocity))
        torque = servo.correct(native, 0., position, velocity, 1., now) if use_servo else native
        commands.append(torque)
        # Damped low-speed plant; this is a regression model, not a robot fit.
        velocity = (torque + load) / 1.5
        position += velocity * dt
        if stop_deg is not None and position < math.radians(stop_deg):
            position, velocity = math.radians(stop_deg), 0.
        arrival = servo.observe(math.degrees(position), now, ZERO_PRECISION_DEG)
        if arrival['stalled']:
            return math.degrees(position), commands, now, arrival
    return math.degrees(position), commands, now, arrival


@pytest.mark.parametrize('dt', [.01, .02, 1/30, .05])
def test_feedback_correction_removes_native_pd_residual_without_raising_cap(dt):
    baseline, _, _, _ = simulate(False, dt=dt)
    actual, commands, _, arrival = simulate(True, dt=dt)
    assert baseline == pytest.approx(2.483)
    assert abs(actual) < ZERO_PRECISION_DEG
    assert not arrival['stalled']
    assert max(map(abs, commands)) <= 1.0


def test_mechanical_stop_is_failure_instead_of_success_or_endless_push():
    actual, commands, elapsed, arrival = simulate(True, stop_deg=2.483)
    assert actual == pytest.approx(2.483)
    assert arrival['stalled']
    assert 2 <= elapsed <= 2.05
    assert max(map(abs, commands)) <= 1.0


def test_saturation_does_not_wind_up_and_sign_reversal_clears_correction():
    servo = ZeroGripperServo()
    for i in range(100):
        assert servo.correct(-1, 0, 1, 0, 1, i*.02) == -1
    assert servo.integral == 0
    servo.integral = -.4
    servo.correct(.1, 0, -.01, 0, 1, 2.)
    assert servo.integral >= 0


def test_scheduler_gap_does_not_integrate_a_large_correction():
    servo = ZeroGripperServo()
    servo.correct(-.5, 0, .04, 0, 1, 0.)
    servo.correct(-.5, 0, .04, 0, 1, .02)
    assert servo.integral < 0
    assert servo.correct(-.5, 0, .04, 0, 1, 2.) == -.5
    assert servo.integral == 0


@pytest.mark.parametrize('field,value', [('native', float('nan')), ('position_rad', float('inf')),
                                       ('velocity_rad_s', float('nan')), ('limit', 0),
                                       ('limit', float('nan')), ('target_rad', .01)])
def test_invalid_inputs_abort_before_returning_a_motor_command(field, value):
    args=dict(native=-.5, target_rad=0., position_rad=.04, velocity_rad_s=0., limit=1., now=0.)
    args[field]=value
    with pytest.raises(ValueError):
        ZeroGripperServo().correct(**args)


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt, BaseException])
def test_faults_restore_the_native_policy_path(error):
    robot = SimpleNamespace()
    with pytest.raises(error):
        with zero_gripper_session(robot):
            raise error('fault')
    assert not hasattr(robot, '_homing_zero_servo')
    # No configuration or motor access at all outside homing.
    assert corrected_gripper_torque(robot, None, 123, .125) == .125


def test_stationarity_is_required_and_jitter_cannot_hide_a_stall():
    servo = ZeroGripperServo()
    assert not servo.observe(.4, 0, .5)['stationary']
    assert not servo.observe(-.4, .02, .5)['stationary']
    servo = ZeroGripperServo()
    for i in range(102):
        outcome = servo.observe(2.483 + (i%2)*.022, i*.02, .5)
    assert outcome['stalled']


def test_driver_bridge_uses_existing_state_and_restores_on_success(monkeypatch):
    import experiments.b601.runtime.hardware.zero_servo as module
    state=SimpleNamespace(pos=math.radians(2.483), vel=0.)
    motor=SimpleNamespace(get_state=lambda:state)
    robot=SimpleNamespace(motors={'gripper':motor}, config=SimpleNamespace(
        gripper_mit_hold_torque_limit=1., gripper_mit_torque_limit=3.5))
    clock=iter([0., .02])
    monkeypatch.setattr(module.time, 'monotonic', lambda:next(clock))
    with zero_gripper_session(robot):
        first=corrected_gripper_torque(robot,motor,0.,-.52)
        second=corrected_gripper_torque(robot,motor,0.,-.52)
        assert -1 <= second < first == -.52
    assert not hasattr(robot,'_homing_zero_servo')
