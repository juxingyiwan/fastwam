## Native homing baseline restored (2026-10-16)

The native baseline follows FastWAM-SGTTT and the pinned Seeed B601 1.0.4 driver:
first CAN 1/4/5/6, then CAN 2/3, followed by gripper closure at 60 physical
degrees/s and measured arrival of all seven joints (arm 2 degrees, gripper 0.5 degrees, stable 0.25 s).
No calibration or motor-zero register is written. Startup failure aborts the
policy; shutdown failure is reported before attempted motors are disabled and
resources close. Existing feedback, temperature, CAN and timeout guards remain.

The shared TCP envelope, retraction planner, limited-return policy continuation
and spatial powered-hold branch are no longer part of the deployed homing path.
Old workspace JSON files do not configure this native path. Native trajectories
do not provide obstacle clearance. This rollback was tested with fake motors;
it does not establish the cause of the previously observed physical drop.

## Historical acceptance workaround (superseded by active refinement below)

Startup and shutdown use a 3-degree gripper arrival tolerance in physical motor
feedback units; the six arm joints retain their 2-degree tolerance and the
0.25-second continuous settling check. On the empty, fully closed lab gripper,
feedback measured 2.461-2.483 degrees. The target remains zero; this tolerance
does not recalibrate the motor or change the native trajectory, speed or gains.

`home_progress` records the native-arm, gripper-close and zero-verification
stages. On failure, `home_failed` and the run report retain the last completed
feedback read, thresholds and joints outside tolerance. No extra CAN read is
made after a timeout. A homing timeout prevents policy startup, inhibits another
homing attempt and retains the existing disable/cleanup path.

## Active gripper zero refinement (2026-10-16)

The 3-degree acceptance change did not remove the physical residual. The native
arm trajectory and gripper closing ramp now lead into an active zero-refinement
phase: the native PD torque receives a bounded integral correction based on
actual monotonic time. Ki is 6 Nm/(rad*s); correction slew is limited to 0.2 Nm/s.
The total torque never exceeds the smaller existing native hold/moving torque
limit (1 Nm with the pinned driver). No position offset, negative zero target,
zero-register write, gain change or policy-time correction is introduced.

Success requires gripper feedback within min(configured tolerance, 0.5 degrees),
estimated gripper speed at most 2 degrees/s, and the configured continuous
settling interval. Arm arrival remains unchanged. A 2.48-degree stop is a failure,
even when the legacy arrival tolerance is 3 degrees. At least 0.1 degree of net
progress toward zero must occur within each 2-second progress window; otherwise
homing aborts with measured position and commanded torque diagnostics. Refinement
is capped at 10 seconds and the existing shorter home timeout still applies.
All compensation is removed on completion, timeout, feedback fault or interrupt.

Offline tests include a damped load model where native PD stays at 2.483 degrees,
a blocked mechanical stop, multiple control rates, torque saturation, telemetry
failure and cleanup. This model is a software regression check, not a measured
robot model. Actual return accuracy requires an attended hardware trial. If the
calibrated zero lies beyond the mechanical stop, this controller reports failure;
it does not relabel the feedback as zero or continue pushing indefinitely.

## FastWAM forward plane, first version (2026-10-16)

FastWAMHardware now checks the complete native arm path before dispatch. The
plane is base_link +X = 0.50 m, using the existing model bundle's gripper_end FK
point, with 0.03 m of planning clearance. This is a nominal TCP bound: the current
codec explicitly marks physical axes, zero and TCP calibration as unverified.
The clearance is not a certified fingertip envelope or collision guarantee.

If the native route fits, the original native routine runs unchanged. Otherwise,
a pure planner first tries all-six-joint coordinated return, then a bounded
family of monotone joint-progress curves. Whole paths are checked with FK and
conservative bounds between samples; new routes also avoid dropping the TCP more
than 5 mm below their lower endpoint. This is not a table/self-collision planner.
The selected path is streamed with quintic timing (40 deg/s, 40 deg/s^2 budgets),
smoothly slowed by tracking lag. The coordinated speed ceiling was raised from
25 to 40 deg/s on 2026-10-17; the acceleration budget and geometric clearance
are unchanged. Acceleration limits can keep short moves below the speed ceiling.
There is no boundary-triggered stop or exception in the streaming loop. It then
hands over near zero to the original native arm routine, gripper closing ramp
and 0.5-degree gripper refinement.

An initial position already in the clearance band or beyond 0.50 m is allowed to
retreat continuously with non-increasing sampled forward excursion until it is
inside the band. No arbitrary-start reachability guarantee is made: if no route
is found, failure is reported before any arm trajectory is sent; the old outward
route is never used as an unsafe fallback. Existing motor, feedback, temperature,
timeout, disable and cleanup protections remain in force. Policy commands,
model decoding, calibrated zero and installed driver sources are unchanged.

Events home_forward_plan/progress/complete record the selected route, predicted
peak, measured TCP X, limit and progress. The first measured robot trial must
check these values and physical clearance. Historical FK replay and fake-motor
tests establish software behavior only, not measured hardware tracking accuracy.
