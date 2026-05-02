# F/T Sensor Calibration SOP

Standard operating procedure for the UR5e wrist F/T sensor used by the
Compliant Insertion Studio. This is **not** a re-calibration — that's a
hardware operation done once per mount. This SOP covers session-level
**confidence checks** plus the small handful of bringup-time decisions
that, if wrong, contaminate every demo collected afterward.

---

## When to run the smoke test

Run `compliant_insertion_studio/shared/ft_smoke_test.py` at:

| Event | Why |
|---|---|
| Start of a session (after warm-up) | Establishes baseline; flags drift since last session |
| After any protective stop | The stop's force spike + recovery transient can shift the bias |
| After a physical bump to the sensor / gripper / payload | Mechanical impacts shift zero |
| When force-mode behavior looks "off" mid-session | Cheap to run, expensive to wrongly trust |
| Whenever you change the payload (e.g., swap gripper, attach/detach jig) | Different gravity load → different baseline |

Do **not** run during motion. Robot must be stationary, hands-free.

---

## Preconditions

1. **Warm-up window: F/T sensor powered for ≥ 10–30 minutes** before the
   first session of the day. Cold-start drift (1–3 N) can persist for
   tens of minutes and biases force-mode against contact rather than
   against true zero. If you started power minutes ago, wait.
2. **Robot is stationary** in a known steady pose. `move_home` (or
   equivalent) before running the smoke test. Vibration from a motion
   that just ended will inflate the drift reading.
3. **Hands off everything** — robot, gripper, payload, table. Operator
   load shows up as bias and contaminates the zero.
4. **Gripper is empty** for the canonical baseline. Smoke tests with
   payload attached are valid, but you'll need to relax the bias
   thresholds (`--bias-fmax`, `--bias-tmax`) by the gravity load of the
   payload — preferred is empty-gripper + canonical thresholds.
5. **Standard bringup is up**: `scaled_joint_trajectory_controller` is
   active, `force_torque_sensor_broadcaster` is publishing on
   `/force_torque_sensor_broadcaster/wrench`, and
   `/io_and_status_controller/zero_ftsensor` service is reachable.

---

## Running the test

Standalone:

```bash
cd /home/aaugus11/Documents/ros-mcp-server
source /opt/ros/humble/setup.bash
source ~/Desktop/ros2_ws/install/setup.bash
python3 compliant_insertion_studio/shared/ft_smoke_test.py
```

Default behavior: zero the sensor → settle 0.5 s → hold 5 s sampling
`/wrench` → report per-axis bias + drift → PASS / FAIL → exit code
`0` (pass), `1` (fail), `2` (infrastructure error).

Diagnostic flags:

| Flag | Effect | When to use |
|---|---|---|
| `--no-zero` | Skip the `zero_ftsensor` call, sample raw bias | Diagnose whether the *zero call itself* is broken vs the sensor is drifting |
| `--hold-s 10` | Longer hold window | Drift detection benefits from more samples |
| `--bias-fmax 5.0` | Relax force bias threshold | Payload attached |
| `--bias-tmax 1.0` | Relax torque bias threshold | Heavy / off-axis payload |

---

## Pass criteria

Default thresholds (gripper empty, post-zero):

| Quantity | Threshold | Source |
|---|---|---|
| Per-axis residual force bias \|F\| | < **2.0 N** | Empirically validated this session: post-zero baseline lands at 0.3–0.5 N; >2 N flags real drift, not noise |
| Per-axis residual torque bias \|T\| | < **0.3 Nm** | UR community guidance for low-force assembly |
| Per-axis force drift over hold | < **0.5 N/s** | Catches in-progress thermal drift before it ruins a session |

If the test passes, the sensor is trustworthy for the current session.
If it fails, **stop — do not collect demos or run inserts.** See
"Failure response" below.

---

## Failure response

The smoke test exits with code `1` when bias or drift exceeds threshold.
**Do not attempt to "push through" by relaxing thresholds.** Diagnose:

1. **Re-run with `--no-zero`** — if bias is still high, the sensor's raw
   reading is biased and the zero call wasn't applied. Check that the
   `io_and_status_controller` is actually active
   (`ros2 control list_controllers`).
2. **Check warm-up** — if the sensor was powered < 30 minutes ago, wait
   another 15 minutes and re-run.
3. **Check for residual operator/payload load** — confirm the gripper is
   empty, the operator is fully hands-off, and no cables are pulling on
   the wrist.
4. **Check for recent mechanical disturbance** — protective stops or
   collisions in the last few minutes may not have fully damped out;
   wait 1 minute and re-run.
5. **Check UR system state** — protective stop indicators on the
   pendant, robot in Local mode (force-mode service may behave oddly if
   pendant is in unexpected state).
6. **`set_payload` mismatch** — if the gripper assembly mass / center of
   mass set at bringup doesn't match reality, gravity compensation is
   wrong and zero won't converge to true zero. See "set_payload single
   source of truth" below.
7. **If all of the above check out and it still fails** — escalate.
   Possible causes: sensor cable damage, UR firmware quirk after a
   protective stop, mounting screw loosened. Do not proceed with demos
   or inserts; the data will be invalid.

---

## set_payload — single source of truth

The UR5e's `set_payload` (called once at bringup) tells the robot the
mass and center-of-mass of whatever is bolted to the wrist (gripper +
F/T sensor + any fixed jig). This affects:

- Gravity compensation in the F/T readings
- Force-mode behavior (the controller subtracts gravity load before
  computing compliance)
- Protective-stop thresholds (the robot knows what a "normal" wrench
  looks like for this payload)

**Convention for this project:** `set_payload` is called **once at
bringup**, with a value that matches the empty-gripper-on-wrist
configuration. Do **not** re-call it mid-session. Reason:

> UR Forum-documented bug (5.4.x): mid-session `set_payload` calls
> interact with `zero_ftsensor` such that the next zero is biased.
> See: <https://forum.universal-robots.com/t/5-4-x-rtde-set-payload-and-zero-ftsensor/5146>

If a payload changes (e.g., grasping a heavy part), do **not** call
`set_payload` to compensate — instead use force-mode's own gravity
compensation, or accept that the canonical bias check is now invalid
and run the smoke test with a relaxed `--bias-fmax`.

The bringup launch file is the single source for the payload value.
Anywhere else that wants to know "what's on the wrist" should read it
from there, not call `set_payload`.

---

## Recording smoke results in episode metadata

The episode wrapper (`compliant_insertion_studio/wrapper/compliant_insert.py`,
Phase 2 onward) runs this smoke test as its `PRE` phase precondition and
writes the result into the episode's `<csv>.meta.json` so post-mortem
analysis can correlate sensor health with episode outcome:

```json
{
  ...
  "smoke_test": {
    "result": "success",
    "bias": {"Fx": 0.31, "Fy": -0.18, "Fz": 0.42, "Tx": 0.011, "Ty": -0.024, "Tz": 0.003},
    "drift_per_s": {"Fx_per_s": 0.012, "Fy_per_s": -0.008, "Fz_per_s": 0.041},
    "thresholds": {"bias_fmax_N": 2.0, "bias_tmax_Nm": 0.3, "drift_max_N_per_s": 0.5}
  }
}
```

A failed smoke test aborts `PRE` — the wrapper does not enter `HOVER`.

---

## Bringing up the F/T-related services

For reference (these run as part of standard UR ROS2 bringup; included
here so a fresh operator can verify):

```bash
# Verify the broadcaster is active and publishing
ros2 control list_controllers | grep force_torque
# Expected: force_torque_sensor_broadcaster ... active

# Verify the wrench topic
ros2 topic hz /force_torque_sensor_broadcaster/wrench
# Expected: ~500 Hz

# Verify the zero service exists
ros2 service list | grep zero_ftsensor
# Expected: /io_and_status_controller/zero_ftsensor
```

If any of these are missing, the smoke test will fail with exit code 2
(infrastructure error). Fix the bringup before re-running.
