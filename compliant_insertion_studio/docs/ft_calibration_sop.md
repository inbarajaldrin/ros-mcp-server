# F/T Sensor Calibration SOP

Three-layer F/T calibration procedure for the Compliant Insertion Studio on
UR5e + OnRobot RG2. Read **all three sections** the first time — the layers
work together and skipping foundational calibration is the most common
failure mode in force-controlled assembly.

See `.planning/codebase/CONVENTIONS.md` §"When to use which calibration layer"
for the at-a-glance summary, and `_references/articles/ft_payload_calibration_math.md`
for the mathematical derivation.

---

## TL;DR — what to run when

| Trigger | What to run |
|---|---|
| New gripper / new jig / sensor remount / orientation-dependent bias observed | **Foundational** — `ft_calibration.py` |
| Start of a session (first time today, or after >4 hours since last) | **Session smoke** — `ft_smoke_test.py` |
| Just before any force-mode entry inside the wrapper | **Per-pose** — `zero_ftsensor` (wrapper does this automatically in PRE phase) |
| After a protective stop | Session smoke (and re-foundational if smoke fails) |
| After physical bump / collision | Session smoke (and re-foundational if smoke fails) |
| When force mode misbehaves mid-session (wrong direction, drift, oscillation) | Session smoke (and re-foundational if smoke fails) |

---

## Layer 1 — Foundational calibration (per-mount, one-time)

### What it is

Recovers the **payload mass + center of gravity** of whatever is bolted to the
wrist (gripper + F/T sensor + any jig), plus the **6-axis F/T sensor bias**.
The mass + CoG values feed the UR controller's gravity compensation via
`set_target_payload(mass, cog)`. Without this, the F/T sensor's reading
includes an orientation-dependent gravity bias that **no per-pose
`zero_ftsensor` can fully correct** — the bias re-emerges as soon as you
move to a different EE orientation.

If you observed today's symptom (zero in pose A, then move to pose B and the
bias is back), foundational calibration is the fix — the rest of the SOP
won't help until this is right.

### When to run

- **First time** for any new gripper / wrist sensor mount
- **Whenever the payload changes** in a persistent way: gripper swap, jig
  added or removed, sensor remounted with different mounting screws, or
  torque on the mounting bolts changed
- **When you observe orientation-dependent bias** in the session smoke
  (smoke passes in the home pose but bias appears at other poses)

### How to run

```bash
cd /home/aaugus11/Documents/ros-mcp-server
source /opt/ros/humble/setup.bash
source ~/Desktop/ros2_ws/install/setup.bash

# 1. Verify the pose plan first (no robot motion)
python3 compliant_insertion_studio/shared/ft_calibration.py \
  --gripper-id onrobot_rg2_with_camera \
  --dry-run

# 2. Visually verify each pose is reachable. (Optional: Freedrive through
#    them on the pendant first to confirm no collisions / joint limits.)

# 3. Run for real (~75 seconds for 8 poses)
python3 compliant_insertion_studio/shared/ft_calibration.py \
  --gripper-id onrobot_rg2_with_camera \
  --expected-mass-kg 1.05      # whatever you think the gripper weighs, for sanity check
```

### Preconditions (before Layer 1)

1. **F/T sensor warm-up**: ≥ 10–30 minutes powered on. Cold-start drift
   (1–3 N) over the first 30 minutes will show up as bad LSQ residuals
   and bias values that won't match next session's. **Skip this and the
   calibration is invalid.**
2. **Gripper in target configuration**: with the jig you'll use during
   normal operation. **No part in the gripper** for the canonical "empty
   gripper" calibration — payload + part is a different calibration done
   only if you specifically need part-attached force-mode behavior (rare).
3. **Workspace clear**: the calibration moves the wrist through 8 varied
   orientations. No obstructions in a ~50 cm radius of the wrist.
4. **Hands off the robot for the entire run** (~75 seconds for 8 poses).
   Operator load contaminates every measurement and silently corrupts the
   recovered mass + CoG. The script announces "HANDS OFF" and waits 3
   seconds before starting; honor this.
5. **Pendant in Local mode is fine** — no Remote-mode services are called.
6. **Standard ROS2 bringup is up**: `scaled_joint_trajectory_controller`
   active, `force_torque_sensor_broadcaster` publishing on
   `/force_torque_sensor_broadcaster/wrench` at ~500 Hz.

### What the script does

1. Moves robot through 8 calibration poses (joint-space, varied wrist
   orientations chosen so the gravity vector in F/T frame spans 3D)
2. At each pose: settle 1.5 s, sample `/wrench` for 1 s (averaged), read
   TCP orientation from `/tcp_pose_broadcaster/pose`
3. Builds 6×10 measurement matrix per pose per Kubus 2007 (algorithm in
   `_references/articles/ft_payload_calibration_math.md`)
4. Stacks all 48 equations and solves least-squares for
   `[m, m·cx, m·cy, m·cz, FBx..z, TBx..z]`
5. Recovers physical mass, CoG, and bias
6. Writes `compliant_insertion_studio/configs/ft_calibration_<gripper_id>_<YYYYMMDD>.yaml`
7. Prints the `set_target_payload(...)` line you'll paste into bringup

### Output — what to do with it

The script prints something like:

```
=== NEXT STEP — paste into your UR bringup launch ===
  set_target_payload(1.0823, [0.0014, -0.0025, 0.0521])
Then restart the bringup once. See docs/ft_calibration_sop.md.
```

**Paste this into your UR bringup launch file**, in the `set_target_payload`
URScript call (or via the URCap installation file's payload setting if your
bringup uses the URCap path). Restart bringup once. After this, the
controller's gravity compensation is correct and `zero_ftsensor` becomes
a quick bias subtraction (not a workaround for bad gravity comp).

The YAML is committed to the project (per-gripper history). The bringup
launch update is committed to your bringup repo.

### Pass criteria

The script self-checks and warns on any of these:

| Check | Threshold | Why |
|---|---|---|
| Mass within ±20% of `--expected-mass-kg` | If you provided `--expected-mass-kg` | Catches gross errors (e.g., F/T sensor unplugged, poses didn't span the sphere) |
| Max force residual | < 0.5 N per axis | Indicates LSQ fit quality. Higher = poses too similar or unmodeled motion |
| Max torque residual | < 0.05 Nm per axis | Same as above for torques |
| Matrix rank | == 10 | All 10 parameters recoverable. Lower means poses too similar |
| Condition number | < 1000 | Pose distribution quality. Higher = poorly conditioned (some parameters poorly estimated) |

If any check fails: the script still writes the YAML and prints the
recommended values, but with `result: success_with_warnings`. **Do not
paste the values into bringup if there are warnings.** Investigate (rerun
preconditions, change pose set, check sensor health).

### When the calibration script fails

| Failure | Diagnosis | Fix |
|---|---|---|
| "Pose unreachable" | Joint config exceeds limits or hits a singular configuration for your URDF | Edit `CALIBRATION_POSES_RAD` in the script (or `--num-poses 4` to use only the first 4 simpler poses) |
| "Too few wrench samples" | `/wrench` topic slow or stalled | Check `ros2 topic hz /force_torque_sensor_broadcaster/wrench` reads ~500 Hz |
| "Mass ~0" | Pose set is rank-deficient (gravity vector pointing same way in all poses) or wrench is identically zero | Check the script ran the full pose set; verify F/T sensor is connected |
| Mass differs > 20% from expected | Bad pose set, sensor mounting strain, operator interference, or expected value is off | Re-check `--expected-mass-kg`, re-run with operator fully hands-off, inspect mounting |
| Residuals high | Robot moving during sample (motion not damped), pose too close to a singularity, or the wrist payload isn't actually rigid (cables flopping) | Increase `--settle-s`, secure cables, avoid singular configs |

### Customizing the pose set

The 8 default poses are conservative and wrist-only — they assume the
shoulder/elbow stay at `[0, -π/2, π/2]`. If your workspace requires a
different shoulder config (e.g., to avoid a fixture), edit
`CALIBRATION_POSES_RAD` in `ft_calibration.py` and verify with `--dry-run`.
The math doesn't care which joint configs you use as long as the **gravity
vector in the F/T frame spans at least 3 linearly independent directions**
across poses (practically: vary `wrist_1` and `wrist_2` significantly,
not just `wrist_3`).

A future improvement (not in v1) is to load poses from a YAML so per-
workspace customization doesn't require Python edits. See REQUIREMENTS.md
CAL-04 — pose set parameterized in YAML.

### Alternative: UR's URCap "Measure" wizard

UR's PolyScope teach pendant has a built-in payload Measure wizard
(Installation tab → General → Payload → Measure → 4 poses). It computes
the same mass + CoG with a similar least-squares fit, just with fewer
poses (less precision) and via Freedrive instead of script motion. Use
this if you can't bring up ROS at calibration time. Output is interchangeable
with our script's — both give numbers for `set_target_payload`.

See `_references/articles/ur_polyscope_payload_measure_wizard.md`.

---

## Layer 2 — Session F/T smoke test (per-session)

### What it is

A 10-second confidence check that the F/T sensor is still trustworthy
this session: zeros the sensor, holds 5 s, samples `/wrench`, computes
per-axis residual bias and drift rate, reports PASS / FAIL.

This **assumes foundational calibration is correct**. If smoke passes,
the wrapper can safely enter force mode. If smoke fails, foundational
calibration is suspect — re-run Layer 1 before any force-mode work.

### When to run

| Event | Why |
|---|---|
| Start of a session (after F/T warm-up) | Establishes baseline; flags drift since last session |
| After any protective stop | Stop's force spike + recovery transient can shift bias |
| After a physical bump to the sensor / gripper / payload | Mechanical impacts shift zero |
| When force-mode behavior looks "off" mid-session | Cheap to run, expensive to wrongly trust |

Do **not** run during motion. Robot must be stationary, hands-free.

### Preconditions

1. F/T sensor warm-up complete (≥ 10–30 min powered on) — same as Layer 1
2. **Foundational calibration is current** for this gripper / mount
3. **Robot in a steady neutral pose** — `move_home` first if uncertain
4. **Hands off everything** — robot, gripper, payload, table
5. **Gripper is empty** for the canonical baseline (or relax `--bias-fmax`
   if running with a payload attached)
6. Standard ROS2 bringup live (`zero_ftsensor` service reachable)

### How to run

```bash
# Standalone
python3 compliant_insertion_studio/shared/ft_smoke_test.py

# Diagnostic options
python3 compliant_insertion_studio/shared/ft_smoke_test.py --no-zero        # raw bias readout
python3 compliant_insertion_studio/shared/ft_smoke_test.py --hold-s 10       # longer hold for drift
python3 compliant_insertion_studio/shared/ft_smoke_test.py --bias-fmax 5.0   # relax for payload
```

The wrapper (`compliant_insertion_studio/wrapper/compliant_insert.py`,
Phase 2) runs this automatically as the PRE phase precondition before
each episode and writes the result into the episode's `<csv>.meta.json`.

### Pass criteria

| Quantity | Threshold |
|---|---|
| Per-axis residual force `|F|` | < 2.0 N |
| Per-axis residual torque `|T|` | < 0.3 Nm |
| Per-axis force drift over 5 s window | < 0.5 N/s |

Exit code 0 on PASS, 1 on FAIL (any threshold exceeded), 2 on infrastructure
error (topic/service missing).

### Failure response

A failed smoke test means **stop — do not collect demos or run inserts.**
Diagnose:

1. **Re-run with `--no-zero`** — if bias is still high, the zero call wasn't
   applied. Check `ros2 control list_controllers` shows `io_and_status_controller`
   active.
2. **Check warm-up window** — sensor < 30 min powered? Wait 15 min, retry.
3. **Check operator/payload load** — confirm gripper empty, hands off, no
   cables pulling on wrist.
4. **Recent disturbance?** — protective stops or bumps in the last few
   minutes may not have damped out. Wait 1 minute, retry.
5. **Check UR system state** — protective stop indicators on pendant?
   Pendant in unexpected mode?
6. **Foundational drift?** — re-run Layer 1 (foundational calibration). If
   the gripper has been swapped/changed since last calibration, this is
   the issue.
7. **All of the above clean and still failing?** — escalate. Possible:
   sensor cable damage, UR firmware quirk, mounting screw loosened. Do not
   proceed.

---

## Layer 3 — Per-pose `zero_ftsensor` (inside the wrapper)

### What it is

A `std_srvs/srv/Trigger` call to `/io_and_status_controller/zero_ftsensor`
that sets the *current* sensor reading as the new zero. Single-pose bias
subtraction. Fast, cheap, automatic. Done by the episode wrapper in its
PRE phase, immediately after the smoke test passes and immediately before
entering force mode.

### When to run

- Inside the wrapper, every episode, in PRE phase, after the smoke test
- **Never as a substitute for foundational calibration** — if you find
  yourself wanting to re-zero between every motion, the foundational
  calibration is wrong and you need Layer 1, not more Layer-3 calls

### How

The wrapper handles this automatically. Manual invocation for debugging:

```bash
ros2 service call /io_and_status_controller/zero_ftsensor std_srvs/srv/Trigger
```

After the call, wait ≥ 0.5 s before reading `/wrench` so the broadcaster
publishes post-zero samples (settle window per stash convention).

### What zero does, and what it doesn't

`zero_ftsensor` subtracts whatever the sensor is reading right now and
makes that the new zero. Does **not** know about payload mass, gravity,
or anything physical. Just bias subtraction.

If the sensor is reading 19 N because of gripper gravity in the current
pose, zero subtracts 19 N. Move to a different pose where gripper gravity
shows up as 15 N in some other axis, and now you read 4 N in that axis
— the zero is "wrong" for the new pose. **This is what foundational
calibration fixes**: with correct payload set in the URDF / bringup, the
controller subtracts gripper gravity automatically per-pose, leaving only
true contact forces in the sensor reading.

---

## set_target_payload — single source of truth

The mass + CoG produced by Layer 1 calibration goes into UR's
`set_target_payload(mass, cog)` URScript call, which should be present
**exactly once** in your bringup launch file. Do **not** call
`set_target_payload` (or the deprecated `set_payload`) mid-session.

> **UR Forum-documented bug (5.4.x):** mid-session `set_payload` calls
> interact with `zero_ftsensor` such that the next zero is biased.
> Source: <https://forum.universal-robots.com/t/5-4-x-rtde-set-payload-and-zero-ftsensor/5146>
>
> The newer `set_target_payload()` is the recommended replacement
> (the old name is deprecated). Both have the same mid-session
> interaction risk — keep the call at bringup only.

If a payload changes (e.g., grasping a heavy part), do **not** call
`set_target_payload` to compensate — instead either accept the
canonical bias-check is now invalid, or run Layer 2 smoke with relaxed
thresholds.

---

## Recording calibration in episode metadata

The episode wrapper writes the smoke test result and the foundational
calibration's source YAML path into each episode's `<csv>.meta.json`:

```json
{
  ...
  "foundational_calibration": {
    "yaml_path": "compliant_insertion_studio/configs/ft_calibration_onrobot_rg2_with_camera_20260501.yaml",
    "mass_kg": 1.0823,
    "cog_xyz_m": [0.0014, -0.0025, 0.0521],
    "calibration_age_days": 4
  },
  "smoke_test": {
    "result": "success",
    "bias": {"Fx": 0.31, "Fy": -0.18, "Fz": 0.42, "Tx": 0.011, "Ty": -0.024, "Tz": 0.003},
    "drift_per_s": {"Fx_per_s": 0.012, "Fy_per_s": -0.008, "Fz_per_s": 0.041},
    "thresholds": {"bias_fmax_N": 2.0, "bias_tmax_Nm": 0.3, "drift_max_N_per_s": 0.5}
  }
}
```

Post-mortem analysis can then correlate sensor health and calibration
freshness with episode outcome.

A failed smoke aborts PRE — the wrapper does not enter HOVER. A stale
foundational calibration (> N days, configurable) prints a warning but
does not block.

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

If any of these are missing, both the calibration script and the smoke
test will fail with infrastructure errors. Fix the bringup before
re-running.
