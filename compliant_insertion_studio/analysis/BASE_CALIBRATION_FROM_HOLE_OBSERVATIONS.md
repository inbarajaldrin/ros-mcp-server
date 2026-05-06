# Base-position calibration from operator-marked hole observations

**Status:** Planned post-collection. Not implemented yet.

**Concept:** Each GUIDED-mode demo writes `hole_observed_operator.xy_m` to meta (the TCP xy at the moment the operator pressed Enter to mark "above hole"). This is a direct measurement of the slot's projected world location (modulo grasp-offset rotation). Inverting the CAD chain yields one estimate of `T_world_base.xyz` per demo. Aggregating across demos tightens the estimate.

## Math

The CAD chain `predict_tcp_at_seat` computes:

```
TCP_xy_at_seat = base_world.xy + R_base_world × (T_base_object_seat.xy + R_object_world × grasp_offset_in_object).xy
```

For each GUIDED demo we have:
- Measured: `TCP_xy_at_seat = hole_observed_operator.xy_m` (operator's hand at SIGUSR1)
- Known: `T_base_object_seat`, `grasp_offset_in_object`, `R_object_world` (from held_quat after rotate_object)
- Assumed identity: `R_base_world = (0, 0, 0, 1)` (per project convention)

→ Solve:
```
base_world.xy = hole_observed.xy - (T_base_object_seat.xy + R_object_world × grasp_offset_in_object).xy
```

Each demo gives one `(base_world.x, base_world.y)` estimate. With N demos:
- **Median** across estimates → robust to operator noise.
- **Trimmed mean** (drop top/bottom 10%) → robust to occasional operator mismarks.
- **stddev** of estimates → tells us the operator-marking precision (and the calibration confidence).

## Per-base, not per-object

Every (object, grasp_id) pair targeting `base1` gives an independent estimate of `base1`'s world position. So:
- 15 demos for u_orange/grasp_id=1 → 15 estimates
- 10 demos for u_brown/grasp_id=1 → 10 more estimates of the SAME base
- Aggregate all 25 → tighter calibration

The calibration is a property of the *base's physical placement on the workspace*, not the object/grasp. The operator may move the base between sessions, in which case re-calibrate.

## Output

`compliant_insertion_studio/configs/base_calibration_<base_name>_<date>.yaml`:

```yaml
schema_version: 1
calibration_type: base_position_from_hole_observations
base_name: base1
timestamp_iso: '2026-05-XX...'
n_demos_used: 25
demos_basenames: [insert_u_orange_..., insert_u_brown_...]
result:
  xyz_m: [calibrated_x, calibrated_y, -0.0625]   # Z stays at known table height
  xyz_stddev_mm: [σx, σy, 0.0]
  outlier_indices: [...]  # demos rejected as outliers
  prior_xyz_m: [0.0, -0.4, -0.0625]              # what was assumed before
  delta_from_prior_mm: [Δx, Δy, 0.0]             # how much it shifted
notes:
  - 'Use this in primitives.shared.config.DEFAULT_BASE_POSITION'
  - 'Re-run if base is physically moved'
```

## Pipeline (when ready)

1. After at least 10 GUIDED demos exist for any (object, base, grasp_id):
2. `analysis/scripts/40_calibrate_base_from_observations.py --base base1`
   - Scans `compliant_insertion_studio/logs/insert_*.meta.json`
   - Filters demos where `hole_observed_operator` is set (= GUIDED demos)
   - Filters by `base_name == base1`
   - For each demo: read held_quat from `cad_prediction.fold_symmetry_used` or recompute from `current_object_orientation`, compute one estimate of base_world.xy
   - Aggregate (trimmed mean + stddev)
   - Write calibration YAML
3. Operator commits + uses the calibrated value in `DEFAULT_BASE_POSITION`.
4. Future autonomous runs: `predicted_tcp_xy` is mm-accurate → FIND_HOLE algorithm only handles perception-noise-scale offsets.

## Self-improving

Each GUIDED demo is also a calibration datapoint. After every collection session, re-run `40_calibrate_base_from_observations.py` to get an updated calibration with smaller stddev. Calibration confidence grows asymptotically as demos accumulate.

## Why this matters for the autonomous algorithm

Current state: `predicted_tcp_xy` from CAD has ~10-15mm bias relative to the actual slot location (CAD prior is wrong about where the base is, mostly). FIND_HOLE has to search a 10-15mm envelope to find the slot.

After calibration: bias drops to perception-noise-scale (~1-2mm — limited by camera-to-base aruco accuracy + grasp-jaw-closure precision). FIND_HOLE only needs to search a 1-2mm envelope. Operator's "minimal nudge" insight: the autonomous algorithm becomes a small fine-correction loop instead of a wide search, much more reliable.

Equivalent to going from a navigation system that's right within "a city block" to one right within "a parking spot."
