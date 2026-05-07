# Handoff for the Next Agent — Compliant Insertion Studio

> Read this **first.** Then route to the right doc based on what you're being asked to do.
>
> Date of last update: **2026-05-07** (tag `real-world-verified-2026-05-07`, commit 75a418d).
> State of the system: **all four FMB1 objects insert end-to-end via the replay script** —
> u_brown, u_orange, line_green, inverted_u_yellow seat correctly with zero
> manual recovery. Two insertion code paths in production:
>
> - `compliant_insert` wrapper (autonomous SEARCH spiral + v4 detector) — u_brown, u_orange
> - `prismatic_peg_insertion` (stash) routed via `translate_object` line_green/yellow branch — line_green, inverted_u_yellow
>
> Routing key: `primitives/translate_object.py` `if args.object_name in ('line_green', 'inverted_u_yellow')`. **Unification of the two paths is queued for later work** — the split was retained because each path's compliance configuration suits a different geometry (XY-locked descent vs XY-compliant settling with Rx/Ry compliance).

---

## How to localize yourself in the repo (5 minutes)

### 1. What the project IS

**Compliant Insertion Studio** = a force-compliant peg-in-hole **insert primitive** + the operator-demo collection methodology that produced it. Operates on a UR5e + OnRobot RG2 + workspace camera. Replaces the broken `prismatic_peg_insertion.py` real-mode insert path with a per-object parameterized algorithm. FMB1 assembly proof-of-concept (u_brown, u_orange, line_green, inverted_u_yellow).

The high-level deliverable is one CLI: `run_assembly_step.py --autonomous-search` that takes any FMB1 part on the table, picks it via camera, and inserts into base1 — no operator hands.

### 2. The four entry-point docs (read in this order)

1. **`compliant_insertion_studio/CLAUDE.md`** (project root) — constraints, tech stack, conventions. Skim first; the Anti-patterns and Conventions sections are binding.
2. **`compliant_insertion_studio/docs/AUTONOMOUS_INSERTION_METHODOLOGY.md`** — current working architecture. Calibration, SEARCH director, parameters. Treat as the reference for how things work now.
3. **`compliant_insertion_studio/docs/COLLECTION_METHODOLOGY.md`** — the upstream operator-demo pipeline that produced the data v4 was derived from. Read if you're collecting more data.
4. **`compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md`** — methodology rules (data > guesswork; structural > parametric; tool-frame for sensor features). §13–§19 are the running record of session-by-session lessons. Binding for any FSM code change.

### 3. The session reasoning trace

5. **`compliant_insertion_studio/docs/ITERATION_TRACE_2026-05-06.md`** — full chronological reasoning for the 2026-05-06 session. Every iteration, every dead-end, every hypothesis I had to update. **Read this before re-exploring any of the rejected paths** (e.g., `-r_cop` direction control, position-PD tracking, hardcoded bias correction).

### 4. What's true and binding right now

| File | What it is |
|---|---|
| `analysis/CONTROL_LAW.md` | v4 Found Hole detector spec — `\|fz\|` state-transition predicate, validated 10/10 |
| `analysis/SEARCH_CONTROL_LAW.md` | SEARCH director (Archimedean spiral + constant-force + lag-pause + gradient-following). **u_brown / u_orange path only** — line_green / yellow use prismatic. |
| `analysis/AUTONOMOUS_RUN_LOG.md` | Per-iteration test journal — append-only |
| `ablations/eval_resources/fmb1_assembly.json` | Source of truth for `gripper_width_mm` and `grasp_id` per object |
| `ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json` | Canonical FMB1 tool-sequence; `replay_real_assembly.py` consumes this |
| `ablations/replay_real_assembly.py` | Real-mode counterpart to `replay_verify.py`. Drives the full assembly via `--assembly-json` + `--only`. |
| `primitives/shared/config.py:DEFAULT_BASE_POSITION` | (+58.25, -472.10) mm — calibrated for FMB1 base1 fixture from u_brown insert (2026-05-07) |
| `primitives/shared/config.py:PER_OBJECT_BASE_OFFSET_M` | Per-object delta (mm) on top of DEFAULT_BASE_POSITION; populated for ALL 4 objects from observed seat TCP. **Both insertion paths apply this offset** via `--final-base-pos` (wired in `translate_object.py`). |
| `primitives/_real_mode_stash/prismatic_peg_insertion.py` | line_green / yellow insert primitive. Force-added 2026-05-07; gitignored elsewhere. Geometric exit: depth_err ∈ [-8, +1] mm AND `\|dz/dt\| ≤ 0.5 mm/s` over 1.5 s, sustained 1 s. |
| `compliant_insertion_studio/scripts/loop_line_green_prismatic.sh` | Smoke-test loop for line_green (regrasp + rotate + translate_object insert) |

---

## Run commands (copy-paste)

### The autonomous insert (what worked in iter 8)

```bash
# Start from object on table (full pick path; camera must be up):
python3 -u -m compliant_insertion_studio.scripts.run_assembly_step \
  --object-name u_brown --base-name base1 --grasp-id 1 \
  --mode real \
  --base-offset-xy 0.0 0.0 \
  --fz -9.0 --override-fz-cap \
  --step-back auto --step-back-seconds 1.0 \
  --autonomous-search \
  --search-F-press-N 5.0 --search-Fmax-N 5.0 --search-v-s-mm-s 5.0
# (--grasp-width auto-resolves from fmb1_assembly.json)
```

### Multi-iteration validation with fresh regrasps

```bash
bash compliant_insertion_studio/scripts/loop_autonomous_insert.sh 3 \
  --object u_brown --no-randomize --regrasp
```

Auto-resolves `grasp_id` per object. Camera must be up:

```bash
bash compliant_insertion_studio/scripts/launch_camera.sh --background
```

### Collecting a new GUIDED demo (single)

```bash
python3 -u -m compliant_insertion_studio.scripts.collect_regime_data \
  --object <name> --base base1 --grasp-id <N> --grasp-width <W> \
  --fz 9.0 --step-back-seconds 5.0 \
  --held-quat <QX QY QZ QW> \
  --variations A_pos_x_10mm --reps 1
```

### Re-grasp a held part cleanly (camera-driven)

```bash
python3 -u -m compliant_insertion_studio.scripts.regrasp_held_object \
  --object-name <name> --grasp-id <N> --mode real --skip-camera-check
```

---

## Architecture summary (condensed)

```
APPROACH (Fz=-9N descent)
  ↓ contact (|fz_smoothed|>3N sustained 0.1s, after 1s grace)
SEARCH (autonomous spiral PD director)
  • center: predicted_tcp_xy from CAD chain (no FSM-side bias)
  • r0=1.5mm, pitch=2mm, v_s=5mm/s, R_max=8mm
  • constant-force tracking with sign-flip: F_xy = -Fmax * unit(p_ref - p_tcp) - Kd*v
  • lag-pause: theta only advances when peg-to-ref ≤ 2mm
  • gradient override: when |fz| dropping fast AND <6N, command F_xy = -Fmax*unit(v_peg)
  • stall detector: aborts if peg progress / spiral arc < 15% over 1s
  ↓
v4 detector fires (|fz| transition >4N → <3N sustained 0.3s, recent on-rim in 2.5s)
  OR
global seat detector fires (peg dropped straight through during APPROACH/SEARCH)
  ↓
INSERT_DESCENT (Fz=-9N, xy locked, until predicted_tcp_z reached + motion stopped)
  ↓
DONE
```

Code: `compliant_insertion_studio/wrapper/contact_search_fsm.py` (FSM, SearchDirector, FoundHoleDetector). FSM is reusable; tweak parameters via cfg or CLI.

---

## What's binding (must respect)

From `CLAUDE.md` (project conventions) and `SKILL.md` (methodology):

- **Pendant in Local mode.** No `dashboard_client/recover` calls.
- **Force-mode wrench ≤ 5N default**, override only with explicit operator awareness.
- **SIGTERM cleanup must reach safe-state exit.** All states have idempotent cleanup; force_mode → scaled_joint_trajectory_controller switch must complete.
- **No commits unless explicitly approved.** No `Co-Authored-By` lines (operator's global rule).
- **Don't trust FSM stdout outcome labels.** Verify physically from raw CSV. `outcome=success/fsm_seated` ≠ peg in hole — check Δz_to_predicted < 5mm + motion stopped.
- **F/T calibration is three layers**: foundational `set_payload` (per-mount, one-time), session smoke test, per-pose `zero_ftsensor`. The pre-bringup `launch_robot.sh` calls `set_payload` from latest `configs/ft_calibration_*.yaml`.
- **`primitives/shared/config.py:DEFAULT_BASE_POSITION` is calibrated** to (0.00152, -0.40378, ...). Don't reset it unless you re-derive from a centered-grasp demo.
- **Empirical sign-flip** in SearchDirector: `F_xy = -Fmax * unit(error)`. Multi-axis force-mode in our base_link↔base setup inverts. Don't "clean up" without empirical re-validation.

---

## What's still open (priorities for next session)

### Critical-path

1. **Unify the two insertion code paths.** `compliant_insert` (FSM autonomous SEARCH, used for u_brown/u_orange) and `prismatic_peg_insertion` (used for line_green/yellow) live separately and have evolved different compliance configurations. The line_green/yellow case needs Rx/Ry compliance + geometric depth-based exit; the u_brown/u_orange case needs XY-locked descent. Either fold both into one wrapper with per-object compliance config, or formalize the two-track split with a clean interface. Tracking note: `prismatic_peg_insertion.py` lives under `_real_mode_stash/` (force-added to git) — eventual home should be a tracked location like `primitives/inserts/`.

2. **Single-shot reliability is not 100 %.** Today's full assembly required:
   - Manual relaunch of `grasp_points_publisher` after it died silently.
   - Retry on a `force_mode_controller_did_not_activate` (>15 s timeout) — `replay_real_assembly.py` has no built-in retry.
   - Re-run `--only u_orange` after a one-shot camera-detection dropout in `move_to_grasp` (single-poll, no retry).
   Suggested fixes: camera-detection retry loop in `move_to_grasp`; auto-restart `grasp_points_publisher` on death (or fold into replay startup); replay-script-level retry-on-fail.

### Nice-to-have

3. **Two-stage insertion** (operator-flagged): when one part has to clear an alignment phase before the final slot. Spiral can't apply pressure on the underlying object (it'd move). FSM has experimental support but never engaged in a real run.

4. **Sign convention root cause.** `verify_baselink_motion.py` validated single-axis convention; multi-axis SEARCH inverted in practice. The `-Fmax * unit(error)` sign-flip works empirically but the physics gap is unresolved.

5. **Per-object F_press tuning.** Currently 5N for u_brown/u_orange and 1.5N for line_green/yellow (`PER_OBJECT_INSERT_FORCES_N`). Could be auto-tuned from observed rim-contact wrench amplitude.

---

## Reproducibility checklist

To replicate the working full-assembly run (all 4 objects):

1. **Bringup** (in order): launch_robot.sh → launch_camera.sh → grasp_points_publisher.py (see `CLAUDE.md` Bringup section). STOP+PLAY on pendant.
2. **Verify pre-flight**: `ros2 topic echo --once /objects_poses_real | grep child_frame_id` shows all 4 parts; `ros2 topic hz /grasp_points_real` ~5 Hz; `ros2 control list_controllers` shows scaled_joint_trajectory_controller active.
3. **Sequential replay, fail-stop**:
   ```bash
   python3 -u ablations/replay_real_assembly.py --assembly-json <JSON> --only u_brown
   python3 -u ablations/replay_real_assembly.py --assembly-json <JSON> --only u_orange --skip-startup
   python3 -u ablations/replay_real_assembly.py --assembly-json <JSON> --only line_green --skip-startup
   python3 -u ablations/replay_real_assembly.py --assembly-json <JSON> --only inverted_u_yellow --skip-startup
   ```
   On any non-zero exit: stop, inspect `ablations/logs/replay_*_<ts>.log` for the failure reason, address root cause, then continue. Do NOT advance to the next object on failure.
4. **Calibration** (one-time per fixture): `DEFAULT_BASE_POSITION` from a single u_brown insert; per-object delta in `PER_OBJECT_BASE_OFFSET_M` from observed seat TCP vs predicted (see `primitives/shared/config.py` for the calibration arithmetic of each object).
5. **Forensic analysis on failures**: `compliant_insertion_studio/logs/insert_<obj>_<ts>.csv` + `.meta.json`. Schema-v1 columns documented in `wrapper/schema_v1.py`. Both insertion paths now write the same schema, so the analyzer dashboard ingests both.

---

## Anti-patterns the current session worked through (DO NOT repeat)

- ❌ Hardcoding "bias" in FSM from un-centered demos. (Conflates b_obj + g.)
- ❌ Using `-F_lat` or `-r_cop` as autonomous direction signal. (Friction positive-feedback loop.)
- ❌ Position-error PD with default Kp=350. (Below stiction at typical errors.)
- ❌ Spiral starting at θ=0 with center at peg landing position. (Initial direction = arbitrary +X.)
- ❌ Excluding SEARCH/APPROACH from global seat detector. (Misses peg-already-seated.)
- ❌ Tilt-relax detector. (Tilt < 0.01° throughout under full 6-DOF compliance.)
- ❌ Freezing spiral at chamfer-edge dip. (Doesn't help; needs ACTIVE control to push peg into chamfer.)
- ❌ Per-slot bias hardcoded from non-centered demos. (Same conflation as global bias.)
- ❌ Tuning parameters without first analyzing data. (Project rule per CLAUDE.md.)

Each of these was tried and falsified by data this session. See `ITERATION_TRACE_2026-05-06.md` for the data.

---

## File layout (you'll need these)

```
compliant_insertion_studio/
├── CLAUDE.md                                     # project conventions (binding)
├── .claude/skills/insertion-control-law-derivation/SKILL.md  # methodology (binding)
├── docs/
│   ├── HANDOFF_NEXT_AGENT.md                     # this file
│   ├── COLLECTION_METHODOLOGY.md                 # operator-demo pipeline
│   ├── AUTONOMOUS_INSERTION_METHODOLOGY.md       # working autonomous architecture
│   ├── ITERATION_TRACE_2026-05-06.md             # full reasoning trace
│   ├── SCHEMA.md                                 # CSV+meta schema
│   └── SETUP.md                                  # cold-start runbook
├── analysis/
│   ├── CONTROL_LAW.md                            # v4 Found Hole detector
│   ├── SEARCH_CONTROL_LAW.md                     # SEARCH director
│   └── AUTONOMOUS_RUN_LOG.md                     # per-iteration journal
├── wrapper/
│   ├── compliant_insert.py                       # main FSM wrapper
│   ├── contact_search_fsm.py                     # FSM + SearchDirector + FoundHoleDetector
│   └── cad_lookup.py                             # predict_tcp_at_seat
├── scripts/
│   ├── run_assembly_step.py                      # full pick→insert pipeline
│   ├── regrasp_held_object.py                    # camera-driven regrasp
│   ├── loop_autonomous_insert.sh                 # multi-iteration test harness
│   ├── collect_regime_data.py                    # GUIDED-mode demo collection
│   ├── launch_robot.sh                           # robot bringup with set_payload
│   └── launch_camera.sh                          # aruco_camera_localizer
└── configs/
    └── defaults.yaml                             # FSM cfg (search params, v4 thresholds)

primitives/shared/config.py                       # DEFAULT_BASE_POSITION + grasp lookups
ablations/eval_resources/fmb1_assembly.json       # per-(object,grasp_id) widths
```

If you ask "where's X?", the answer is above.
