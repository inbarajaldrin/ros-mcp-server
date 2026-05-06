# Handoff for the Next Agent — Compliant Insertion Studio

> Read this **first.** Then route to the right doc based on what you're being asked to do.
>
> Date of last update: 2026-05-06.
> State of the system: **fully autonomous u_orange + u_brown insertion working** (6+ runs incl. fresh-regrasp validation). u_brown also tested with operator-out-of-loop pick→insert. inverted_u_yellow autonomous **not yet working** — multi-prong physics requires further work (open).

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
| `analysis/SEARCH_CONTROL_LAW.md` | SEARCH director (Archimedean spiral + constant-force + lag-pause + gradient-following) |
| `analysis/AUTONOMOUS_RUN_LOG.md` | Per-iteration test journal — append-only |
| `ablations/eval_resources/fmb1_assembly.json` | Source of truth for `gripper_width_mm` and `grasp_id` per object |
| `primitives/shared/config.py:DEFAULT_BASE_POSITION` | Calibrated for FMB1 base1 fixture from u_brown centered-grasp demo |

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

1. **Multi-prong parts (inverted_u_yellow).** `|fz|` saturates 7N during autonomous SEARCH (vs 3N during operator GUIDED). v4 collapse signal absent. Operator-mode Z-locked vs autonomous Z-compliant gives different signatures. Lower F_press helps but doesn't solve. **Suggested next moves**: relative `|fz|` drop detection (vs recent median), peg-z-descent dominance, or hybrid Z control (lock during search).

### Nice-to-have

2. **Two-stage insertion** (operator-flagged): when one part has to clear an alignment phase before the final slot. Spiral can't apply pressure on the underlying object (it'd move). Likely needs a separate FSM state with different constraints.

3. **Sign convention root cause.** `verify_baselink_motion.py` validated single-axis convention; multi-axis SEARCH inverted in practice. The `-Fmax * unit(error)` sign-flip works empirically but the physics gap is unresolved.

4. **Per-object F_press tuning.** Currently 5N for all. Multi-prong wants less; single peg might want more for stronger rim contact signal.

5. **`line_green` autonomous validation.** Has not been tested. `gripper_width=39.8` per assembly JSON. Single-peg geometry — should work like u_brown/u_orange. Validate.

---

## Reproducibility checklist

To replicate the working autonomous insertion from raw data:

1. Confirm part listed in `fmb1_assembly.json` with correct grasp_id + gripper_width_mm.
2. Calibrate `DEFAULT_BASE_POSITION` from a SINGLE centered-grasp GUIDED demo per fixture (one-time per fixture).
3. Wire the part into autonomous: `loop_autonomous_insert.sh 1 --object <name> --no-randomize --regrasp`.
4. If success → run with N=3 for repeatability validation.
5. If failure → forensic analysis on the run's CSV (see `ITERATION_TRACE_2026-05-06.md` §A.6 for example pattern). Then update parameters or detect signal source per `SKILL.md` rules.

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
