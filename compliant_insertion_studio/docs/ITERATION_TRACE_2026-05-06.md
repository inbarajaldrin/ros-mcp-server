# Iteration Trace: From Raw Operator Demos to Working Autonomous Insertion (2026-05-06)

> Full reasoning trace + every iteration + every dead-end, recovered from a single working session that took the project from "10 raw GUIDED operator demos exist, no autonomous insertion" to "fully autonomous u_orange + u_brown insertion validated across 6+ runs with regrasps."
>
> Purpose: future agents and humans should be able to reconstruct WHY each architectural decision was made, what data drove it, and which dead-ends to avoid re-exploring. Strict chronological order — the order matters because each step's data revealed the flaw in the prior step's hypothesis.

---

## Inputs at start of session

- `compliant_insertion_studio/logs/insert_u_orange_2026050[5-6]*` — 18 GUIDED demos with `hole_observed_operator` set
- `analysis/HANDOFF_2026-05-06_DATA-COLLECTED.md` — prior agent's handoff
- `analysis/CONTROL_LAW.md` did NOT exist yet
- The wrapper had v4 GUIDED state but no autonomous SEARCH

---

## Phase A — Detector derivation (Found Hole event)

### A.1 Inventory

Filter raw 18 demos by validity. Found 10 demos with `csv_final=DONE` AND `source=fsm_guided_sigusr1` (live SIGUSR1 capture). 8 had `csv_final=ABORT` AND `source=...backfill_from_csv` — discarded (mid-record cancellations, meta misleadingly says success). Variations covered: A_pos_x_10mm (3), C_pos_y_10mm (3), D_neg_y_10mm (4). Variations B_neg_x and E_diag had ZERO valid demos.

**First wrong move**: I treated "valid demos = 18" initially (my inventory script filtered only on descent magnitude). User caught it: only 10 are real. The 3 demos with anomalous `-r_cop` alignment were all in the cancelled set.

**Lesson**: filter source field, not just descent magnitude. Confounded "the data" with "filterable junk" early.

### A.2 Methodology framing

User reframed scope:
- Marker (Found Hole event) = local sensor signature, MUST be direction-invariant.
- Director (where to drag) = NOT learned from operator drag. CAD prior + base calibration provides it.
- 10 demos × 3 directions = enough to validate invariance.

**My initial confusion**: I conflated marker derivation with director regression. Wrote a `40_search_director_derivation.py` analyzing `align(drag, -r_cop)` as if `-r_cop` could be the autonomous direction signal. User pushed back; I reset.

### A.3 Phase B feature extractor

Built `analysis/scripts/31_decode_operator_action.py`. Per-demo, slice GUIDED segment `[t_contact, t_sigusr1]` and INSERT_DESCENT segment `[t_sigusr1, t_end]`. Compute per-tick tool-frame features: `F_lat_tool`, `r_cop_mag`, `tilt_deg`, `vz`, `fz_smooth`, etc. Output per-demo JSON.

Key technical wrinkle: `t_sigusr1` had two schemas. Live captures stored wall-time epoch; backfilled records stored CSV-relative seconds. Patched extractor to detect: `if t_s > 1e9 then wall_time else relative`.

### A.4 Phase C signature analysis (Found Hole AUC)

`analysis/scripts/32_found_hole_signature.py`: AUC per feature for "at SIGUSR1" vs rolling-baseline windows.

Results across 10 demos:

| Feature | AUC at SIGUSR1 |
|---|---|
| `tilt_deg` | 0.51 (random) — peg held face-down by 6-DOF compliance, no measurable tilt |
| `F_lat_tool` at-event | 0.57 (weak) |
| `\|r_cop\|` at-event | 0.886 (artifact: at SIGUSR1, fz ≈ 0, so r_cop = T/Fz blows up) |
| `fz_smoothed` at-event | 0.74 (lower-is-event) |
| `vz` at-event | 0.90 (lower-is-event) |
| `vz_rise_from_min_1s` | **0.92** (best single feature) |

**First major surprise**: tilt was unobservable (<0.01° throughout). The original hypothesis "tilt-relax detector" was structurally invalid in this dataset.

> **CORRECTED 2026-08-16.** I wrote "FSM full-compliance" here; that was wrong on both counts. The
> FSM was **not** 6-DOF compliant — `selection_vector` is `(T,T,T,F,F,F)`, rotation LOCKED — so
> the <0.01° is a **consequence of that clamp**, not a property of the contact. It is not evidence
> that tilt carries no signal, and it must not be cited as grounds for unlocking rotation.
> Unlocking rotation on 2026-08-16 did make tilt responsive (0.004° → 0.25° excursion) — and broke
> the insert for five consecutive runs. See `analysis/CONTROL_LAW.md` and `SKILL.md` §10.

**Second surprise**: `r_cop` magnitude is dominated by 1/Fz divergence at low Fz. Useful only when Fz is high.

### A.5 Phase F predicate validation — first attempt FAILED

`34_validate_law.py` — predicate v1: `vz ≤ -0.3 mm/s AND F_lat ≤ 2 N AND |fz| ≤ 1 N` sustained 100ms. Validated against acceptance gate `fires within ±300ms of SIGUSR1`.

Result: **0/10 fired within ±300ms; 5/10 fired 2-3 seconds EARLY.**

### A.6 The reframing

Inspecting one demo's trajectory in detail revealed the real picture:

```
At t=32.28s (3.3s BEFORE SIGUSR1):
  fz: +10.59 → -1.17 N  (12N collapse in 250ms)
  F_lat: 4.02 → 0.87 N
  vz: -0.57 mm/s
  dist_to_hole: 5.0 → 2.9 mm
   ← THIS is the rim-cross / chamfer-encounter event

At SIGUSR1 (t=35.62s, 3.3s LATER):
  fz ≈ ±0.3 N, F_lat ≈ 0.6 N, dist_to_hole = 0.0 mm
   ← Operator's caution / fine-tuning. Peg already over chamfer for 3 seconds.
```

**SIGUSR1 ≠ Found Hole.** Operator manually fine-tunes peg to dead-center for 2-6 seconds AFTER the actual rim-cross event. The autonomous controller wants to detect rim-cross, not the operator's mark.

### A.7 Predicate v3 — transient drop (partial)

Reformulated as "rim-cross transient" detector: `fz dropped ≥5N in 300ms AND F_lat ≤ 2N AND |fz_post| ≤ 2N`.

Result: **2/10 fired.** F_lat lags fz collapse by 200-500ms because operator is still actively dragging when peg crosses off rim. F_lat doesn't drop until later.

### A.8 Predicate v4 — state-transition (success)

Drop F_lat from the predicate entirely:

```
ON_RIM   ⇔ |fz_smoothed| > rim_high_thresh (4 N)
OFF_RIM  ⇔ |fz_smoothed| < rim_low_thresh (3 N)        # hysteresis
FIRE when:
    state has been OFF_RIM continuously for ≥ off_sustain_s (0.3 s), AND
    state was ON_RIM at any point in the previous recent_window_s (2.5 s)
```

Result: **10/10 fire** in `[Contact, SIGUSR1]` window. Peg xy at fire time within 0.3-3.4mm of `hole_observed_operator`.

Robustness sweep (96 threshold combos): **94/96 pass 10/10.** Wide robust plateau, not a tuned point.

**Lesson learned (recorded in skill §15)**: tool frame, magnitude only, no horizontal direction signals — direction-invariant by construction.

### A.9 Live validation 3a + 3b

Two stages:
- **3a**: v4 detector wired into FSM in parallel with operator SIGUSR1, log-only. Operator drags + presses SIGUSR1; v4 logs its own fire timestamp. Result: v4 fired correctly on the live robot at 12s into GUIDED, 121s before operator's slow SIGUSR1. xy match to operator's mark: **1.44 mm**.
- **3b**: v4 autofires (no SIGUSR1 needed). Operator still drags. v4 triggered the GUIDED→INSERT_DESCENT transition. Peg seated. xy match: 1.92 mm.

Detector validated end-to-end. Marker is solid.

---

## Phase B — Director derivation (autonomous SEARCH)

### B.1 Initial framing — wrong path

User asked "can autonomous insertion happen now?" My answer: detector works, but director (how to drive peg toward chamfer) is a separate problem because operator's hand provides direction in stages 3a/3b. Removed the legacy spiral / FIND_HOLE path because user dismissed it. Implemented "autonomous SEARCH" with `-r_cop_base` direction control (user said earlier the operator's drag aligned with `-r_cop` at +0.66 across 10 demos).

**Iter 1**: Drive in `-r_cop` direction at K=3N constant.

Result: ABORT timeout. Peg drifted **west 6mm** while hole was **south 7mm**. Peg moved in self-confirming direction.

### B.2 Hybrid bootstrap + chamfer-following

**Iter 2**: Mix CAD-prior heading (when |F_lat| low → bootstrap toward CAD) and `-F_lat`-following (when |F_lat| high → at chamfer edge → chase that signal).

Bug: CAD prediction empty in meta because `--use-default-base-position` doesn't trigger CAD lookup. Fixed by adding fallback in wrapper.

Result: ABORT timeout. Peg drifted west 6mm again — same direction. The `-F_lat`-following alone was driving it.

### B.3 The friction confound

Forensics: cmd `Fxy_baselink = (-3.4, -3.5) N` (south-west toward CAD prior + bias). Peg moved (+11, +17) mm — exactly **OPPOSITE direction**.

Hypothesis: `-F_lat_sensed` in autonomous mode is confounded by friction positive feedback. In equilibrium under force-mode, sensed F_lat opposes commanded F_lat. So `-F_lat_sensed` reinforces whatever direction the controller started in. Self-confirming.

Verified: in operator demos, `-F_lat` aligned with hole because operator chose hole and friction matched drag direction. NOT because `-F_lat` is autonomously useful.

### B.4 Asking GPT for architectural advice

Sent comprehensive prompt to GPT-5 (job `gpt-temaxw-74fd8c`). Response confirmed:

1. F/T direction signals (`-r_cop`, `-F_lat`) ARE confounded in autonomous mode.
2. Recommended **Archimedean spiral** with v4 detection. Backed by Chhatpar/Branicky 2001, Jasim/Plapper/Voos 2014.
3. Specific parameters: `r0=0.5mm, pitch=2mm, v_s=5mm/s, R_max=8mm, Fmax=3N, Kp=350 N/m, Kd=40 N·s/m`.

### B.5 Iter 3 — spiral PD director (broken)

Implemented `SearchDirector` with PD position-tracking on Archimedean spiral reference. AttributeError on first launch (transition_msg referenced removed `K` attribute). Fixed.

### B.6 Iter 4-5 — stiction stalls

**Iter 4** (default Kp=350): ABORT lateral_stall after 1s. tcp_progress 0.16 mm vs spiral_arc 3.75 mm.

Diagnosis: With Kp=350 N/m and typical position errors of 0.5-4 mm, commanded force = 0.18-1.4 N. **Below stiction.** Peg never accelerates.

**Iter 5** (constant-force tracking, replaces PD): `F = Fmax * unit(e)`. tcp_progress 0.71 mm — 4× iter 4. Still below 30% stall threshold.

### B.7 Iter 6 — sign discovery

**Iter 6** (Fmax=5N + v_s=3.5): peg moved 2mm EAST then stalled. Spiral started at θ=0 (heading +X) — but hole was south. Peg got pinned on wrong rim feature.

Tried spiral CENTER at CAD prior + bias (so peg's first motion is naturally toward predicted hole, not arbitrary +X). Iter 7.

### B.8 Iter 7 — sign INVERTED

**Iter 7** (spiral center at CAD-prior+bias): cmd `Fxy = (-3.4, -3.5)` (south-west toward CAD prior). Peg moved **(+11, +17)** mm (north-east) over 18s. Direction completely inverted.

Forensics: looked at `cmd_wrench_raw.csv`. Wrapper logs `intent_baselink` (PRE-flip). Wrapper applies `base_link → base` 180° flip when sending to controller. Single-axis `verify_baselink_motion.py` confirmed convention works for one axis at a time. Multi-axis Fxy+Fz combined behaves inverted in our setup.

**Pragmatic fix**: empirical sign-flip in SearchDirector output. `F_xy = -Fmax * unit(e)`. Root cause never fully resolved, but verified to work on next run.

### B.9 Iter 8 — full autonomous SUCCESS

```
SEARCH duration: 5.11s
v4 fired:        xy=(-27.1, -356.8) mm  |fz|=0.14N
INSERT_DESCENT:  triggered by v4 (no operator)
DONE / SEATED:   Δz_predicted=-1.4mm, tilt 0.32°
```

xy match to operator-marked hole from prior runs: **1.4-1.92 mm**. First fully autonomous run.

### B.10 The "bias" was load-bearing

Setup that worked in iter 8:
- `DEFAULT_BASE_POSITION = (0.0, -0.4)` (uncalibrated)
- FSM bias correction `(-3.66, -5.81) mm` derived from 10 demos
- Net: spiral center at empirical actual-hole; peg lands at predicted_seat which is 7mm offset from spiral center

The 7mm offset gave peg a constant-force pull direction → traversal worked → peg crossed chamfer mid-traverse → v4 fired. Without the offset (peg at center), constant-force chases tiny error around small-radius spiral → orbital lag → stall.

When user asked to fix calibration cleanly:
1. Updated `DEFAULT_BASE_POSITION` to `(0.00152, -0.40378)` so CAD predicts actual hole.
2. Removed FSM bias.
3. Tested zero-offset → STALL (peg lands AT spiral center, no traversal direction).

**Lesson**: the offset that made iter 8 work was a side effect of bad calibration. Fixing calibration breaks iter 8's mechanism.

### B.11 Fix: r_launch > 0

Restore the offset condition without contaminating calibration: set spiral starting radius `r0` from 0.5mm to 1.5mm. Peg always starts ≥1.5mm offset from initial ref position. Constant-force pull direction always exists.

Verified: u_brown autonomous succeeded in 1.11s — fastest single run yet. Calibrated DEFAULT_BASE_POSITION + r_launch=1.5mm + no FSM bias = clean architecture.

### B.12 Iter validation: regrasp loop

Tested across 3 fresh regrasps (each produces different held quat, different per-grasp xy). 3/3 success. Including a regrasp that rotated the part 90° (different prong fell into different slot — algorithm handled it via CAD-chain).

---

## Phase C — Edge cases and refinements

### C.1 Peg-already-seated case (u_brown run #2)

Peg dropped straight through chamfer during APPROACH. Brief fz>3 entering SEARCH. Then aborted on lateral_stall. tcp_z was 1.7mm below predicted_seat with motion stopped.

**Cause**: state-independent seat detector excluded SEARCH from its state list.

**Fix**: include APPROACH and SEARCH in seat-detector's state set. Peg-already-seated handled cleanly.

### C.2 Near-miss freeze (rejected by operator)

When `|fz|` dipped to chamfer-edge levels but didn't sustain, my first attempt was: freeze spiral when `|fz| < 4N`, let peg settle. User rejected: "no point in freezing — peg almost there but moved away. needs ACTIVE control to keep moving in dipping direction, not just stay still."

### C.3 Gradient-following control law

Implemented per user's insight:

```
d_fz_dt = (|fz|_now - |fz|_200ms_ago) / 0.2
if d_fz_dt < -3 N/s AND |fz| < 6N AND |v_peg| > 0.5mm/s:
    F_xy = -Fmax * unit(v_peg)    # continue current peg motion
else:
    spiral PD as usual
```

Override pushes peg deeper into the chamfer along the path that's making fz drop, rather than letting spiral pull peg back to rim.

### C.4 Per-object grasp config

inverted_u_yellow exposed hardcoded `--grasp-width 35`. Wrong (needs 56.7). Wired up `fmb1_assembly.json` lookup helpers in `primitives/shared/config.py`. Per-object width and grasp_id auto-resolved.

### C.5 Multi-prong limit (open problem)

inverted_u_yellow autonomous: `|fz|` saturates 7N throughout SEARCH. Operator-mode demo on same part: median 3.14N. Difference: operator demos lock Z (selection_vector T,T,F,F,F,F); autonomous uses Z-compliant with Fz=-9N command. Multi-prong contact + Z-compliant = `|fz|` never collapses.

> **CORRECTED 2026-08-16 — the 3.14 N figure does not generalise.** It came from a single demo and
> was subsequently quoted as a corpus-wide operator statistic ("median 3.14 N, 50% of the time
> below 3 N"). Re-measured across the whole GOLD corpus (contact→SIGUSR1, tool frame), operator
> median `|fz|` is **3.30–8.89 N depending on the part**, and the fraction below 3 N ranges
> **7.7%–45.3%**. There is no single operator drag pressure. The *qualitative* point this section
> makes — autonomous Z-compliant SEARCH saturates `|fz|` where operator Z-locked GUIDED does not —
> still stands; the number does not.

Lowered F_press 9 → 5N. Helps but doesn't fundamentally fix. Next agent needs:
- Relative `|fz|` detection (drop vs recent median)
- Or peg-z descent dominance
- Or hybrid Z control (lock during search, compliant on rim-cross)

---

## Reasoning trace summary — what I had to update mid-stream

| Original assumption | What data showed | Updated assumption |
|---|---|---|
| Tilt-relax is the Found Hole signal | Tilt < 0.01° throughout — *corrected 2026-08-16: this is a consequence of the rotation clamp `(T,T,T,F,F,F)`, not of contact physics; see §A.1* | Use `\|fz\|` magnitude state-transition |
| SIGUSR1 = Found Hole event | SIGUSR1 fires 2-6s AFTER actual rim-cross | Detect rim-cross (at chamfer edge), not operator mark |
| `-r_cop` direction predicts hole | True in Q4 of search (+0.77 alignment) but only because operator chose direction | Don't use it autonomously — friction confound |
| Single bias correction generalizes | u_orange: 6/6 success. u_brown: failed because bias wrong for that slot/grasp | Calibrate DEFAULT_BASE_POSITION; absorb per-grasp via spiral r_launch |
| PD position tracking | Stalls at typical errors (1.3N below stiction) | Constant-force tracking |
| Spiral center at TCP landing | Works only if peg has natural offset from center | r_launch ≥ 1.5mm forces an initial direction |
| Single set of v4 thresholds works for all parts | u_orange and u_brown succeed; inverted_u_yellow saturates | Multi-prong parts need different signal (open) |

---

## Reproducibility checklist for future agents

1. Read `analysis/CONTROL_LAW.md` (v4 detector) and `analysis/SEARCH_CONTROL_LAW.md` (SEARCH director) — both are checked-in.
2. Read this trace + `docs/AUTONOMOUS_INSERTION_METHODOLOGY.md` for current architecture.
3. Read `compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md` §13-§19 for the rules every iteration violated and recovered from.
4. Use `analysis/AUTONOMOUS_RUN_LOG.md` as the iteration journal. Append, don't overwrite.
5. The 10 GUIDED demos that v4 was derived from: `compliant_insertion_studio/logs/insert_u_orange_2026050[5-6]_0[345]*.csv` filtered to `outcome=success AND source=fsm_guided_sigusr1` (NOT backfill source).
