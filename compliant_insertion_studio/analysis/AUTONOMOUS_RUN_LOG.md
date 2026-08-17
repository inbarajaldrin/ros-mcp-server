# Autonomous Search Iteration Log

Per-run record of: expectation declaration → run → evaluation → next action.
Methodology: hypothesis-driven, falsifiable, no parameter sweeps without
data-derived structural reasons.

Companion docs: `SEARCH_CONTROL_LAW.md` (control law), `CONTROL_LAW.md` (Found Hole detector).

---

## Iteration 1 — Default parameters

**Date:** 2026-05-06
**Object:** u_orange / base1 / grasp_id=1
**Mode:** `--autonomous-search` (replaces operator drag with F/T director)

### Parameters

| Param | Value | Source |
|---|---|---|
| `K_search_N` | 3.0 N | Operator's median p95 F_lat = 3.84 N. Start gentler. |
| `F_press_N` | 9.0 N | Same as APPROACH default for u_orange |
| `fz_gate_low_N` | 3.0 N | v4 OFF_RIM threshold |
| `rcop_min_m` | 5 mm | Operator's median |r_cop| was 26 mm — well above |
| v4 thresholds | rim_high=4, rim_low=3, off_sustain=0.3s, recent=2.5s | Default validated in stages 3a/3b |
| `search_max_duration_s` | 15.0 | 4× operator median (4.0 s) |

### Expectation declaration (BEFORE run)

| # | Quantity | Predicted | Source / reasoning |
|---|---|---|---|
| E1 | SEARCH duration (Contact → v4 fire) | 3–10 s | Operator median 4.0 s; 2× margin for sub-optimal autonomous direction |
| E2 | Lateral path length during SEARCH | 5–18 mm | Operator median 13.4 mm |
| E3 | v4 fire xy distance from Contact xy | 5–18 mm | Same as E2 |
| E4 | v4 fire xy match to stage 3b's hole_observed | < 5 mm | If autonomous direction agrees with operator's, peg ends up in same place |
| E5 | At v4 fire: \|fz_smoothed\| | < 3 N | v4 predicate definition |
| E6 | Outcome | `success / fsm_seated` | If E1–E5 satisfied + chamfer self-aligns |
| E7 | Z-drop post-Contact | 28–35 mm | All 10 GUIDED demos: 31 ± 1 mm |
| E8 | Tilt at seat | < 1° | All 10 GUIDED demos: < 0.5° |
| E9 | No SAFETY_RETRACT or F_lat overload | true | K_search=3 N << 30 N abort threshold |
| E10 | Direction agreement with operator (offline replay) | 92.8% positive aligned | Verified from `insert_u_orange_20260506_043633` replay |

### Failure mode predictions (to watch for)

| Mode | Symptom | If observed → next iteration's fix |
|---|---|---|
| F1 | Peg oscillates back-and-forth (r_cop reverses) | Add direction smoothing (low-pass 0.5s on commanded direction) |
| F2 | Peg doesn't move (K=3N too low to overcome friction) | Raise K_search to 4 N |
| F3 | Peg moves but timeout before v4 fires | Inspect path; may need K bump or longer timeout |
| F4 | r_cop direction drives peg toward wrong rim region | Check tool↔base frame conversion in director math |
| F5 | F_lat overload abort (>30 N) | Director computed huge force; investigate r_cop sign |
| F6 | Peg lifts off rim (Fz compliance drift) | Raise F_press_N to 12 N |

### Run command

```
python3 -u -m compliant_insertion_studio.scripts.run_assembly_step \
  --object-name u_orange --base-name base1 --grasp-id 1 \
  --mode real \
  --already-held \
  --current-object-orientation 0.0296 -0.7065 0.7065 -0.0296 \
  --base-offset-xy 0.0 0.0 \
  --fz -9.0 --override-fz-cap \
  --step-back auto --step-back-seconds 1.0 \
  --autonomous-search
```

> **CORRECTED 2026-08-16 — do not copy `--step-back-seconds 1.0` out of this record.** That gate
> is the arm-settle window before `zero_ftsensor`, and 1.0 s is inside the danger band: measured
> this session, **0.0 s settle → 15.6 N post-zero bias**, which drove the TCP **116 mm upward**
> once force mode engaged; 5.36 s → 0.11 N. Use **≥ 1.5 s**, and 5.0 s to match the collection
> protocol. The reading is never an obvious error — it is a plausible number that force mode then
> acts on. Left in place because it is what iter 1 actually ran.

### Run outcome (Iter 1 — FAILED, ABORT timeout)

- Outcome: `abort, fsm_abort: SEARCH timeout 15.0s`
- Peg moved 6.6 mm net (path 17.2 mm) in WRONG direction (mostly west; hole was south)
- v4 never fired (|fz|>4N for 97% of search — peg never crossed off rim)
- Net: peg ended at (-30.5, -354.2) when hole was at (-26, -358) → 5 mm west, 4 mm north of hole

### Forensic findings (informed iter 2 design)

- E10 was WRONG. The +0.66 alignment claim was confounded by two issues:
  1. **Operator demos started with deliberate +10mm offsets** (variations A/C/D); peg sat unambiguously on one rim side, where -r_cop direction was meaningful.
  2. **F/T direction signals are unreliable when peg is on flat rim** (Q1 of search): align(-r_cop, →hole) = +0.16 (random); same for -F_lat.
- F/T direction signals **strengthen dramatically** as peg approaches chamfer edge:
  - align(-r_cop, →hole) goes from +0.16 (Q1) → **+0.77 (Q4)**
  - align(-F_lat, →hole) goes from -0.13 → **+0.61**
  - |F_lat| grows 2.4N → 2.8N approaching rim-cross
- This matches the user's hint: F_lat "indicates where the object is blocked" — but only when the chamfer's slope provides directional resistance.

---

## Iteration 2 — Hybrid bootstrap + chamfer-following director

**Date:** 2026-05-06
**Hypothesis:** SearchDirector should use CAD prior to bootstrap heading (peg-on-flat-rim regime where F/T signals are noise), then switch to -F_lat-following as chamfer edge is encountered (|F_lat|>2N).

### Control law v2

```
At each tick during SEARCH:
  if |fz_smoothed| < 3N:                      return zero F_lat (peg might be off-rim)
  Direction A (BOOTSTRAP) = unit(target_xy - tcp_xy)         // CAD prior
  Direction B (CHAMFER)   = unit(-F_lat_sensed)              // away from rim push
  weight_B = clip(|F_lat| / F_chamfer_thresh, 0, 1)
  direction = (1-w)*A + w*B normalized
  Fx_cmd = K_search * direction.x
  Fy_cmd = K_search * direction.y
  Fz_cmd = -F_press
  Selection vector: (T,T,T,F,F,F)              // X,Y,Z compliant, rotation locked
```

### Parameters

| Param | Value | Rationale |
|---|---|---|
| K_search_N | 3.0 | Same as iter 1 |
| F_press_N | 9.0 | Same as iter 1 |
| F_chamfer_thresh_N | 2.0 | Q4 mean |F_lat| was 2.8N; pick threshold below |
| target_xy | CAD prior from `predicted_tcp_xy` | Seeded at SEARCH entry |
| Other v4 params | Default | Verified in stages 3a/3b |

### Expectation declaration (BEFORE iter 2 run)

| # | Quantity | Predicted |
|---|---|---|
| E1 | First several seconds: mode = "bootstrap" or "blend" with low |F_lat| | Peg sits on flat rim near CAD prior |
| E2 | Peg drives toward CAD prior at 3-5 mm/s | Bootstrap is active |
| E3 | At some point during search: |F_lat| crosses 2N | Peg reaches chamfer edge |
| E4 | After |F_lat| > 2N: mode flips to "chamfer", direction reverses if necessary | Director switches signal sources |
| E5 | Peg crosses rim, v4 fires | Final state |
| E6 | SEARCH duration | 5-12 s (slower than iter 1's 15s timeout) |
| E7 | Outcome | `success / fsm_seated` |
| E8 | Peg final xy: within 5mm of (-26, -358) | Hole region |
| E9 | If iter 2 still fails: structural redesign needed (probing, etc.) | TBD |

### Iter 2 outcome (FAILED — same drift pattern, different mechanism)

- ABORT timeout, peg moved 6.4 mm WEST (hole was south)
- CAD prior was empty in meta (`--use-default-base-position` doesn't trigger CAD lookup)
- Even pure -F_lat-following drove peg in self-confirming west direction
- Forensics: -F_lat in autonomous mode is confounded by friction positive feedback (sensed F_lat opposes commanded F_lat in equilibrium → -F_lat reinforces commanded direction)
- The +0.61 Q4 alignment in operator demos was ALSO partly artifact: operator's drag-direction CHOSE the hole, friction made -F_lat match drag, alignment metric ≠ "F_lat is autonomously useful"

---

## Iteration 3 — Archimedean spiral with v4 termination

**Date:** 2026-05-06  
**Source:** GPT-5 analysis (job `gpt-temaxw-74fd8c`, 2026-05-06), backed by Chhatpar/Branicky 2001 + Jasim/Plapper/Voos 2014.

**Hypothesis:** Iter 1 + 2 demonstrated F/T direction signals are unreliable on flat rim (Q1 alignment ≈ 0). The published-evidence-backed answer is **blind-but-bounded spiral** with the v4 detector providing the "hole found" event. v4 is empirically validated 10/10 — it removes the historical pain of spiral search (knowing when to stop). Spiral is parameterized for **chamfer-capture** (~few mm precision), not fine-clearance positioning.

### Control law v3 (spiral PD)

```
On SEARCH entry:
  center_xy = current TCP xy at first-contact
  theta = 0
  spiral_path_len = 0

Per tick:
  if v4 fires: SEARCH → INSERT_DESCENT
  if |fz_smoothed| < 3N: pause spiral advance, hold Fz=-9N
  else:
    advance theta at theta_dot = v_s / max(r, r0)
    r = r0 + (pitch / 2π) * theta
    if r > R_max: ABORT(spiral_exhausted)
    p_ref = center + r * (cos θ, sin θ)
    e = p_ref - tcp_xy
    F_xy = sat(Kp * e − Kd * v_tcp, |F| ≤ Fmax)
  Fz_cmd = -F_press

Stall detector (every 0.75s window): if actual_tcp_progress / spiral_arc_progress < 0.30, ABORT
Selection vector: (T,T,T,F,F,F)
```

### Parameters (from GPT analysis, iter 3)

| Param | Value | Rationale |
|---|---|---|
| r0_m | 0.5 mm | Avoid divide-by-zero at θ=0 |
| pitch_m | 2.0 mm | Chamfer-capture scale (not clearance scale). 4 turns to reach R_max. |
| v_s_m_s | 5 mm/s | Operator median drag speed |
| R_max_m | 8 mm | Covers iter 2's 6.3 mm hole offset + safety margin |
| Kp_xy | 350 N/m | Holds spiral position with 3 N saturation @ 8.6 mm error |
| Kd_xy | 40 N·s/m | Damping for stability |
| Fmax | 3 N | Operator p95 was 3.84 N; conservative |
| F_press | 9 N | Same as APPROACH |
| stall_progress_ratio | 0.30 | Abort if peg can't keep up |
| stall_window_s | 0.75 | Detection window |

### Expectation declaration (BEFORE iter 3 run)

| # | Quantity | Predicted |
|---|---|---|
| E1 | Spiral starts at contact_xy ≈ (-24.4, -351.9) mm | hover landing |
| E2 | Spiral expands at ~v_s = 5 mm/s tangential, reaching r=2mm in ~5s | calculated |
| E3 | At some r ≤ 8mm, peg crosses chamfer edge | hole offset is ~7mm from contact based on iter 2 |
| E4 | v4 fires (|fz| transitions <3N sustained) | mode = "rim-cross" |
| E5 | SEARCH duration | 4–10 s (operator was 4 s; spiral may be slower) |
| E6 | Outcome | `success / fsm_seated` |
| E7 | Z-drop post-Contact | 28–35 mm |
| E8 | Final tilt | < 1° |
| E9 | If ABORT: stall, spiral_exhausted, or moment overload | F1–F6 below |

### Failure mode predictions (from GPT)

| If | Implication | Iter 4 fix |
|---|---|---|
| Clean spiral but R_max reached without v4 | Hole farther than 8mm; too tight stall window | Raise R_max to 12 mm |
| Stall: tcp_xy doesn't follow spiral_ref (Fxy saturated, no motion) | Friction high or fz too aggressive | Reduce Fz to 7N or raise Fmax to 3.5N |
| Fz spikes / moment spikes at same θ | Geometric interference, not search-heading | Re-level peg before search |
| Brief Fz dips, no sustained v4 | Grazing chamfer, not dwelling | Slow v_s to 3.5 mm/s |
| Drift in one Cartesian axis only | Frame mapping bug | Audit base_link → force_mode wiring |

### Run command

Same as iter 1/2 (FSM-internal change).



### Iter 3 outcome: AttributeError ('K' attr)
Code bug — old transition_msg referenced removed attribute. Fixed.

### Iter 4-6 outcomes: Stall variants
Iter 4 (F_press 7N + Fmax 4N PD): stalled at 0.16mm in 0.75s. PD output too small (1.3N << stiction).
Iter 5 (constant-force at Fmax 4N): peg moved 0.71mm — 4× iter 4 but still under 30% threshold.
Iter 6 (constant-force Fmax 5N + v_s 3.5): peg moved EAST 2mm then pinned. Spiral started at θ=0 = +X heading (away from hole).

### Iter 7 outcome: 20mm in WRONG direction
Spiral CENTER at CAD-prior+bias (south-west of contact). Commanded F = (-3.4, -3.5)N south-west. Peg moved (+11, +17)mm NORTH-EAST. Direction inverted.

### Iter 8: FULL AUTONOMOUS SUCCESS

After flipping sign on F_lat output (peg moves opposite of commanded F_lat in this setup):

```
SEARCH active:    1778075599.53s
v4 fired:         1778075604.64s    +5.11s into SEARCH
                  xy=(-27.1, -356.8) mm  |fz|=0.14N
INSERT_DESCENT triggered by v4 predicate (no operator)
DONE / SEATED:    1778075609.62s    +5.0s after INSERT_DESCENT
                  Final tcp_z=199.4mm vs predicted 200.8mm (Δ=-1.4mm, within 5mm tol)
                  motion_stopped, tilt 0.32°
```

**Outcome: success / fsm_seated. Total time from APPROACH-contact to seated: ~10 s.**

### Working configuration (frozen at iter 8, 2026-05-06)

> **CORRECTED 2026-08-16 — this table is a snapshot of iter 8, not the current configuration.**
> Three rows were superseded within days of being written; read
> `docs/AUTONOMOUS_INSERTION_METHODOLOGY.md` §2.1 and the defaults in
> `wrapper/contact_search_fsm.py:SearchDirector` for what actually runs.
>
> | Row | Frozen here | Current | Why it changed |
> |---|---|---|---|
> | Spiral center | CAD prior **+ (−3.66, −5.81) mm bias** | `predicted_tcp_xy`, **no FSM-side bias** | The hardcoded bias is the project's #1 anti-pattern — it conflates fixture-calibration error with per-grasp offset, so it "worked" for u_orange and broke u_brown. Fixture error belongs in `DEFAULT_BASE_POSITION`; per-grasp offset is what the spiral exists to absorb. |
> | `r0` | 0.5 mm | **1.5 mm** | 0.5 mm still lands close enough to centre to stall for want of a traversal direction. |
> | `F_press` / `Fmax` | 7 N / 5 N | **9 N / 8 N** via `translate_object` | Tuned since; 5/5 is measurably worse on u_brown. |
>
> The rest of the table (pitch, v_s, R_max, gain, damping, v4 thresholds, stall params, the
> negated `F_lat` sign) still matches the code.

| Component | Value (iter 8) | Source |
|---|---|---|
| Spiral center | CAD prior + (-3.66, -5.81) mm bias | Empirical from 10 GUIDED demos |
| r0 | 0.5 mm | GPT |
| pitch | 2 mm | GPT (chamfer-capture, not clearance) |
| v_s | 5 mm/s | GPT (operator median) |
| R_max | 8 mm | GPT |
| Fmax | 5 N | iter 6+: needs > 4N to break stiction |
| F_press | 7 N | iter 4+: 9N had too much friction |
| Force-mode gain | 0.5 | Default |
| Force-mode damping | 0.7 | Default |
| F_lat output sign | NEGATED | iter 8: empirical (cmd opposite of desired motion direction) |
| v4 detector thresholds | rim_high=4N, rim_low=3N, off_sustain=0.3s, recent=2.5s | Stages 3a/3b |
| stall_progress_ratio | 0.15 | iter 6+: relaxed from GPT's 0.30 |
| stall_window_s | 1.0 | iter 6+: relaxed from 0.75 |

### Remaining work
1. **Sign convention root cause** — why does cmd Fxy_baselink invert in spiral mode? verify_baselink_motion claimed single-axis works correctly. Multi-axis (Fxy + Fz) might be different. Worth a controlled test.
2. **Generalization** — same control law on B/E variations of u_orange offsets, then on u_brown / line_green / inverted_u_yellow.
3. **Stress test** — run iter 8 5+ times back-to-back, measure success rate and seat-time variance.
4. **Cleanup** — fold AUTONOMOUS_RUN_LOG.md into CONTROL_LAW.md as the production reference.
