# Search Director — Active F/T-Driven Control Law (`-r_cop`)

> ## ⚠️ HISTORICAL — this control law was tried and REFUTED. Do not implement it.
>
> **Status (corrected 2026-08-16):** derived from 10 GUIDED demos, run live on 2026-05-06, and
> **refuted**. The `-r_cop` direction signal is a **friction positive-feedback loop**: the sensed
> lateral reaction reinforces whatever direction was just commanded, so the controller confirms
> its own heading and walks the peg away from the hole. The `+0.66` alignment with the operator's
> drag below is real but confounded — it holds only because *the operator* chose the direction.
> See `docs/ITERATION_TRACE_2026-05-06.md` and the anti-pattern lists in
> `docs/HANDOFF_NEXT_AGENT.md` and `docs/AUTONOMOUS_INSERTION_METHODOLOGY.md` §8.
>
> **The director actually in production** is the Archimedean spiral with constant-force tracking,
> lag-pause and gradient-following override — specified in
> **`docs/AUTONOMOUS_INSERTION_METHODOLOGY.md` §2**, implemented in
> `wrapper/contact_search_fsm.py:SearchDirector`. Read that, not this.
>
> **The one statement here that is still current** is the SEARCH selection vector below:
> `(True, True, True, False, False, False)` — X/Y/Z compliant, **rotation LOCKED**. That is not
> optional; unlocking rotation broke five consecutive real-arm runs on 2026-08-16.
>
> Everything else — the parameter table, the Expectation Declaration, the predicted failure modes,
> the iteration plan — is the pre-run record of a hypothesis that did not survive contact with the
> robot. Kept so the reasoning stays readable.

**Companion to:** `CONTROL_LAW.md` (Found Hole detector).

This document specifies the autonomous SEARCH director — the control law that
replaces the operator's hand during the GUIDED phase. With this director +
the v4 Found Hole predicate, the entire insertion pipeline is autonomous:
APPROACH → Contact → SEARCH (active) → v4 fires → INSERT_DESCENT → Seated.

---

## Hypothesis *(refuted — see banner)*

When the peg is pressed down on the rim around the hole:
1. Contact is at the rim edge — off the peg's central axis.
2. Wrist torque from this off-axis contact: `τ = r × F` where `r` is the
   contact point on the peg face (relative to peg axis = tool0 origin), `F`
   is the contact force.
3. Center-of-pressure direction in tool frame: `r_cop_tool = (-Ty/Fz, Tx/Fz)`
   — this points from peg-axis toward the contact point.
4. Geometrically: `-r_cop` points AWAY from the rim, into the open region
   (the hole side, where there is no rim resistance).
5. Therefore, commanding a lateral wrench `F_lat` in the `-r_cop` direction
   drives the peg along the rim toward the hole's open side.

## Empirical evidence

Across 10 clean GUIDED demos in 3 directions (A_pos_x / C_pos_y / D_neg_y),
during the search segment with `|fz_smoothed|>4 N`:

| Metric | Value |
|---|---|
| `align(operator_drag_dir, -r_cop_base)` mean | **+0.66** |
| `align(operator_drag_dir, -r_cop_base)` p5 | +0.51 |
| Worst demo | +0.48 (043426) |
| Best demo | +0.85 (043221) |
| `align(-r_cop_base, contact→rim_cross direction)` mean | **+0.67** |

All 10 demos show positive alignment. The operator's hand is provably
following the `-r_cop` direction (with ~50° average error from perfect
alignment, which is reasonable since the operator also makes diagonal /
overshoot movements not captured by the average direction).

The fact that `-r_cop_base` predicts the OVERALL contact→rim_cross direction
even better (+0.67) shows that the time-averaged search direction matches
the geometric direction to the chamfer.

## Control Law (v1) *(refuted — kept for the record, do not implement)*

At each tick during the SEARCH state:

```python
# Inputs (per tick)
fx, fy, fz       : raw wrench in tool0_controller frame, bias-corrected
tx, ty, tz       : raw torque in tool0_controller frame, bias-corrected
fz_smoothed      : 0.1s moving average of fz (used for gating + r_cop denom)
tcp_quat_xyzw    : current TCP orientation (rotation tool→base)

# Gating: only command when peg is genuinely on rim
if abs(fz_smoothed) < FZ_GATE_LOW_N:           # below 3N: peg may be off-rim, defer to v4
    return ZERO_LATERAL_WRENCH                  # passive (operator-style) — peg holds position

# r_cop in tool frame (m)
fz_for_cop = fz_smoothed if abs(fz_smoothed) >= 0.5 else copysign(0.5, fz_smoothed)
rcop_x_tool = -ty / fz_for_cop
rcop_y_tool =  tx / fz_for_cop
rcop_mag    = hypot(rcop_x_tool, rcop_y_tool)

if rcop_mag < RCOP_MIN_M:                       # very small lever-arm: noise-dominated
    return ZERO_LATERAL_WRENCH

# Rotate -r_cop tool-XY vector to base XY
neg_rcop_tool = (-rcop_x_tool, -rcop_y_tool, 0)
neg_rcop_base = rotation_from_quat(tcp_quat_xyzw).apply(neg_rcop_tool)
ux, uy = neg_rcop_base[0], neg_rcop_base[1]
norm = hypot(ux, uy)
if norm < 1e-6:
    return ZERO_LATERAL_WRENCH
ux, uy = ux/norm, uy/norm

# Command wrench in base_link frame
F_lat_cmd_x = K_SEARCH_N * ux
F_lat_cmd_y = K_SEARCH_N * uy
F_z_cmd     = -F_PRESS_N                         # press peg down, same as APPROACH

return (F_lat_cmd_x, F_lat_cmd_y, F_z_cmd, 0, 0, 0)
```

### Parameters (initial, from data)

| Param | Value | Justification |
|---|---|---|
| `FZ_GATE_LOW_N` | 3.0 N | Same as v4 OFF_RIM threshold. Below this, peg may already be in chamfer; defer to v4 detector. |
| `RCOP_MIN_M` | 0.005 (5 mm) | r_cop must be larger than F/T noise floor + small-lever-arm noise. Operator data showed median `|r_cop|` = 26 mm (well above this). |
| `K_SEARCH_N` | **3.0 N** | Operator's median p95 F_lat = 3.84 N. Start gentler at 3.0 N to be conservative; raise if peg doesn't move. |
| `F_PRESS_N` | **9.0 N** | Same as APPROACH (`fz=-9N`). Maintains rim contact for r_cop measurement. |

### Selection vector for SEARCH state

```
selection_vector = (True, True, True, False, False, False)
                   # X, Y, Z compliant; rotations LOCKED
```

**This line is still current and still binding** — unlike the rest of this document. Same as
APPROACH/INSERT_DESCENT. Different from GUIDED (which had Z locked because operator was pushing
externally — autonomous needs Z compliant to press down).

Rotation must stay locked. With rotation compliant, lateral force applies a moment about the
grasp point: the part pivots in the jaws while the gripper translates, so TCP displacement stops
being peg displacement and any swept-area or coverage figure computed from TCP is invalid. The
"all-True selection vector" rule in `SKILL.md` §10 is wrong for SEARCH; following it broke the
insert for five consecutive real-arm runs on 2026-08-16.

### Force-mode params

```
gain          = 0.5    # same as APPROACH
damping       = 0.7    # same as APPROACH
speed_limits  = (0.02 m/s linear, 0.2 rad/s angular)  # default
```

---

## Expectation Declaration (BEFORE first run)

This is the falsifiable prediction the next autonomous run is expected to
satisfy. Recorded BEFORE the run so we can evaluate whether reality matched.

### Predicted observations

| # | Quantity | Predicted range | Source |
|---|---|---|---|
| E1 | SEARCH phase duration (Contact → v4 fire) | 3–10 s | Operator median 4.0 s; allow 2× margin for sub-optimal autonomous direction |
| E2 | v4 fire xy distance from Contact xy | 5–18 mm | Operator median 13.4 mm; allow some over/undershoot |
| E3 | v4 fire xy distance from CAD-predicted hole xy | 0–8 mm | Hole prior is ~10–15 mm off seat per project history; v4 fires AT rim-cross which is 0–5 mm from actual hole |
| E4 | At v4 fire moment, |fz_smoothed| | < 3 N | v4 predicate's OFF_RIM gate |
| E5 | Outcome | `success / fsm_seated` | If E1–E4 satisfied + chamfer self-aligns, peg should seat |
| E6 | Final z-drop post-Contact | 28–35 mm | All 10 demos: 31 ± 1 mm |
| E7 | Tilt at seat | < 1° | All 10 demos: <0.5° |
| E8 | No SAFETY_RETRACT or F_lat overload | true | F_lat command 3 N is well below 30 N abort threshold |

### Predicted failure modes (to watch for)

| Mode | Symptom | Fix path |
|---|---|---|
| F1: r_cop direction reverses (oscillation) | Peg moves back-and-forth, never crosses rim | Add direction smoothing (low-pass on commanded direction over 0.5s window); reject sudden direction reversals |
| F2: K_SEARCH = 3 N insufficient to overcome friction | Peg stays at Contact xy, no motion | Raise K_SEARCH to 4 or 5 N |
| F3: peg crosses off rim into wrong region | v4 fires but xy not within hole | Check actual peg xy vs CAD prior; may need bias compensation |
| F4: r_cop direction unstable at low fz | Erratic motion when peg briefly lifts off rim | FZ_GATE_LOW_N=3 should prevent this; if not, raise gate to 5 N |
| F5: peg gets stuck on rim corner | v4 doesn't fire after long search | Add a path-bound: if search >10 s, abort and report |
| F6: Z-compliance drift (peg lifts off rim) | `|fz|` drops while peg still on rim region | F_PRESS_N=9 should maintain contact; if not, raise to 12 N |

---

## Iteration plan

1. **Run #1**: Default params (`K_SEARCH=3 N, F_PRESS=9 N`). Predict using E1–E8 above. Run autonomous. Compare to prediction.
2. **If success matches all E1–E8**: control law verified. Document and commit.
3. **If failure**: classify by F1–F6 (or new mode), apply specified fix, re-predict, re-run.
4. **Maximum iterations**: 5. If we haven't succeeded by then, the control law is structurally wrong and we need new analysis (not parameter tuning).

This is the methodology the user asked for: hypothesis → expectation → run → evaluation → iterate or commit.
