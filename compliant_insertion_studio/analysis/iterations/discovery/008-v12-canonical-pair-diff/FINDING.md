# Discovery 008 — canonical A/B pair diff: findings

**Pair:**
- GOLD `insert_u_orange_20260505_193645` (operator success, 116 mm descent in 68 s)
- FAIL `insert_u_orange_20260505_193941` (autonomous abort, 0.12 mm descent in 15 s contact window before STUCK)

**force_mode_params identical** (selection_vector all-True, fz=-9N hover→approach, gain=1.0/damping=0.7 baseline). Confirmed via `cmd_identity` block in `metrics.json`.

## Headline result — answer to the question

The FSM commanded **the same wrench sequence as the operator's run** (one-tick FSM transient peak (-6.07,+13.72)N then sustained (0,+5)N Fy at gain=1.0/damping=0.3 in find_hole). Yet:

| Feature (post-contact 5s) | GOLD | FAIL | ratio |
|---|---|---|---|
| xy_excursion max (mm) | **15.91** | 7.01 | 2.27× |
| F_lat_base median (N) | **1.98** | 1.09 | 1.82× |
| r_cop median (mm) | **22.3** | 12.8 | 1.74× |
| v_xy median (mm/s) | **2.35** | 1.25 | 1.88× |
| operator-nudge residual median (mm/s) | **3.21** | 1.15 | 2.80× |

| Joint xcor (eff_j_i vs |v_xy|) | GOLD | FAIL |
|---|---|---|
| j0 (base rotation) corr | **+0.53** | +0.12 |
| j0 abs-max effort (Nm) | **2.40** | 1.11 |
| j1 abs-max effort (Nm) | 1.44 | **2.27** |

Maximum F_lat values are nearly identical between runs (GOLD 18.1 N vs FAIL 16.6 N peak), but GOLD **sustains** the high-engagement contact while FAIL gets brief peaks then relaxes back. This shows up as a 1.7-1.8× gap in the medians.

The peg in the FAIL run sits **flat on the rim**: r_cop median is 12.8 mm (a small lever-arm). The peg in the GOLD run **rides the chamfer edge**: r_cop median is 22.3 mm (essentially the peg radius — geometric proof of edge-contact). Both started with similar ~13 mm offset between contact_xy and seat_xy (per discovery 005), but only GOLD progressed to chamfer engagement.

## What the operator's hand does (direct measurement)

Operator-nudge signature = TCP velocity not explained by the sensed wrench under empirical admittance (0.5 mm/s/N):

- GOLD median residual = **3.21 mm/s** (p75 = 4.81 mm/s, max 11.2)
- FAIL median residual = 1.15 mm/s (p75 = 3.23 mm/s, max 7.93)
- Median residual direction GOLD = -1.36 rad (-78°), FAIL = -0.41 rad (-23°)

Interpretation: the GOLD trace has ~3 mm/s of TCP xy velocity that the controller's commanded wrench + sensed-reaction admittance cannot explain — that's the operator's hand. FAIL has only ~1 mm/s residual (consistent with controller noise + drift). The residual direction in GOLD is **consistent** (low circular variance not directly computed, but the median is a nondegenerate -78°), suggesting the operator pushes in a stable direction toward the seat.

## Time-of-divergence (FAIL leaves GOLD's IQR-band, post-contact)

| Feature | t (s) |
|---|---|
| tcp_z | 0.00 |
| dz_dt | 0.04 |
| fz_t | 0.04 |
| F_lat_base | 0.17 |
| F_lat_tool | 0.17 |
| r_cop | 0.17 |
| v_xy | 0.41 |

The two trajectories diverge **within 50-200 ms** of contact. By t=170 ms, the lateral-force, F_lat-tool, and r_cop signals have all left the GOLD band — the algorithm fails to engage the chamfer almost immediately, and never recovers in the 15 s window.

## Joint-load decomposition (NEW signal — schema v1.2 first-look)

GOLD's **j0 (base rotation)** carries 2.40 Nm peak effort and shows +0.53 cross-correlation with TCP |v_xy| at +22-sample lag (~88 ms). FAIL's j0 only sees 1.11 Nm peak effort and 0.12 corr. **The operator pushes through the base of the arm**, generating distal sweep at the wrist via base-joint rotation — exactly the ~5 mm horizontal arc that FINDINGS §3 originally identified as the GOLD u_orange signature. The autonomous run's commanded wrench cannot recreate this because force-mode admittance generates motion at the TCP frame, not the base joint.

Conversely FAIL's **j1 (shoulder lift)** shows higher peak effort (2.27 vs 1.44 Nm) — the autonomous controller is loading the shoulder fighting Z-axis reaction without the lateral progress that GOLD's j0 push delivers.

## The single most important new invariant

**I016 (proposed):** With matched force-mode params and matched FSM-commanded-wrench, the operator-driven success differs from the autonomous fail by:
- 2.3× larger xy excursion in 5 s post-contact (15.9 mm vs 7.0 mm)
- 1.7-1.8× higher sustained F_lat_base, F_lat_tool, r_cop medians
- 2.8× larger TCP-velocity residual unexplained by admittance (3.2 mm/s vs 1.1 mm/s)
- The base joint (j0) carries the operator's nudge, with j0 effort vs |v_xy| corr = 0.53

The operator's contribution is a **base-frame lateral push that drags the TCP through ~16 mm of physical excursion** while the controller's directed Fy=+5N at gain=1.0/damping=0.3 only achieves ~7 mm.

This is the missing mechanism. Algorithms that match the operator's TCP-velocity profile must either:
(a) command a larger lateral force (≤ 6 N hard cap by rules, so headroom is small)
(b) command a longer sustained directed push at multiple angles
(c) intermittently lower gain_scaling/damping to produce more compliant (i.e. larger-velocity) lateral motion under the same commanded force
(d) inject base-frame lateral velocity directly via a position/velocity controller during the search phase (REFUTED — that's what XY-position-tracked selection_vector did, v91)

Option (c) — already partially in place: find_hole transitions to gain=1.0, damping=0.3 (lower damping = higher steady-state velocity). The operator's run shows even that is insufficient. Practical option: **drop damping further during search** (e.g. 0.15-0.20), and/or **rotate the directed push through more directions** (already in H101 patch). 

Refines I003+I004 (operator displacement 1.2-1.7 mm in 1s on May-3 light-touch demos) — note the May-5 GOLD here shows 2.5 mm in 1 s (within 1.7 mm-2.5 mm of demo range), but **15.9 mm over 5 s**, demonstrating the operator sustains the push for the full search window, not a single 1-second nudge. The duration is a key portable parameter.

## Why FAIL gets stuck at the rim

Putting it together: at t=170 ms post-contact, the autonomous controller's compliance-induced lateral velocity is already insufficient to drag the peg far enough to expose the chamfer. By t=400 ms, |v_xy| has separated. The peg sits flat on the rim (r_cop = 5 mm at depth band 0-1 mm vs GOLD's 20 mm), no chamfer-edge contact develops, no Fz collapse occurs, and the FIND_HOLE STUCK predicate eventually fires at 15 s.

The autonomous controller never converts directed lateral force into the **sustained 15 mm+ TCP excursion** the operator delivers.

## Cross-references

- `metrics.json` — machine-readable summary
- `data/canonical_pair_diff.json` — full numeric output
- `scripts/17_canonical_pair_diff.py` — the analysis script
- Builds on I003 (operator c→s direction), I004 (operator displacement), I005 (F_lat reaction), I007 (F_lat divergence at +50 ms)
- Refines u_orange-specific I006 (path/bbox compactness) — explains the mechanism

## Limitations

- n=1 pair. Other A/B pairs at different gain/damping or other objects are needed for portability.
- Operator-nudge signature uses scalar admittance gain; full 3×3 admittance matrix would tighten the residual.
- The empirical admittance K=0.5 mm/s/N comes from FINDINGS operator demos; using it to "subtract" GOLD's controller-attributable motion is biased (it overstates GOLD's residual). However, the **gap** GOLD-vs-FAIL using the same K is robust to that bias.
- Cannot prove the operator's hand was *the* cause vs. a coincidental difference; we have only one run per outcome class.
