# Reverse-engineered Insert Algorithm — Findings from Raw Telemetry

> **READ FIRST:** `compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md` is the authoritative methodology. Findings below were captured in chronological order; some early sections (§1–§9) used **z_drop-only metrics and treated `hole_xy_prior` as a hard target** — both refuted in the May-5 session. The orientation-feedback model in the skill supersedes those framings. Sections marked `<!-- STALE_PER_SKILL_v2 -->` should be re-validated against the v1.2 sidecar pair (`insert_u_orange_20260505_193645` GOLD + `insert_u_orange_20260505_193941` FAIL) before being cited as findings.

**Dataset:** 185 episodes across 4 FMB1 parts (60 May-3 operator demos schema_v1.1 + 132 May-4 u_orange schema_v1.1 + 4 May-5 episodes schema_v1.2 with full sidecars).
Tools: per-sample feature extraction, depth-banded pooling, cross-run divergence analysis, 3-way pose comparison (GOLD ↔ FAIL ↔ PROJECTED CAD).
All scripts under `compliant_insertion_studio/analysis/scripts/`.

---

## 1. The single universal invariant

**`Fz_t` (peg-axis reaction force, tool frame) collapse from ~9 N to <2 N is the moment the peg slides off the rim into the slot.**

Verified across all 4 objects. Once the collapse happens, descent is essentially passive at **dz/dt ≈ -6.4 mm/s** (constant, geometry-independent — gravity + admittance with cmd_fz=-9 N).

| Object | Fz_t pre-collapse | Fz_t post-collapse | dz/dt steady |
|---|---|---|---|
| u_orange GOLD | +9.15 N | +0.08 N | -6.47 mm/s |
| u_brown GOLD | +9.21 N | +0.04 N | -6.36 mm/s |
| inv_u_yellow GOLD | +9.15 N | +0.18 N | -6.42 mm/s |
| line_green GOLD | +9.25 N | (rebounds — shallow slot) | varies |
| u_orange FAIL | +8.53 N | **+8.84 N (no collapse)** | -0.05 mm/s |

## 2. The search-phase action that PRODUCES the collapse

For each successful demo, in the **1 second before** Fz collapse:

| Quantity | GOLD median | FAIL median |
|---|---|---|
| TCP displacement (mm) in contact→seat direction | **1.2 – 1.7 mm** | 0.28 mm |
| Direction of TCP motion vs geometric c→s vector | within **5–15°** | random/spiral |
| Sensed F_lat (base frame) | **2.0 – 2.9 N** | 0.94 N |
| Time from contact to collapse | 2.1 – 3.0 s | never (or 4.3 s brief dip) |
| Commanded Fz | -9.0 N | -7.5 N |

The operator's TCP motion direction during search aligns with the geometric **contact→seat vector** within 5–15° in 8/10 u_orange GOLD demos. (Cases >30° deviation correlate with c→s magnitude < 2 mm — direction is meaningless when contact is nearly at seat.)

## 3. The motion-pattern discriminator

10-s post-contact window TCP path geometry:

| Metric | GOLD | FAIL |
|---|---|---|
| arc length (mm) | 13.7 | 11.4 (similar) |
| bounding-box diagonal (mm) | **4.7** | **2.2** |
| path / bbox ratio | 2.8 (wide arc) | **5.4 (tight spiral)** |

Both move TCP about the same arc length. **GOLD sweeps it across 4.7 mm of physical space; FAIL spirals back on itself in 2.2 mm.** The algorithm's spiral is half the radius needed to slide off the rim.

## 4. The single-shot success discriminator (validated)

`durable_collapse` ≡ `|Fz_t (smoothed)| < 2 N AND dz/dt < -2 mm/s sustained 0.5 s`

| Class | Episodes | durable_collapse rate | actually-seated rate (>20 mm) |
|---|---|---|---|
| u_orange GOLD operator | 10 | **100%** | 100% |
| u_brown GOLD operator | 10 | 80% | 80% |
| inv_u_yellow GOLD operator | 20 | 80% | 80% |
| line_green GOLD operator (shallow slot) | 20 | 70% | 0% (slot only 7 mm deep) |
| u_orange autonomous abort | 74 | 20% | 7% |
| u_orange autonomous timeout | 7 | 0% | 0% |

**98% recall, 1 FN/185 episodes.** Safer than the existing FSM's z-drop predicate, which trips on metastable rim-perched states.

<!-- STALE_PER_SKILL_v2: §5 (control-law pseudocode) was based on z_drop-only signals and treated hole_xy_prior as hard. Superseded by the orientation-feedback design in the skill (Section 13). The Fz-collapse predicate at the bottom of this section IS still valid as a state-transition signal, but cannot be the sole control law for tight-clearance pegs. -->

## 5. Reverse-engineered control law

```
state = APPROACH
  cmd = move_to(hover_xy, hover_z)
  on touchdown: state = SEARCH

state = SEARCH
  cmd_fz = -9 N
  cmd_F_lat = 5 N applied in direction (seat_xy - tcp_xy).normalized
  predicates:
    if |Fz_t (smoothed 0.5s)| < 2 N AND dz/dt < -2 mm/s for 0.5 s:
      state = INSERT (LATCH this transition)
    if t_in_search > 6 s with xy_displacement < 1 mm:
      state = ABORT_NEED_REPLAN (operator-clear: target xy is wrong)

state = INSERT
  cmd_fz = -9 N
  cmd_F_lat = 0     # admittance carries the rest; over-driving here jams
  predicates:
    if Fz_t > +6 N (smoothed 0.5s) AND |dz/dt| < 0.5 mm/s for 0.5 s:
      state = SEATED
    if t_in_insert > 8 s without seat (i.e. peg got back on a ledge):
      state = SEARCH   # mid-descent ledge-stuck — push laterally again

state = SEATED
  cmd_fz = -2 N (settle)
  declare DONE after 0.5 s
```

The state machine uses **Fz_t as the primary state signal**, NOT z_drop. z_drop is the symptom; Fz_t is the cause.

## 6. What the current FSM is doing wrong

Reading `contact_search_fsm.py` against these findings:

| Current FSM | Data says |
|---|---|
| Spiral with ~1–2 mm radius | Insufficient — needs ~5 mm sweep in c→s direction |
| `engagement_dist_thresh_m = 6 mm` (xy distance gate) | Wrong predicate — should be Fz_t collapse |
| `motion_stopped_first_t` resets per state | Should latch globally; the state-independent seat detector you added is right but use `Fz_t` not z_drop |
| INITIAL_PRESS at fixed +Y baselink for 1.5 s, 5 N | Direction was wrong — should be `seat_xy - tcp_xy`, computed from CAD or last-known seat. Magnitude ~5 N is correct. Duration 1.5–2 s is correct. |
| WEDGE_RECOVERY using counter-residual direction | Anti-pattern (per CLAUDE.md) — operator demos show direction = c→s, not -F_residual |
| stuck-detection by instantaneous v_z | `Fz_t > 6 N for 2 s` is the durable wedge signal |

## 7. What's NOT a discriminator

These features are **similar** between GOLD and FAIL — chasing them is wasted iteration:

- F_lat magnitude (1.2 N GOLD vs 1.1 N FAIL — sensed reaction is similar)
- TCP arc length over 10 s (13.7 mm GOLD vs 11.4 mm FAIL — both moving)
- Tilt at contact (0.5° GOLD vs 1.4° FAIL — not the cause)
- r_cop (the COP "lever-arm" — large in both cases)
- Commanded Fz (-9 vs -7.5 — small effect)

The only features that strongly diverge are: bbox_diag (motion compactness), Fz_t collapse, dz/dt sustained, and final_z_drop. Of these, the **earliest-discriminating** is Fz_t collapse (within 2–3 s post-contact).

## 8. Open question — direction prior

The control law commands F_lat in `(seat_xy - tcp_xy)` direction. **Where does seat_xy come from?**

Options ranked by reliability:
1. **Last successful operator demo's seat xy** for this base+grasp+object — most reliable but requires demo first.
2. **CAD prediction `predict_tcp_at_seat`** — has 5–17 mm error per current observations. Could still be a reasonable starting prior (better than spiral around contact_xy).
3. **Previous attempt's seat xy** if any prior succeeded (already in `hole_observed` chain).
4. **Adaptive: rotate the search direction by π/3 every 1.5 s if no collapse** — turns the directed push into a sparse pseudo-spiral over the 6 cardinal+diagonal directions while preserving the ~5 mm sweep magnitude. Likely the right fallback when no prior is available.

The May-3 GOLD demos' c→s direction was within 30° of "due -Y in base" for u_orange; for u_brown and inv_u_yellow it was -Y dominant; for line_green it was +X/-X (the line is the slot's long axis). **The c→s direction is a per-object-per-base prior, not a universal constant.**

---

## 9. Trajectory-alignment findings (added 2026-05-05)

After aligning all 185 episodes on the contact moment (t=0) and cropping to [-1s, +6s]:

| Feature | First-divergence time | GOLD@div | FAIL@div | GOLD band [p25, p75] |
|---|---|---|---|---|
| `fz_t` | t = 0.00 s | +6.00 N | +7.20 N | [+4.99, +6.59] |
| `z_drop_mm` | t = +0.02 s | +0.04 mm | +0.02 mm | [+0.03, +0.06] |
| `F_lat_base` | t = +0.05 s | +1.09 N | +0.65 N | [+0.66, +1.57] |
| `dz_dt_mm_s` | t = 0.00 s | -3.14 mm/s | -3.23 mm/s | [-3.22, -3.06] |
| `tilt_deg` | t = +1.60 s | +1.56° | +0.90° | [+0.91, +2.37] |

**Three new invariants (I007–I009 in STATE.json):**

- **I007**: At t=+50ms post-contact, FAIL F_lat_base (0.65 N) is below the GOLD band's lower bound (0.66 N). The lateral force gap is detectable within 50 ms — this is a fast-fail signal for early abort+replan.
- **I008**: At t=+1.6s, FAIL tilt is *lower* than GOLD (0.90° vs 1.56°). Counter-intuitive but consistent: tilt within the 5° envelope is a *positive* sign of progress (peg engaging chamfer geometry), not a failure mode. Algorithms penalizing tilt are punishing the wrong signal.
- **I009**: At contact (t=0), FAIL fz_t (7.20 N) > GOLD fz_t (6.00 N). Failures land 20% harder on the rim. Approach-velocity reduction is a candidate fix (H105).

## 10. Mined v82–v97 history (added 2026-05-05)

`v82_v97_iteration_history.json` captures the prior 12-hour session's 16 iterations as labeled hypothesis→outcome data. Refuted entries (do not re-test):

- v83 — engaged_tilt_max=1.5° too tight (real demo 3.29°)
- v85 / v89 — spiral_F_max sweeps (4N, 6N, 12N) — magnitude is not the issue
- v90 — bulk parameter bump (Fz, gain, damping) — sweep ineffective
- v91 — XY position-tracked via selection_vector — peg can't find chamfer
- v94 — spiral_v 5× slower — speed is not the issue
- v95 — INITIAL_PRESS in CAD-derived direction — direction was wrong (CAD has 5-17mm error)

Validated entries (kept in code):

- v86 — seating predicate accepts 25mm-from-surface
- v87 — state-independent global seat detector
- v92 — selection_vector all-force-controlled (admittance everywhere) — **superseded, see below**
- v93 — aggressive 15s stuck detection (URCap stress mitigation)
- v96 — INITIAL_PRESS as a phase (direction must be data-derived per-attempt)

> **CORRECTED 2026-08-16 (v92).** This entry is no longer "kept in code" and must not be cited as
> a validated decision. Since 2026-05-06 every `selection_vector` tuple in
> `contact_search_fsm.py` is `(True, True, True, False, False, False)` — **X/Y/Z compliant,
> rotation LOCKED** — including both SEARCH command sites. See
> `SESSION_STATE_2026-05-06.md` (rotation-lock row) for when it changed, and `SKILL.md` §10 for
> the mechanism. An agent that read v92 (or SKILL.md's unqualified all-True rule) and re-enabled
> rotational compliance broke the insert for five consecutive real-arm runs: with rotation
> compliant, lateral force applies a moment about the grasp point, the part pivots in the jaws,
> and TCP displacement stops being peg displacement — invalidating every swept-area figure
> computed from TCP. The v92 finding remains true only in its original narrow sense (admittance
> in the *translational* axes beats XY position-tracking, refuted at v91).

**Meta-lesson:** parameter sweeps (v85/v89/v90/v94) consumed ~half the session and produced nothing. Cross-run CSV analysis (done at session END) produced all the actionable findings. The discovery-mode ralph loop encodes this as default behavior.

---

## 11. I006 cross-object portability — REFUTED (added 2026-05-05)

10 s post-contact window, May-3 GOLD demos vs May-4 FAIL u_orange:

| Group              |  n | path_len_mm (p50) | bbox_diag_mm (p50) | path/bbox ratio (p50) |
|--------------------|---:|------------------:|-------------------:|----------------------:|
| GOLD u_orange      | 10 |             13.66 |             4.67   |                2.82   |
| GOLD u_brown       | 10 |             13.12 |             2.11   |                6.26   |
| GOLD inv_u_yellow  | 20 |             12.69 |             2.18   |                5.98   |
| GOLD line_green    | 20 |             12.64 |             2.61   |                4.82   |
| FAIL u_orange      | 69 |             11.37 |             2.17   |                5.38   |

**The "wide arc" GOLD signature is u_orange-specific.** GOLD u_brown / inv_u_yellow / line_green all sweep <2.6 mm bbox at ratios 4.82-6.26 — overlapping or exceeding FAIL u_orange's 5.38. No universal threshold separates GOLD from FAIL on path/bbox.

What still stands from §3: the *path-length* is universal (~12-14 mm in 10 s, GOLD and FAIL alike). What does NOT generalize: the bbox-diagonal magnitude and the path/bbox ratio.

Implication: drop path/bbox from H103's auto-replan trigger. Keep `xy_disp < 1 mm` as the stuck signal, not the path-shape ratio.

Iteration: `analysis/iterations/discovery/001-i006-cross-object-portability/`

---

## Pointers to data

- `analysis/scripts/01_extract.py` — parser + per-sample features
- `analysis/data/summaries.json` — 185 episode summaries
- `analysis/data/per_sample/` — one JSON per episode with full trajectories
- `analysis/data/aligned_summary.json` — contact-aligned per-feature percentiles per class
- `analysis/data/bin_stats.json` — depth-banded medians/IQRs per group
- `analysis/v82_v97_iteration_history.json` — historical hypothesis→outcome ledger
- `/tmp/insert_analyzer/bin_stats.json` — depth-banded medians/IQRs per group
- `/tmp/insert_analyzer/search_phase.json` — per-episode pre-collapse details
- `/tmp/insert_analyzer/discriminator.py` — Fz-collapse classifier validation

---

## §6 — H109 refuted: at-contact xy is NOT the success/fail discriminator (discovery 005)

**Question (from §5/discovery 004 interpretation A):** does AUTO_success contact land
systematically closer to the slot center than AUTO_fail contact?

**Test:** 119 May-04 u_orange autonomous attempts (19 success / 100 fail). For each,
`offset_mm = ||contact_xy - reference_seat_xy||`. Two reference seat choices: (a) median
of the 19 AUTO_success actual seat positions (data-driven), and (b) STATE.json
`predicted_seat_xy_m` = (+0.0341, -0.3635).

| Reference seat | AUTO_success offset_mm (median, IQR) | AUTO_fail offset_mm (median, IQR) | Δ median | AUC | p (one-sided) |
|---|---|---|---|---|---|
| data_seat (+0.0290, -0.3674) | 13.85 [12.30, 16.53] | 14.72 [13.25, 16.19] | +0.87 mm | 0.581 | 0.13 |
| predicted_seat_state_json | 11.96 [9.72, 14.21] | 12.50 [10.87, 14.31] | +0.54 mm | 0.569 | 0.17 |

**Result: REFUTED.** AUC ≈ 0.57 (random = 0.50, useful ≥ 0.75). Both classes contact
~13 mm from seat with overlapping IQRs. The H109 patch ("retract+re-approach if offset > 4 mm")
would have rejected nearly every AUTO_success — its 25th-percentile offset is 12.3 mm.

**Combined with §5 (I012):** neither at-contact xy nor per-timepoint post-contact medians
discriminate. By elimination, the success/fail signal lives in either:
1. **Post-contact feature aggregates** (max F_lat, integral xy_excursion, time-above-tilt-threshold),
   or
2. **Signals not currently recorded** (joint_states, native-rate wrench, per-sample 6-axis cmd —
   STATE.json:data_recording_gaps G001-G004).

H110 (test 1) is now the highest-priority pending hypothesis. If H110's best AUC < 0.65, we
have hit a measurement floor and the schema bump becomes the bottleneck.

**Secondary finding:** the 19 AUTO_success seat positions have IQR_y = 13.0 mm vs IQR_x = 5.7 mm.
Y-spread is 2.3× X-spread. Could mean u_orange's slot is elongated in Y, or that some
"successes" wedged at intermediate depths in the chamfer rather than fully seating. Worth a
follow-up partitioning at actual_seat_z (deep ≥25 mm vs partial 20-25 mm).

The data-driven seat (+0.0290, -0.3674) and STATE.json predicted seat (+0.0341, -0.3635)
disagree by 6.46 mm — within the known 5-17 mm CAD-chain error from CLAUDE.md. The H109
result is robust across both reference choices.

**Vote of confidence for H101 (directed sweep):** H101 doesn't need a discriminator — it
forces productive search regardless of where contact landed. The negative discrimination
result strengthens the case for H101 as the top staged-patch candidate.

Iteration: `analysis/iterations/discovery/005-h109-initial-xy-offset/`

---

## §12 — Canonical A/B pair: matched-compliance gap (added 2026-05-05, discovery 008)

First A/B pair with **schema v1.2 sidecars** (joints_raw, native-rate wrench, per-tick cmd_wrench, fm_events):

- **GOLD** `insert_u_orange_20260505_193645` — operator-driven success
- **FAIL** `insert_u_orange_20260505_193941` — autonomous abort (FIND_HOLE STUCK)
- **Same** `force_mode_params` (gain=1.0, damping=0.7, selection_vector all-True, fz=-9N)
- **Same** FSM-commanded wrench history (verified row-by-row in cmd_wrench_raw.csv): both transition through (-9, then -6 Fz approach), then a one-tick (-6, +14)N transient, then sustained (0, +5)N Fy at gain=1.0/damping=0.3 in find_hole.

The mechanism cannot be the FSM logic itself or the commanded wrench. It must be the operator's hand. Quantifying the gap, post-contact 5s window:

| Feature | GOLD | FAIL | ratio |
|---|---|---|---|
| xy_excursion max (mm) | 15.91 | 7.01 | 2.27× |
| F_lat_base median (N) | 1.98 | 1.09 | 1.82× |
| r_cop median (mm) | **22.3** | **12.8** | 1.74× |
| v_xy median (mm/s) | 2.35 | 1.25 | 1.88× |
| operator-nudge residual (mm/s) | 3.21 | 1.15 | 2.80× |
| Time-of-divergence F_lat_base (s) | — | 0.17 | — |
| Time-of-divergence v_xy (s) | — | 0.41 | — |

**FAIL never leaves the 0-1mm depth band** (1340 samples there, 0 elsewhere); GOLD progresses through 0→1→2→5→15→30→60mm bands continuously.

**Joint-effort xcor (NEW signal from joints_raw):** GOLD j0 (base rotation) shows corr(eff_j0, |v_xy|) = +0.53 with peak |eff_j0| = 2.40 Nm. FAIL j0: 0.12 corr / 1.11 Nm. **The operator pushes through the base joint** to generate distal sweep at the wrist; force-mode admittance generates motion at the TCP frame and cannot recreate the same kinematic profile within the rule-capped 6 N lateral force budget at gain=1.0/damping=0.3.

**Promoted to I016.** Practical levers within hard-rule envelope:
- Lower `damping_factor` during find_hole from 0.3 → 0.15-0.20 to raise admittance velocity per N.
- Sustain the directed push longer at multiple angles (already in H101).
- The maximum F_lat reactions are similar in both runs (~17-18 N peak) — the difference is **sustained median**, not peak. Algorithms that achieve a couple of brief 17 N peaks aren't enough; the median post-contact F_lat needs to stay near 2 N for several seconds.

Iteration: `analysis/iterations/discovery/008-v12-canonical-pair-diff/`
