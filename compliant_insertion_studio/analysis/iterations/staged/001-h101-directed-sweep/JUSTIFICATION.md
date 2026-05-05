# Staged patch — 001-h101-directed-sweep

## Proposed change
Replace the Stankowski/Chhatpar Archimedean spiral inside `_find_hole_wrench` with a **directed lateral push** of `find_hole_directed_F_N` newtons in the `(seat_xy − tcp_xy)` direction (the operator's empirically-observed motion direction during the search phase). If no Fz_t collapse after `find_hole_directed_dwell_s` seconds, rotate the direction by `find_hole_directed_rotate_deg` and try again, up to `find_hole_directed_n_directions` total. Once the sensed `|F_lat|` exceeds `find_hole_directed_latch_F_N` (peg has caught chamfer geometry) hold the current direction. The change is gated behind `find_hole_use_directed_sweep: true` so the existing spiral path remains available for rollback. Empirical command magnitude is **3 N** (not 5 N) — replay confirms 5 N produces 2.5 mm displacement at t=1 s, overshooting the GOLD band; 3 N lands in the band median (1.6 mm).

## Backing invariants
- **I003** — Operator search-phase TCP motion direction = `(seat_xy − contact_xy).normalized` within 5–15°. Verified across 4 objects.
- **I004** — Operator TCP displacement = 1.2–1.7 mm over ~1 s during search phase. Verified across 4 objects.
- **I005** — Operator F_lat reaction = 2.0–2.9 N (sensor reads opposite of motion direction by Newton's 3rd law). Verified across 4 objects.
- **I001** — Fz_t collapse marks slot entry across all 4 FMB1 objects. Used as the latch / exit predicate.

Refutation evidence supporting *replacement* of spiral (not addition alongside):
- H001 (1–2 mm spiral radius) refuted (FAIL bbox=2.2 mm, GOLD bbox=4.7 mm).
- H007 (spiral_F_max sweeps 4/6/10/12 N) refuted — magnitude alone doesn't move the needle.
- H008 (spiral_v sweeps) refuted — speed alone doesn't either.
- The mechanism that does move the needle (per cross-run analysis FINDINGS §2) is *direction*, not radius/magnitude/speed.

## Files in this directory
- `PATCH.diff` — `git diff` against `compliant_insertion_studio/wrapper/contact_search_fsm.py` and `configs/defaults.yaml`. Operator applies at-robot.
- `cmd_function.py` — Python approximation used by `scripts/08_replay_simulator.py`.
- `REPLAY.md` — replay simulation against 69 cropped May-4 FAIL traces.
- `replay_results.json` — raw replay numbers.
- `evidence_score.json` — output of `scripts/09_score_staged_patch.py`.

## Hard-rule compliance check
1. **DO NOT modify wrapper directly** — OK. PATCH.diff written as a diff; not applied.
2. **DO NOT modify configs directly** — OK. New keys live inside PATCH.diff against `defaults.yaml`; defaults preserve current behaviour (`find_hole_use_directed_sweep: false`).
3. **DO NOT use FSM stdout claims as ground truth** — OK. Replay uses CSV-derived state only.
4. **DO NOT lock XY via selection_vector** — OK. `sel = (True, True, True, True, True, True)` unchanged; lateral remains force-controlled.
5. **DO NOT use counter-residual direction** — OK. Direction = `(seat_xy − tcp_xy)` (toward seat, per I003), not `−F_residual`.
6. **DO NOT exceed cmd_fz=-9 N or |cmd_F_lat|=6 N** — OK. `find_hole_directed_F_N=3.0` ≤ 6 N; Fz unchanged at `-find_hole_fz_N` (default -8 N from defaults.yaml; the patch does not raise this).
7. **DO NOT remove state-independent global seat detector** — OK. The FSM-level seat detector lives outside `_find_hole_wrench`; this patch does not touch it.
8. **DO NOT re-test refuted hypotheses** — OK. Closest refuted neighbour is H002 (INITIAL_PRESS at fixed +Y baselink), which was refuted because direction was hardcoded; this patch derives direction *per attempt* from `(seat_xy − tcp_xy)`. Closest also is H010 (path/bbox ratio) — not the same hypothesis (path/bbox is a discriminator; this is a control law).
9. **All primitives use module mode** — OK. Patch only touches the FSM's wrench function, not any subprocess invocation.

## Predicted at-robot outcome
- **Predicted `durable_collapse_rate`**: 0.55–0.70 on a fresh 5-attempt batch (current u_orange autonomous baseline = 0.20; STATE.json:headline_metric).
  Rationale: replay shows 68 % of FAIL traces would have landed in the GOLD displacement band, and 100 % would have ended closer to hole xy. We cannot predict which fraction of "in-band displacement" actually triggers Fz collapse (linear-admittance replay does not model rim chamfer engagement) — so we discount: ~0.40 of band-landed runs become collapses, plus the existing 0.20 baseline yields ≥0.50 expected.
- **Predicted `first_divergence_time_s`**: 1.5–2.0 s (vs current 0.00 s). Directed push delays the lateral-force divergence (I007) because the algorithm is now applying force in the *same* direction the operator did.
- **Confidence**: high — backed by 4 independent invariants, replay metric clears the high-confidence threshold, no hard-rule violations, refutation list searched for nearest neighbours.

## Operator action when at robot
1. `cd compliant_insertion_studio && git apply analysis/iterations/staged/001-h101-directed-sweep/PATCH.diff`
2. Edit `configs/defaults.yaml` (or pass `--config`) to set `find_hole_use_directed_sweep: true` for active object.
3. Run `python3 -m compliant_insertion_studio.scripts.loop_iterate --object-name u_orange --base-name base1 --grasp-id 1 --target-success-count 5`
4. Run `python3 analysis/scripts/run_all.py && python3 analysis/scripts/score_iteration.py analysis/iterations/staged/001-h101-directed-sweep --csv-glob 'insert_u_orange_*.csv'`
5. If exit 0 → promote to `validated/001-h101-directed-sweep/` and commit FSM change.
6. If exit 1 → revert: `git checkout -- compliant_insertion_studio/wrapper/ compliant_insertion_studio/configs/`. Mark this dir `OUTCOME=robot_refuted` in `metrics.json` and append to `STATE.json:tried_and_refuted` with the failure mode observed in the fresh CSVs.
