# Queued fixes

Opened **2026-08-16** from the real-arm session that re-verified FMB1 4/4.
**None of these are done.** Ordered by risk.

---

## Safety

### 1. `Fmax` / `F_press` unclamped outside the tracking branch

`SKILL.md` §10 caps `|cmd_F_lat|` at 6 N and `|Fz|` at 9 N. The 6 N clamp added on 2026-08-16
covers only `SearchDirector`'s tracking branch — the **gradient-override branch emits `Fmax`
directly**, and nothing validates it at config time.

Live, not theoretical: `primitives/translate_object.py` passes `--search-Fmax-N 8.0`, and the
handoff prescribes it. Clamp at the config boundary so every path is bounded.

### 2. `approach_fz_N = 12.0` exceeds the 9 N rule

On every insert. Either re-derive the limit or record operator approval in
`STATE.json:tried_and_refuted` per §10. Currently silently over.

---

## Correctness

### 3. `post_zero_drift_check` is always `None`

`compliant_insert.py` calls `_sample_bias(settle_s=0.0)`, which sets `self.wrench = None` then
computes `deadline = now + 0.0` — the spin loop never runs, so it returns `None` unconditionally.
Fix: `settle_s=0.05`.

Until then the third F/T calibration layer described in the docs does not exist. `--bias-abort-n`
catches an acutely bad zero; slow drift (sensor warming over a 70 s episode) goes undetected.

### 4. v4 is Fz-only with the z-check as a downstream veto

It fires on `|fz|` collapse; INSERT_DESCENT rejects it ~1 s later if the peg hasn't descended.
Measured on u_brown: repeated false fires at the same rim site, each costing an excursion.

Proposed shape (from the 2026-08-16 GPT review): a `HOLE_CANDIDATE` state that freezes θ, zeroes
lateral force, keeps Z compliant, and confirms on actual Z evidence before committing — with a
robust *local* Z baseline plus noise-derived significance, **not** a fixed `dz ≥ 0.3 mm` gate,
since background dip already reaches 0.39 mm. Bound it with a spatial refractory of one
detection-basin diameter after a rejected candidate.

### 5. Spiral pitch (2 mm) is coarser than the sensing radius (0.5–1.0 mm)

Force contrast vs distance from the true hole, contact samples only, rep3 (seated):

| dist | mean \|fz\| |
|---|---|
| 0.0 mm | 1.79 N |
| 0.5 mm | 3.77 N |
| 1.0 mm | 4.67 N |
| 2–7 mm | 4.6–5.6 N (background ~5.1, scatter ±0.5) |

The hole is a 65% force drop at 0 mm, 26% at 0.5 mm, and noise-level by 1.0 mm. With 2 mm pitch
the peg can pass 1.3 mm away and register nothing, so capture is probabilistic on a *fixed*
grasp — measured 26 s / 33 s / 54 s / never across four identical runs.

Literature agrees pitch should be ≤ the insertion clearance;
[Kang et al., IEEE RA-L 7 (2022) 6661](https://ieeexplore.ieee.org/document/9780009/) instead
shape the trajectory from an uncertainty distribution (90% hole-search success, 100% with
recovery, at 0.1 mm clearance).

> **Do not mutate `SearchDirector.pitch` in place.** Radius is derived as
> `r0 + (pitch/2π)·θ`, so reassigning pitch at θ=661° jumps the reference radius
> 5.17 → 3.34 mm and yanks the peg off the hole at the exact moment a detector would fire.
> Use a two-pass phase-shifted sweep (outward at 2 mm, inward through the gaps) or a
> separately-anchored local path.

Cost of going finer: arc scales as 1/pitch, so 1 mm doubles the sweep to ~138 s ideal.

### 6. Lag-pause conflates along-path lag with cross-track error

The peg sits pinned at exactly `lag_pause_thresh` (2.00 mm) for entire runs. That is the expected
equilibrium when the reference runs 5 mm/s against a 1.4 mm/s peg — not evidence of tracking.
Note `v_s` rate-matching is **already implemented** in the soft re-search path (`v_s = 0.001`) and
still stalls, so that is not the fix.

Replace with projection-paced progress: project TCP onto the planned path, maintain a monotonic
`s_peg`, place the target a small lookahead ahead, and track cross-track error separately since
that is what actually governs coverage.

### 7. Multiple v4 fires overwrite one `v4_predicate_fire` dict

Meta records only the last, so multi-false-fire runs can't be reconstructed post hoc.

---

## Robustness / process

### 8. `move_to_grasp` gaps

- No camera-detection retry (pre-existing; item 2 on the older open list).
- Step-2 y target sits **~11 mm off the camera object centre** for `line_green`, despite a
  `center_point` grasp with a ~0.0001 m offset. Consistent every attempt, so it predates
  2026-08-16 — but for a long thin bar that is a large fraction of the graspable width.
- `compute_duration` floors at 2.5 s, so lowering `cart_vel` for this move does nothing. Worth
  knowing before trying to slow it.

### 9. `translate_object`'s 3.0 s zero settle is only exercised on the replay path

Verified once on 2026-08-16. `run_assembly_step` uses the wrapper's own 5 s default and never
touches it.

### 10. Unify the two insertion paths

Long-standing. `prismatic_peg_insertion.py` still lives under `primitives/_real_mode_stash/`;
eventual home should be `primitives/inserts/`.

### 11. `main` is sim-only

Composite vs flat grasp ids, plus `--grasp-candidate` refusing real mode. Either finish the
candidate-native migration on hardware (verify the top-down gate and W-dynamic offset, lift the
sim-only guard, remap the assembly JSONs) or port this branch's fixes forward.

---

## Investigated and closed

- **`replay_real_assembly.py` exit code** — not a bug. It returns 1 on failure correctly. An
  earlier report of "exits 0" was an artifact of `python3 …; echo "EXIT=$?"` making the *wrapper*
  exit 0.
- **Base fixture drift** — the camera/`DEFAULT_BASE_POSITION` disagreement of 11.6 mm is the
  documented CAD-prior error, not a moved fixture. Searching at the camera-derived centre swept a
  full 8 mm radius and found nothing; the base is fixed and correct.
- **Rate-matching `v_s`** — already implemented in the soft path; still stalls.
- **Operator press duty cycle as a cross-part target** — refuted. GOLD unload duty ranges
  6.1%–35.9% across the four parts. Direction holds (successful autonomous runs unload more than
  failed ones); the level is part-specific.
