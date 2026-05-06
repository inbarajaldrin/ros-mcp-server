# Session state — 2026-05-06 (regime-decoding work)

Handoff doc for the next agent. Read this before doing anything else in `compliant_insertion_studio/analysis/`. Skim the file paths at the bottom for full project context only after understanding what's already done.

---

## Where we are

**Mid-stream**: finalizing marker predicates BEFORE collecting more data, then collecting 13 more demos (+ session 1's 1 valid = 14 total), then running cross-demo segmenter analysis to lock in the data-driven thresholds.

The operator has a specific plan for finalizing markers that they haven't shared yet — **wait for the operator to direct the marker-finalization process**. Don't propose your own approach; the operator is leading.

## Don't redo

These are settled. **Do not** re-open them unless the operator explicitly asks:

| Item | Status | Why settled |
|---|---|---|
| `set_payload` at bringup | DONE — `launch_robot.sh` reads `result.mass_kg` + `result.cog_xyz_m` from latest `ft_calibration_*.yaml`, calls service after controller_manager up | Verified 2026-05-06: TCP drift during 19s descent dropped 7.85mm → 0.83mm (9.5×). Pendant payload does NOT propagate to ROS-side force_mode. |
| Contact detection threshold | DONE — `contact_threshold_N=3.0`, `fz_smooth_window_s=0.1`, `contact_sustain_s=0.1` in `defaults.yaml` | Verified with `--abort-on-first-contact`: detection latency ~50ms, surface_z accurate to <0.1mm of first-touch |
| `--abort-on-first-contact` flag | DONE — in `compliant_insert.py`, mutates `fsm_action.kind = "exit_abort"` so cleanup runs through the existing path | Used for marker validation runs |
| `--base-offset-xy` plumbing | DONE — in `run_assembly_step.py` and `iterate_insert.py`. Smoke-tested at 5mm and 10mm offsets, both X and Y, all 4 sides + diagonals | TCP shifts cleanly with the offset, no IK degradation up to 10mm |
| Bootstrap-from-TCP | REMOVED — it discarded R_grasp's fold info, caused fold-mirror flip across runs. Inline comment in `run_assembly_step.py:148` explains why never to re-add. | Empirical test: chained held_quat correctly recovers BOTH orientation AND fold from a 3.66° pendant offset; bootstrap flipped the fold to mirror image |
| Rotation lock in force-mode | DONE — all `selection_vector` tuples in `contact_search_fsm.py` changed from `(T,T,T,T,T,T)` to `(T,T,T,F,F,F)`. Rotation is position-controlled (locked at start orientation) throughout APPROACH/FIND_HOLE/ENTRY_SETTLE/INSERT/WEDGE_RECOVERY | Per operator's reasoning: prismatic peg + rigid grasp + working gravity comp = no rotational compliance benefit. Eliminates yaw-drift-via-grasp-lever pseudo-translation. |
| APPROACH descent speed | DONE — `approach_fz_N: 6.0 → 12.0` in `defaults.yaml` for ~2× faster descent | Per operator request: 19s descent → ~10s. Detection overshoot still <0.5mm. |
| `insert_min_descent_m` | DONE — `0.005 → 0.025` in `defaults.yaml`. FSM now rejects false-seats at the source (was letting through phantom-contact through-air drops) | Caught false-seat in demo 2 of session 1778055967. Stop-gap until cross-demo distribution available. |
| Physical-seat verifier | DONE — `_verify_physical_seat()` in `collect_regime_data.py`. Checks contact moment >1s into ACTIVE, descent 25-45mm post-contact, xy drift <15mm. Manifest carries `physical_check.status` + `use_for_analysis` flag per demo. |

## Don't reason from

These are mental-model corrections. Re-violating them wastes context:

- **TCP at canonical = peg at canonical, period.** The `fold_angle_err` field in CAD chain meta is camera-noise propagation through a synthetic R_grasp assumption, NOT a real geometric offset. Do not "compensate" for it. (Persisted in `compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md` §12.)
- **FSM stdout claims (outcome=success, reason=fsm_seated, etc.) are NOT physical truth.** They were tuned via guessed thresholds and can fire on phantom contacts (zero contamination) or peg-fell-through cases. Verify against raw CSV before accepting any FSM-claimed outcome. (Documented in `FSM_MARKERS.md` and the skill.)
- **The chained held_quat preserves R_grasp's fold info from initial perception at pick time.** It's the design. Don't add bootstrap-style overrides that synthesize a "canonical" seed — they discard the fold and produce mirror-image insertion targets.
- **`grasp_id` does not encode fold info.** It's a position offset, not a rotational selector. The fold is determined by R_grasp (set at pick time) only.

## What's pending

### Marker finalization — IN PROGRESS, operator-led

User's proposed FSM consolidation (replaces current 5-state + WEDGE_RECOVERY system):

```
States (3):  Inserting | Aligning | At Target

Markers (4):
  1. Contact      — Inserting → Aligning      (peg-bottom touches a surface)
  2. Found Hole   — Aligning → Inserting      (peg cleared rim, descending into slot)
  3. Contact      — Inserting → Aligning      (re-entry; only if NOT At Target)
  4. At Target    — Inserting → done          (peg fully seated)
```

For u_orange/base1/grasp_id=1 (simple geometry): one Contact, one Found Hole, At Target. For complex parts later: marker #3 fires multiple times (re-engaging features).

Status:
- ✓ **Contact** — finalized. Predicate: `fz_smoothed > 3N for 0.1s sustained`. Verified 2026-05-06.
- ⏳ **Found Hole** — operator has a plan, has not yet shared. WAIT for operator direction.
- ⏳ **At Target** — pending after Found Hole.
- ⏳ **Re-Contact (#3)** — same predicate as Contact, just re-evaluated after re-entering Inserting state.

Don't propose to start finalizing Found Hole / At Target on your own — the operator wants to lead.

### Data collection — BLOCKED on marker finalization

Plan once markers are finalized:
- 5 variations × 3 reps = 15 demos via `python3 -m compliant_insertion_studio.scripts.collect_regime_data`
- Variations: A_pos_x_10mm, B_neg_x_10mm, C_pos_y_10mm, D_neg_y_10mm, E_diag_pxpy_7mm
- Already have 1 valid demo from session 1778055967 (demo 1 of `A_pos_x_10mm`). Demo 2 was a false-seat (verified, marked `use_for_analysis=False`).

Operator may run collection from this terminal or via the agent briefing at `compliant_insertion_studio/analysis/AGENT_BRIEFING_DATA_COLLECTION.md` in another terminal with a fresh-context agent.

### Cross-demo regime analysis — BLOCKED on data collection

Once 14+ valid demos are in:
1. Run `analysis/scripts/30_segment_regimes.py` on each demo
2. Aggregate per-regime statistics (median direction unit vector, F_lat magnitude, commitment duration, etc.)
3. Tune detector thresholds in `30_segment_regimes.py:TH` to maximize segmentation consistency
4. Decode regime-transition triggers per `REGIME_DECODING.md` §4
5. Synthesize the regime-conditional control law per §5
6. Validate via self-consistency check (does the law reproduce GOLD's regime sequence?)
7. Translate the law into FSM code changes

Scripts to write at this point: `31_decode_operator_action.py`, `32_decode_transition_triggers.py`, `33_synthesize_law.py`, `34_validate_law.py`. Specs in `REGIME_DECODING.md` §9.

## Files relevant to this work (in priority order)

| File | Purpose |
|---|---|
| `compliant_insertion_studio/analysis/SESSION_STATE_2026-05-06.md` | This file. |
| `compliant_insertion_studio/analysis/FSM_MARKERS.md` | Catalog of all 11 current FSM markers + tuning paths. Source of truth for what's tunable. |
| `compliant_insertion_studio/analysis/REGIME_DECODING.md` | Regime hypotheses, detectors, segmentation algorithm, synthesis template, data-collection spec. |
| `compliant_insertion_studio/analysis/DATA_COLLECTION_NEEDED.md` | What demos to collect, why varied positions matter. |
| `compliant_insertion_studio/analysis/AGENT_BRIEFING_DATA_COLLECTION.md` | Hand-off briefing if collection runs in a separate fresh-context agent terminal. |
| `compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md` | Methodology skill that auto-triggers on any FSM/insertion work. Contains all hard-won lessons including §11 (fold-mirror in fixed priors) and §12 (TCP-canonical=peg-canonical). |
| `compliant_insertion_studio/scripts/collect_regime_data.py` | Interactive collection harness with `_verify_physical_seat()`. |
| `compliant_insertion_studio/scripts/launch_robot.sh` | Bringup with auto-set_payload from latest calibration YAML. |
| `compliant_insertion_studio/configs/defaults.yaml` | All FSM thresholds. Look here for current values; comments record date + reason for each change. |
| `compliant_insertion_studio/wrapper/contact_search_fsm.py` | The FSM. All `selection_vector` tuples now lock rotation. |
| `compliant_insertion_studio/wrapper/compliant_insert.py` | The wrapper. `--abort-on-first-contact` flag added. |
| `compliant_insertion_studio/scripts/run_assembly_step.py` | Orchestrator. `--base-offset-xy`, `--override-fz-cap`, `--abort-on-first-contact` flags added. Inline comment at line 148 explaining why bootstrap was removed. |
| `compliant_insertion_studio/logs/regime_collection_1778055967.json` | Manifest of session 1 (1 valid demo, 1 false-seat). |
| `compliant_insertion_studio/logs/insert_u_orange_20260504_113809.csv` | Single existing v1.2 GOLD demo. Useful reference for marker-finalization. |

## Tasks list

Use TaskList tool to see the running task ledger. Tasks #14–#24 are this session's work (all completed). Tasks #10, #11, #12 (per-regime direction decoder, transition trigger decoder, control law synthesizer) are pending — blocked on data collection.

## Most recent operator interaction

Operator proposed the 3-state / 4-marker FSM consolidation above. Said they have a plan for finalizing the Found Hole marker that they haven't yet shared. Then asked for this session-state doc to be written.

**Next operator interaction expected**: marker-finalization plan for Found Hole (and likely At Target). Wait for it. Do not start your own marker analysis.
