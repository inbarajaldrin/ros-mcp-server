# Ralph Loop — Insert Algorithm Reverse-Engineering (Staged-Patch Model)

> **AUTO-TRIGGER:** before any iteration on this loop, READ the project skill at `compliant_insertion_studio/.claude/skills/insertion-control-law-derivation/SKILL.md`. It contains the canonical methodology for data-derived control-law derivation, the trust hierarchy, the 3-way pose comparison procedure, and the 10 anti-patterns that must not be repeated. The instructions in this PROMPT.md are subordinate to that skill — if anything here contradicts it, the skill wins.

You are iterating to decode a force-compliant peg-in-hole primitive from operator-demo telemetry. The robot is unavailable to you. Your job is to produce a **queue of FSM patches**, ranked by evidence, that will work first-try when the operator returns to the robot.

## Modes

- **Discovery iteration** (no FSM change): pick one open question, run/write an analysis script, surface a new invariant or refute a candidate. Most iterations are this.
- **Staged-patch iteration** (proposes an FSM change but does NOT apply it): when ≥2 invariants justify a specific FSM change, write a `staged/<NNN>-<slug>/` entry containing a git diff + a Python approximation of the change for replay simulation. Replay against 132 FAIL traces. Score evidence. Place in queue.

## Iron rules — never violate

1. **DO NOT modify `compliant_insertion_studio/wrapper/` directly.** All FSM changes are written as `iterations/staged/<NNN>-<slug>/PATCH.diff`. The operator applies the patch at-robot. Validation happens at-robot, never in this loop.
2. **DO NOT modify `compliant_insertion_studio/configs/`.** Same reason — that's an FSM-runtime config; modifications are part of `PATCH.diff`.
3. **DO NOT use FSM stdout claims as ground-truth labels.** See `STATE.json:trust_hierarchy`. The CSV is the only source of truth.
4. **DO NOT lock XY via `selection_vector`** (REFUTED v91). This rule is about the **translational**
   axes only: XY must stay force-controlled so the peg can find the chamfer. It is **not** a
   licence to make all six DOFs compliant. **Rotation stays LOCKED in SEARCH** —
   `selection_vector = (True, True, True, False, False, False)`. Unlocking it broke the insert for
   five consecutive real-arm runs on 2026-08-16: with rotation compliant, lateral force applies a
   moment about the grasp point, the part pivots in the jaws while the gripper translates, and TCP
   displacement stops being peg displacement — so every swept-area figure computed from TCP is
   fiction. `SKILL.md` §10's unqualified "all-True" phrasing is wrong for SEARCH; see the
   correction block there.
5. **DO NOT use counter-residual direction for force corrections** (use TOWARD-seat per I003).
6. **DO NOT exceed `cmd_fz = -9 N` or `|cmd_F_lat| = 6 N`.**
7. **DO NOT remove the state-independent global seat detector** (validated v87).
8. **DO NOT re-test a hypothesis already in `tried_and_refuted` or `v82_v97_iteration_history.json` REFUTED list.**
9. **All primitive subprocesses use `python3 -m primitives.X` module mode.**

## Per-iteration workflow

1. **Read** in this order:
   - `PROMPT.md` (this file)
   - `STATE.json` — `mode`, `known_invariants`, `tried_and_refuted`, `pending_hypotheses_discovery`, `open_questions`, `convergence_criteria`, `trust_hierarchy`
   - `FINDINGS.md` — narrative invariants
   - `v82_v97_iteration_history.json` — frozen prior-session record
   - `iterations/discovery/` (latest) and `iterations/staged/` (current queue)

2. **Decide iteration type:**
   - If `STATE.json:open_questions` is non-empty → **discovery iteration** on one of them.
   - If a `pending_hypotheses_discovery` entry has ≥2 backing `known_invariants` AND no `staged/` entry exists for it → **staged-patch iteration**.
   - If neither → check convergence; if all Phase A criteria met, write `<promise>RALPH_CONVERGED</promise>` to a final iteration's `RESULTS.md` and stop.

3. **Discovery iteration steps:**
   1. Create `iterations/discovery/<NNN>-<slug>/HYPOTHESIS.md` (use the template in `discovery/_TEMPLATE`).
   2. Write or extend a script under `analysis/scripts/NN_*.py` (NN ≥ 10) that answers the question.
   3. Run it.
   4. Write `iterations/discovery/<NNN>-<slug>/FINDING.md` and `metrics.json`.
   5. If a portable invariant emerged: append to `STATE.json:known_invariants` with new ID `I0NN`, evidence link, `discovered` date, `portability` field.
   6. If the open question resolved: remove it from `STATE.json:open_questions`.
   7. If a new pending hypothesis surfaced: append to `STATE.json:pending_hypotheses_discovery`.

4. **Staged-patch iteration steps:**
   1. Create `iterations/staged/<NNN>-<slug>/JUSTIFICATION.md` (use template).
   2. Write `PATCH.diff` — a `git diff` against `compliant_insertion_studio/wrapper/` or `configs/`. Use `git diff --no-color` from a temporary working tree if you need to construct it; do NOT actually apply the change.
   3. Write `cmd_function.py` — a Python function `cmd_wrench(state) -> (fx, fy, fz, tx, ty, tz)` that approximates the patch's runtime behavior for replay. State dict contains: `t_s`, `t_since_contact_s`, `tcp_xy_m`, `tcp_z_m`, `fz_t_N`, `F_lat_base_N`, `z_drop_mm`, `tilt_deg`, `phase_idx`, `hole_xy_m`.
   4. Run replay: `python3 analysis/scripts/08_replay_simulator.py iterations/staged/<NNN>-<slug>/`
   5. Run score: `python3 analysis/scripts/09_score_staged_patch.py iterations/staged/<NNN>-<slug>/`
   6. Update `JUSTIFICATION.md` with the Hard-rule compliance check filled in (one line per rule, "OK" or "VIOLATES because …").

5. **Commit** the iteration's artifacts (analysis-only changes). Never commit `wrapper/` or `configs/` from this loop.

## Headline metric

`durable_collapse_rate` ≡ fraction of episodes with `|Fz_t (smoothed 0.5s)| < 2 N AND dz/dt < -2 mm/s` sustained ≥0.5 s post-contact. Defined in FINDINGS §4. Validated 98% recall.

## Convergence

**Phase A (this loop's job — away-from-robot):** STOP by `touch iterations/.RALPH_CONVERGED` (a flag file the bash loop watches for) when ALL of:
- `open_questions` is empty (or remaining items are explicitly tagged "needs new robot data")
- ≥1 `iterations/staged/<NNN>-<slug>/` has `evidence_score.json:confidence == "high"`
- All current `pending_hypotheses_discovery` entries are either staged or marked unstageable (with reason)
- `known_invariants` cover ≥3 of (u_orange, u_brown, line_green, inv_u_yellow) — portability sanity check

**DO NOT write the literal string `<` `promise>RALPH_CONVERGED<` `/promise>` (or any close variant) into ANY file as narrative or example.** Only the bash-watched flag file `iterations/.RALPH_CONVERGED` triggers convergence. Earlier prompts had a string-grep trigger that was poisoned by self-referential prose; the flag file is the only authoritative signal now.

**Phase B (operator's job — at-robot, NOT in this loop):** operator applies the top-ranked staged patch via `ralph.sh apply <name>`, runs ≥5 attempts, scores. Pass → `validated/`. Fail → fresh CSVs feed Phase A's next round.

## Stop early if

- 5 consecutive iterations produce no new invariant AND no new staged patch
- A staged patch's `evidence_score.json:rule_violations > 0` after self-check — fix or discard before continuing
- An open question is provably unanswerable from existing data — tag it `needs_robot_data` and proceed

## Hand-off rule

Every staged patch must be self-contained: reading just `JUSTIFICATION.md`, `PATCH.diff`, `REPLAY.md`, `evidence_score.json` should be sufficient for the operator (or a fresh agent) to apply, validate, and decide.
