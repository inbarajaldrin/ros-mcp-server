# Staged patch — &lt;NNN&gt;-&lt;slug&gt;

## Proposed change
One paragraph in plain English.

## Backing invariants
List of `STATE.json:known_invariants` IDs (e.g. I001, I003, I007) that motivate this change. Minimum 2.

## Files in this directory
- `PATCH.diff` — `git diff` against `compliant_insertion_studio/wrapper/` (the actual change to apply at-robot)
- `cmd_function.py` — Python approximation of the proposed control function for replay simulation. Signature: `cmd_wrench(state: dict) -> (fx, fy, fz, tx, ty, tz)`. Used by `scripts/08_replay_simulator.py`.
- `REPLAY.md` — output of replay simulation against 132 FAIL traces
- `evidence_score.json` — output of `scripts/09_score_staged_patch.py`

## Hard-rule compliance check
Verify against `PROMPT.md:Hard rules`. List each rule and confirm the patch does NOT violate it.

## Predicted at-robot outcome
- Predicted `durable_collapse_rate` after applying patch
- Predicted `first_divergence_time_s`
- Confidence (low / medium / high) and why

## Operator action when at robot
1. `cd compliant_insertion_studio && git apply analysis/iterations/staged/&lt;NNN&gt;-&lt;slug&gt;/PATCH.diff`
2. Run `loop_iterate` for ≥5 attempts on active object
3. Run `python3 analysis/scripts/run_all.py && python3 analysis/scripts/score_iteration.py analysis/iterations/staged/&lt;NNN&gt;-&lt;slug&gt; --csv-glob 'insert_&lt;obj&gt;_&lt;date&gt;_*.csv'`
4. If exit 0 → promote to `validated/&lt;NNN&gt;-&lt;slug&gt;/` and commit FSM change
5. If exit 1 → revert with `git checkout -- compliant_insertion_studio/wrapper/`, mark this dir `OUTCOME=robot_refuted` in metrics
