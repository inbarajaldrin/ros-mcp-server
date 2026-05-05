# Discovery iteration — &lt;NNN&gt;-&lt;slug&gt;

## Question
The single open question this iteration tries to answer. Cite STATE.json:open_questions or describe a residual the current model fails to explain.

## Method
What script (analysis/scripts/NN_*.py) was run, on what data subset, with what parameters.

## Result
The answer. Either:
- **New invariant** → also append to STATE.json:known_invariants with new ID + evidence link
- **Open question resolved** → also remove from STATE.json:open_questions
- **New pending hypothesis surfaced** → also append to STATE.json:pending_hypotheses_discovery
- **Inconclusive** → describe what additional data or analysis would settle it

## Files written
- `FINDING.md` — narrative result
- `metrics.json` — machine-readable summary
