#!/usr/bin/env bash
# Reference: Ralph-loop runner for the staged-patch model.
#
# Two modes:
#   ralph.sh discover [max_iter]   -- away-from-robot. Pure analysis. Each iteration: invokes claude with
#                                     PROMPT.md, claude reads STATE.json + FINDINGS.md + iterations/, picks
#                                     ONE open question or pending hypothesis, runs/writes a discovery
#                                     iteration OR a staged patch (with cmd_function.py + PATCH.diff),
#                                     runs replay+score on staged patches, commits artifacts, loops.
#                                     STOPS when all open_questions resolved AND >=1 staged patch with
#                                     confidence=high. Or when max_iter reached.
#
#   ralph.sh apply <staged_name>   -- at-robot. Prints operator instructions for applying the named staged
#                                     patch, running fresh batch, and promoting to validated/.
#                                     This is the OPERATOR step — not an autonomous run.
#
# Completion promise: an iteration writes <promise>RALPH_CONVERGED</promise> into its iteration dir's
# RESULTS.md when convergence criteria from STATE.json are met.
#
# This loop never modifies compliant_insertion_studio/wrapper/ directly. Output is staged/ patches only.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ANALYSIS_DIR/../.." && pwd)"
PROMPT_FILE="$ANALYSIS_DIR/PROMPT.md"
STATE_FILE="$ANALYSIS_DIR/STATE.json"
ITER_ROOT="$ANALYSIS_DIR/iterations"
# Convergence is signaled by a flag file the agent touches when ALL phase-A criteria are met.
# (The previous string-grep approach was poisoned by narrative prose containing the literal tag.)
CONVERGED_FLAG="$ITER_ROOT/.RALPH_CONVERGED"

mode="${1:-help}"

case "$mode" in
  help|"-h"|"--help")
    cat <<EOF
Usage:
  $0 discover [max_iter]       Run away-from-robot discovery loop (default max_iter=20)
  $0 apply <staged_name>       Print operator instructions for at-robot validation
  $0 status                    Show current convergence state + ranked staged queue
  $0 sanity                    Verify pipeline + STATE.json + scripts before launching loop

Files the loop reads:
  $PROMPT_FILE
  $STATE_FILE
  $ANALYSIS_DIR/FINDINGS.md
  $ANALYSIS_DIR/v82_v97_iteration_history.json
  $ITER_ROOT/discovery/*  $ITER_ROOT/staged/*  $ITER_ROOT/validated/*

Files the loop writes:
  $ITER_ROOT/discovery/<NNN>-<slug>/   (new discovery iterations)
  $ITER_ROOT/staged/<NNN>-<slug>/      (proposed FSM patches; NOT yet applied)
  Updates to $STATE_FILE and $ANALYSIS_DIR/FINDINGS.md
EOF
    ;;

  status)
    echo "=== convergence state ==="
    python3 -c "
import json
s = json.load(open('$STATE_FILE'))
print('mode:', s.get('mode'))
print('best_known durable_collapse_rate:', s.get('best_known', {}).get('durable_collapse_rate'))
print('open questions:', len(s.get('open_questions', [])))
print('pending hypotheses:', len(s.get('pending_hypotheses_discovery', [])))
print('known invariants:', len(s.get('known_invariants', [])))
"
    echo
    echo "=== staged patch queue (ranked) ==="
    python3 "$SCRIPT_DIR/rank_staged.py"
    ;;

  sanity)
    echo "=== sanity ==="
    [ -f "$PROMPT_FILE" ] && echo "PROMPT.md ✓" || echo "PROMPT.md ✗"
    [ -f "$STATE_FILE" ] && echo "STATE.json ✓" || echo "STATE.json ✗"
    [ -f "$ANALYSIS_DIR/FINDINGS.md" ] && echo "FINDINGS.md ✓" || echo "FINDINGS.md ✗"
    [ -d "$ANALYSIS_DIR/data" ] && [ -f "$ANALYSIS_DIR/data/summaries.json" ] && echo "data/summaries.json ✓" || echo "data/summaries.json ✗ — run scripts/run_all.py first"
    for s in 01_extract.py 02_bin_by_depth.py 06_discriminator.py 07_align_on_contact.py 08_replay_simulator.py 09_score_staged_patch.py rank_staged.py score_iteration.py; do
      [ -f "$SCRIPT_DIR/$s" ] && echo "scripts/$s ✓" || echo "scripts/$s ✗"
    done
    ;;

  discover)
    max_iter="${2:-20}"
    # Default: Claude Code in print (non-interactive) mode with permission bypass for unattended use.
    # Override examples:
    #   RALPH_CMD='codex exec' bash ralph.sh discover     # use Codex CLI instead
    #   RALPH_CMD='claude -p --dangerously-skip-permissions --model claude-sonnet-4-6' bash ralph.sh discover
    : "${RALPH_CMD:=claude -p --dangerously-skip-permissions}"
    : "${RALPH_PER_ITER_BUDGET_USD:=2.00}"
    echo "=== discover loop: max_iter=$max_iter ==="
    echo "    cmd: $RALPH_CMD"
    echo "    per-iter budget cap: \$$RALPH_PER_ITER_BUDGET_USD"

    if ! command -v "${RALPH_CMD%% *}" >/dev/null 2>&1; then
      echo "command not found on PATH: ${RALPH_CMD%% *}"; exit 2
    fi

    iter=0
    cd "$REPO_ROOT" || exit 2
    while [ "$iter" -lt "$max_iter" ]; do
      iter=$((iter + 1))
      echo
      echo "=========================="
      echo " ITERATION $iter / $max_iter"
      echo " $(date -Iseconds)"
      echo "=========================="
      # Pass PROMPT.md as the prompt argument; budget caps each iteration.
      $RALPH_CMD --max-budget-usd "$RALPH_PER_ITER_BUDGET_USD" "$(cat "$PROMPT_FILE")" || {
        rc=$?
        echo "iteration $iter: claude exit $rc — continuing"
      }
      # Refresh derived data so newly added iterations are reflected
      python3 "$SCRIPT_DIR/run_all.py" >/dev/null 2>&1 || true
      # Check convergence: a flag file (not a string match) so narrative prose can't trigger it.
      if [ -f "$CONVERGED_FLAG" ]; then
        echo
        echo "$CONVERGED_FLAG present → converged in $iter iterations"
        break
      fi
      # Auto-commit analysis-only artifacts
      cd "$REPO_ROOT"
      git add compliant_insertion_studio/analysis 2>/dev/null
      git diff --cached --quiet || git commit -m "ralph discover iter $iter" >/dev/null 2>&1 || true
    done
    echo
    echo "=== final status ==="
    "$0" status
    ;;

  apply)
    name="${2:-}"
    if [ -z "$name" ]; then
      echo "usage: $0 apply <staged_name>"; exit 2
    fi
    staged_dir="$ITER_ROOT/staged/$name"
    if [ ! -d "$staged_dir" ]; then
      echo "no such staged dir: $staged_dir"; exit 2
    fi
    if [ ! -f "$staged_dir/PATCH.diff" ]; then
      echo "missing $staged_dir/PATCH.diff"; exit 2
    fi
    obj="$(grep -oP 'object[: ]+\K[a-z_]+' "$staged_dir/JUSTIFICATION.md" 2>/dev/null | head -1)"
    obj="${obj:-u_orange}"
    today="$(date +%Y%m%d)"

    cat <<EOF
=== AT-ROBOT validation steps for staged/$name ===

1.  cd $REPO_ROOT
    git apply $staged_dir/PATCH.diff

2.  Verify ROS2 bringup is up + URCap on PLAY.

3.  Run loop_iterate (≥5 attempts):
    nohup python3 -m compliant_insertion_studio.scripts.loop_iterate \\
        --object-name $obj --base-name base1 --grasp-id 1 \\
        --target-success-count 5 \\
        > /tmp/loop_${name}.log 2>&1 &
    disown

4.  After completion (or aggressive stuck aborts), score the iteration:
    python3 $SCRIPT_DIR/run_all.py
    python3 $SCRIPT_DIR/score_iteration.py $staged_dir \\
        --csv-glob 'insert_${obj}_${today}_*.csv'

5.  If exit 0 (held or improved durable_collapse_rate):
       a) mv $staged_dir $ITER_ROOT/validated/$name
       b) git add compliant_insertion_studio/wrapper compliant_insertion_studio/analysis
       c) git commit -m "ralph apply $name (validated)"
    If exit 1 (regression):
       a) cd $REPO_ROOT && git checkout -- compliant_insertion_studio/wrapper compliant_insertion_studio/configs
       b) Move $staged_dir/metrics.json into $staged_dir/ROBOT_REFUTED.json
       c) Add an entry to STATE.json:tried_and_refuted with evidence pointing here
       d) Operator: trigger another discovery loop with this new failure data
EOF
    ;;

  *)
    echo "unknown mode: $mode"; "$0" help; exit 2
    ;;
esac
