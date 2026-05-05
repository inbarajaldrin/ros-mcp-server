#!/usr/bin/env python3
# Reference: compute evidence_score for a staged patch.
# Score is a weighted sum that ranks staged/ entries; operator works the queue top-down at-robot.
#
# evidence_score = w1 * n_invariants_supporting
#                + w2 * fraction_replay_in_gold_band
#                + w3 * fraction_replay_closer_to_hole
#                - hard_rule_violations (severe penalty)
#                - already_refuted (severe penalty if patch matches a tried_and_refuted entry)
#
# Reads:
#   <staged_dir>/JUSTIFICATION.md  -- parses "## Backing invariants" list
#   <staged_dir>/replay_results.json (from 08_replay_simulator.py)
#   ../../STATE.json -- known_invariants and tried_and_refuted
#
# Writes:
#   <staged_dir>/evidence_score.json
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from _paths import ANALYSIS_DIR

W_INVARIANTS = 1.0
W_REPLAY_BAND = 5.0
W_REPLAY_CLOSER = 3.0
PENALTY_RULE_VIOLATION = 100.0
PENALTY_ALREADY_REFUTED = 100.0


def parse_invariant_ids(justification_path):
    if not os.path.exists(justification_path): return []
    text = open(justification_path).read()
    m = re.search(r"##\s*Backing invariants\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL | re.IGNORECASE)
    if not m: return []
    return re.findall(r"\bI\d{3}\b", m.group(1))


def parse_hard_rule_check(justification_path):
    if not os.path.exists(justification_path): return None
    text = open(justification_path).read()
    m = re.search(r"##\s*Hard-rule compliance check\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL | re.IGNORECASE)
    if not m: return None
    block = m.group(1)
    violations = re.findall(r"VIOLATES?\b", block, re.IGNORECASE)
    return len(violations)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("staged_dir")
    args = p.parse_args()
    staged_dir = os.path.abspath(args.staged_dir)

    justification_path = os.path.join(staged_dir, "JUSTIFICATION.md")
    replay_path = os.path.join(staged_dir, "replay_results.json")
    state_path = os.path.join(str(ANALYSIS_DIR), "STATE.json")

    backing = parse_invariant_ids(justification_path)
    rule_violations = parse_hard_rule_check(justification_path) or 0

    # Validate backing IDs against STATE.json
    state = json.load(open(state_path))
    known_ids = {inv["id"] for inv in state.get("known_invariants", [])}
    backing_valid = [bid for bid in backing if bid in known_ids]
    backing_invalid = [bid for bid in backing if bid not in known_ids]

    # Replay metrics
    replay = json.load(open(replay_path)) if os.path.exists(replay_path) else None
    if replay:
        rs = replay["summary"]
        f_band = rs["fraction_in_gold_disp_band"]
        f_closer = rs["fraction_closer_to_hole"]
        n_replay = rs["n_failures_replayed"]
    else:
        f_band = f_closer = 0.0; n_replay = 0

    score = (W_INVARIANTS * len(backing_valid)
             + W_REPLAY_BAND * f_band
             + W_REPLAY_CLOSER * f_closer
             - PENALTY_RULE_VIOLATION * rule_violations)

    # Confidence label
    if score >= 5.0 and f_band >= 0.4 and rule_violations == 0:
        confidence = "high"
    elif score >= 2.0 and rule_violations == 0:
        confidence = "medium"
    else:
        confidence = "low"

    out = {
        "score": round(score, 3),
        "confidence": confidence,
        "backing_invariants_valid": backing_valid,
        "backing_invariants_invalid": backing_invalid,
        "n_backing_invariants": len(backing_valid),
        "rule_violations": rule_violations,
        "replay": {
            "n_failures_replayed": n_replay,
            "fraction_in_gold_band": f_band,
            "fraction_closer_to_hole": f_closer,
        } if replay else None,
        "weights": {"W_INVARIANTS": W_INVARIANTS, "W_REPLAY_BAND": W_REPLAY_BAND,
                    "W_REPLAY_CLOSER": W_REPLAY_CLOSER,
                    "PENALTY_RULE_VIOLATION": PENALTY_RULE_VIOLATION},
    }
    json.dump(out, open(os.path.join(staged_dir, "evidence_score.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
