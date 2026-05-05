#!/usr/bin/env python3
# Reference: rank all staged/ entries by evidence_score so operator works queue top-down at-robot.
# Prints a one-line summary per staged dir + the top-N to consider applying first.
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from _paths import ITERATIONS_DIR

def main():
    staged_root = os.path.join(str(ITERATIONS_DIR), "staged")
    rows = []
    for d in sorted(glob.glob(os.path.join(staged_root, "*"))):
        if not os.path.isdir(d): continue
        if os.path.basename(d).startswith("_"): continue
        sp = os.path.join(d, "evidence_score.json")
        if not os.path.exists(sp):
            rows.append({"name": os.path.basename(d), "score": None, "confidence": "no_score"})
            continue
        s = json.load(open(sp))
        rows.append({"name": os.path.basename(d),
                     "score": s.get("score"),
                     "confidence": s.get("confidence"),
                     "n_inv": s.get("n_backing_invariants", 0),
                     "f_band": s.get("replay", {}).get("fraction_in_gold_band") if s.get("replay") else None,
                     "violations": s.get("rule_violations", 0)})
    rows.sort(key=lambda r: (-(r["score"] or -999), r["name"]))
    print(f"{'rank':>4} {'name':40s} {'score':>7} {'conf':>8} {'n_inv':>5} {'f_band':>7} {'viol':>5}")
    for i, r in enumerate(rows, 1):
        score = f"{r['score']:.2f}" if r["score"] is not None else "-"
        f_band = f"{r['f_band']:.2f}" if r.get("f_band") is not None else "-"
        print(f"{i:>4} {r['name']:40s} {score:>7} {r['confidence']:>8} {r.get('n_inv','-'):>5} {f_band:>7} {r.get('violations','-'):>5}")
    if rows and rows[0]["score"] is not None:
        print(f"\nNext to apply at-robot: {rows[0]['name']}")

if __name__ == "__main__":
    main()
