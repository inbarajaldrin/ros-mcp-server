"""
11_line_green_seat_depth.py
Resolve open_question Q3: is line_green's ~7mm final descent geometric (shallow slot)
or algorithmic (premature seat detection)?

Method: aggregate final_z_drop_mm across all GOLD (May-3) operator demos by object.
If line_green clusters tightly at a fixed depth across 20 independent demos, the
operator hit a hard geometric floor. If it scattered, some other mechanism is gating.

Output: per-object {n, min, p25, median, p75, max, n_below_5mm} → metrics.json next to
this script's iteration dir.
"""
import json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = ROOT / "data/summaries.json"
OUT = ROOT / "iterations/discovery/002-line-green-geometric-floor/metrics.json"

with open(SUMMARIES) as f:
    summaries = json.load(f)

by_object = {}
for s in summaries:
    if "20260503" not in s.get("csv_path", ""):
        continue  # GOLD only
    by_object.setdefault(s["object"], []).append(s["final_z_drop_mm"])

def quartiles(xs):
    xs = sorted(xs)
    n = len(xs)
    p25 = xs[n // 4]
    p75 = xs[(3 * n) // 4]
    return p25, p75

result = {}
for obj, fz in sorted(by_object.items()):
    p25, p75 = quartiles(fz)
    # Drop outliers <5mm (likely partial/aborted demos labeled success): cluster of full inserts
    full = [v for v in fz if v >= 5.0]
    full_p25, full_p75 = quartiles(full) if len(full) >= 4 else (None, None)
    result[obj] = {
        "n": len(fz),
        "min": round(min(fz), 2),
        "p25": round(p25, 2),
        "median": round(statistics.median(fz), 2),
        "p75": round(p75, 2),
        "max": round(max(fz), 2),
        "n_below_5mm": sum(1 for v in fz if v < 5.0),
        "n_full_descent": len(full),
        "full_descent_min": round(min(full), 2) if full else None,
        "full_descent_max": round(max(full), 2) if full else None,
        "full_descent_p25": round(full_p25, 2) if full_p25 else None,
        "full_descent_p75": round(full_p75, 2) if full_p75 else None,
        "full_descent_iqr": round(full_p75 - full_p25, 2) if full_p25 else None,
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)

for obj, r in result.items():
    print(f"{obj}: n={r['n']} median={r['median']} mm  "
          f"full[{r['n_full_descent']}]={r['full_descent_min']}-{r['full_descent_max']} mm  "
          f"IQR={r['full_descent_iqr']}")
print(f"\nwrote {OUT}")
