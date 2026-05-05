"""Discovery 003: contact-Fz distribution per (object, outcome_class).

Reads summaries.json + per_sample/*.json. Re-labels outcomes from CSV-derived
final_z_drop_mm (>=20mm => success) per trust_hierarchy item 3 (do NOT trust
FSM outcome flag).

Outputs metrics.json next to this script's caller-supplied output path.
"""
import json
import sys
from pathlib import Path
from statistics import median, quantiles

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PER_SAMPLE_DIR  # noqa: E402

SUMMARIES = json.load(open(DATA_DIR / "summaries.json"))
WINDOW = 5  # +/- samples around contact_idx for smoothed Fz


def classify(s):
    """Return (object, class) where class ∈ {GOLD, AUTO_success, AUTO_fail}.

    GOLD = May-3 demo (assist_level == 'assisted' and outcome != 'abort').
    AUTO = May-4 u_orange autonomous; relabel by final_z_drop_mm.
    """
    name = Path(s["csv_path"]).name
    obj = s["object"]
    if "20260503" in name:
        return obj, "GOLD"
    if "20260504" in name:
        ok = (s.get("final_z_drop_mm") or 0) >= 20.0
        return obj, "AUTO_success" if ok else "AUTO_fail"
    return obj, "OTHER"


def fz_at_contact(s):
    ps_path = PER_SAMPLE_DIR / (Path(s["csv_path"]).stem + ".per_sample.json")
    if not ps_path.exists():
        return None
    ps = json.load(open(ps_path))
    fz = ps.get("fz_t") or []
    cmd = ps.get("commanded_fz") or []
    idx = s.get("contact_idx_active")
    if idx is None or idx < 0 or idx >= len(fz):
        return None
    lo = max(0, idx - WINDOW)
    hi = min(len(fz), idx + WINDOW + 1)
    fz_window = fz[lo:hi]
    if not fz_window:
        return None
    cmd_at_contact = cmd[idx] if 0 <= idx < len(cmd) else None
    return {
        "fz_at_contact_raw": float(fz[idx]),
        "fz_at_contact_smoothed_10s": float(np.mean(fz_window)),
        "fz_at_contact_abs": float(abs(fz[idx])),
        "fz_at_contact_smoothed_abs": float(abs(np.mean(fz_window))),
        "cmd_fz_at_contact": float(cmd_at_contact) if cmd_at_contact is not None else None,
        "object": s["object"],
        "csv": Path(s["csv_path"]).name,
    }


def stats(vals):
    if not vals:
        return {"n": 0}
    arr = np.array(vals)
    qs = np.quantile(arr, [0.25, 0.5, 0.75])
    return {
        "n": len(arr),
        "median": float(qs[1]),
        "q25": float(qs[0]),
        "q75": float(qs[2]),
        "iqr": float(qs[2] - qs[0]),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    rows = []
    for s in SUMMARIES:
        rec = fz_at_contact(s)
        if rec is None:
            continue
        obj, cls = classify(s)
        rec["class"] = cls
        rec["final_z_drop_mm"] = s.get("final_z_drop_mm")
        rows.append(rec)

    by_obj_class = {}
    for r in rows:
        key = f"{r['object']}::{r['class']}"
        by_obj_class.setdefault(key, []).append(r)

    out = {
        "n_total": len(rows),
        "groups": {},
        "raw_per_attempt": rows,
    }
    for key, group in sorted(by_obj_class.items()):
        out["groups"][key] = {
            "n": len(group),
            "fz_at_contact_smoothed_abs": stats(
                [g["fz_at_contact_smoothed_abs"] for g in group]
            ),
            "fz_at_contact_abs": stats([g["fz_at_contact_abs"] for g in group]),
            "cmd_fz_at_contact": stats(
                [g["cmd_fz_at_contact"] for g in group if g["cmd_fz_at_contact"] is not None]
            ),
        }

    # Summary banner: per-object GOLD vs AUTO_fail vs AUTO_success contrast
    out["summary_table"] = []
    for obj in sorted({r["object"] for r in rows}):
        row = {"object": obj}
        for cls in ("GOLD", "AUTO_success", "AUTO_fail"):
            key = f"{obj}::{cls}"
            if key in out["groups"]:
                row[cls] = {
                    "n": out["groups"][key]["n"],
                    "median_abs_smoothed": out["groups"][key]["fz_at_contact_smoothed_abs"]["median"],
                    "iqr_abs_smoothed": out["groups"][key]["fz_at_contact_smoothed_abs"]["iqr"],
                }
        out["summary_table"].append(row)

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("metrics.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(out["summary_table"], indent=2))


if __name__ == "__main__":
    main()
