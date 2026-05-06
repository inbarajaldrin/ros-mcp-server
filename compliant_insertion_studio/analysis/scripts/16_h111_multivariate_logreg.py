"""Discovery 007: Test H111 — does a multivariate logistic regression on the
top-5 univariate features from discovery 006 cross AUC>=0.75?

Direct successor to discovery 006 (univariate H110 refuted, best AUC=0.641).
The 4/4 highest-AUC features are direction-consistent (succ peaks higher) — the
exact pattern multivariate methods exploit when residuals are independent.

Top 5 features per iterations/discovery/006-h110-post-contact-aggregates/metrics.json:
  1. max_F_lat_base_N        AUC=0.641
  2. max_r_cop_m              AUC=0.617
  3. max_abs_fz_t_N           AUC=0.616
  4. time_tilt_above_1deg_s   AUC=0.614
  5. max_tilt_deg             AUC=0.611

Method:
  - Reuse 15_h110_post_contact_aggregates.py extraction (same window, same classes).
  - Standardize features (z-score on full pooled distribution).
  - Fit logistic regression by gradient descent with L2 regularization (lambda=0.5).
    Hand-rolled (no sklearn dependency, per stack rule).
  - Leave-one-out cross-validation: for each of n=119 episodes, fit on n-1, score
    the held-out, accumulate predicted probabilities. Compute pooled AUC over
    held-out scores.
  - Also report:
    - In-sample AUC (sanity check; should exceed LOO)
    - Per-feature standardized weights (interpretation)
    - Mann-Whitney p on held-out scores

Decision rules:
  - LOO AUC >= 0.75 → STRONG, recommend staged real-time stuck classifier (H111 promoted)
  - 0.70 <= LOO AUC < 0.75 → MODERATE, candidate but below convergence bar
  - LOO AUC < 0.70 → REFUTED, elimination chain extends through multivariate space;
    only schema bump G001-G004 can unblock further discriminator discovery

Anti-leakage: same as discovery 006 — features are post-contact aggregates over
[contact_t_s, contact_t_s + 5s]. Final outcome label uses final_z_drop_mm at
end-of-episode, well past the window. No feature can monotonically reflect the
label by construction.

Usage:
    python3 16_h111_multivariate_logreg.py [out_path.json]
"""
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PER_SAMPLE_DIR  # noqa: E402

# Reuse extraction logic from discovery 006.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "h110", Path(__file__).parent / "15_h110_post_contact_aggregates.py"
)
_h110 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h110)

SUMMARIES = json.load(open(DATA_DIR / "summaries.json"))

TOP5 = [
    "max_F_lat_base_N",
    "max_r_cop_m",
    "max_abs_fz_t_N",
    "time_tilt_above_1deg_s",
    "max_tilt_deg",
]


def collect():
    succ = []
    fail = []
    for r in SUMMARIES:
        cls = _h110.classify(r)
        if cls is None:
            continue
        ct = r.get("contact_t_s")
        if ct is None or not np.isfinite(ct):
            continue
        ps = _h110.per_sample_for(r)
        if ps is None:
            continue
        win = _h110.slice_window(ps, float(ct))
        if win is None:
            continue
        ag = _h110.aggregates(win)
        feats = [ag.get(k) for k in TOP5]
        if any(v is None or not np.isfinite(v) for v in feats):
            continue
        rec = {"feats": feats, "csv": Path(r["csv_path"]).name, "outcome": cls}
        if cls == "AUTO_success":
            succ.append(rec)
        else:
            fail.append(rec)
    return succ, fail


def auc_pooled(scores, labels):
    """AUC where higher score predicts label=1."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0
    ties = 0
    for x in pos:
        wins += np.sum(neg < x)
        ties += np.sum(neg == x)
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def mannwhitney_p(scores_pos, scores_neg):
    """One-sided p (pos > neg)."""
    a = np.asarray(scores_pos, dtype=float)
    b = np.asarray(scores_neg, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined))
    ranks[order] = np.arange(1, len(combined) + 1)
    sorted_vals = combined[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U1 - mu) / sigma if sigma > 0 else 0.0
    p_gt = 1 - 0.5 * (1 + erf(z / sqrt(2)))
    return float(p_gt)


def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(X, y, lam=0.5, lr=0.1, n_iter=2000):
    """L2-regularized logistic regression by full-batch gradient descent.
    X already standardized; bias separate.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        p = sigmoid(X @ w + b)
        err = p - y
        grad_w = (X.T @ err) / n + lam * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def standardize_train_apply(X_train, X_test):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def main():
    succ, fail = collect()
    n_s, n_f = len(succ), len(fail)
    print(f"AUTO_success n={n_s}  AUTO_fail n={n_f}")
    if n_s < 5 or n_f < 5:
        print("ERROR: insufficient samples")
        sys.exit(1)

    X_succ = np.asarray([r["feats"] for r in succ], dtype=float)
    X_fail = np.asarray([r["feats"] for r in fail], dtype=float)
    X = np.vstack([X_succ, X_fail])
    y = np.concatenate([np.ones(n_s), np.zeros(n_f)])
    n = len(y)

    # In-sample AUC
    Xs, _, _, _ = standardize_train_apply(X, X)
    w_full, b_full = fit_logreg(Xs, y)
    in_sample_logits = Xs @ w_full + b_full
    in_sample_auc = auc_pooled(in_sample_logits, y)

    # LOO CV
    loo_scores = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, X_te, mu, sd = standardize_train_apply(X[mask], X[i:i + 1])
        w, b = fit_logreg(X_tr, y[mask])
        loo_scores[i] = float(X_te[0] @ w + b)
    loo_auc = auc_pooled(loo_scores, y)
    p_one_sided = mannwhitney_p(loo_scores[y == 1], loo_scores[y == 0])

    # Standardized weights from full fit (interpretation)
    weights = {name: float(w_full[i]) for i, name in enumerate(TOP5)}

    if loo_auc >= 0.75:
        decision = "STRONG"
        interpretation = (
            f"Joint LOO AUC = {loo_auc:.3f} >= 0.75. Multivariate H111 holds. "
            f"Promote to invariant; candidate for staged real-time stuck classifier "
            f"(score sliding [t-5s, t] window every tick; abort if p_fail > threshold)."
        )
    elif loo_auc >= 0.70:
        decision = "MODERATE"
        interpretation = (
            f"Joint LOO AUC = {loo_auc:.3f} ∈ [0.70, 0.75). Below convergence bar "
            f"but above noise. Consider richer feature set or different model class."
        )
    else:
        decision = "REFUTED"
        interpretation = (
            f"Joint LOO AUC = {loo_auc:.3f} < 0.70. Multivariate H111 refuted. "
            f"Combined with I012 (no median band leaving), I013 (no at-contact xy "
            f"separation), and I014 (no univariate aggregate above 0.65), this "
            f"closes the elimination chain over the currently-recorded post-contact "
            f"signal space. Schema bump G001-G004 (joint_states / native-rate wrench "
            f"/ cmd_fxy) is now the only remaining source of new signal for "
            f"u_orange success/fail discrimination."
        )

    out = {
        "object": "u_orange",
        "session_tag": "20260504",
        "window_s": _h110.WINDOW_S,
        "n_AUTO_success": n_s,
        "n_AUTO_fail": n_f,
        "features": TOP5,
        "model": {
            "type": "logistic_regression",
            "regularization": {"l2_lambda": 0.5},
            "optimizer": {"method": "full_batch_gd", "lr": 0.1, "n_iter": 2000},
        },
        "in_sample_auc": in_sample_auc,
        "loo_auc": loo_auc,
        "loo_mannwhitney_one_sided_p": p_one_sided,
        "weights_standardized": weights,
        "bias_standardized_full_fit": float(b_full),
        "thresholds": {"strong": 0.75, "moderate": 0.70},
        "decision": decision,
        "interpretation": interpretation,
    }

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("metrics.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}\n")

    print("--- H111 multivariate logistic regression ---")
    print(f"in-sample AUC = {in_sample_auc:.3f}")
    print(f"LOO AUC       = {loo_auc:.3f}")
    print(f"LOO MW one-sided p = {p_one_sided:.4f}")
    print("\nstandardized weights:")
    for k, v in weights.items():
        print(f"  {k:<28} {v:+.3f}")
    print(f"\nDECISION: {decision}")
    print(f"\n{interpretation}")


if __name__ == "__main__":
    main()
