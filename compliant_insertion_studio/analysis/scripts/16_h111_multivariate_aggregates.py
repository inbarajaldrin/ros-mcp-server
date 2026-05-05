"""Discovery 007: Test H111 — does a multivariate linear classifier on the top
5 univariate features from discovery 006 cross AUC>=0.75 on u_orange AUTO?

Discovery 006 (I014) found 4/4 of {max_F_lat_base_N, max_r_cop_m,
max_abs_fz_t_N, max_tilt_deg} go in the same physical direction (succ peaks
higher) with 3/4 at p<0.05, plus time_tilt_above_1deg_s shows a 10x median gap.
Univariate AUC ceiling was 0.641. This iteration tests whether a linear
combination of the 5 features can amplify the cross-feature consistency into
AUC>=0.75.

Method:
  - Same data subset as discovery 006: u_orange AUTO session 20260504.
  - Class: AUTO_success (final_z_drop_mm>=20, n=19), AUTO_fail (<5, n=100).
  - Window: [contact_t_s, contact_t_s+5.0].
  - Features (top 5 from discovery 006):
      max_F_lat_base_N, max_r_cop_m, max_abs_fz_t_N,
      time_tilt_above_1deg_s, max_tilt_deg
  - Two classifiers, both hand-rolled (no sklearn):
      1. Fisher linear discriminant: w = pinv(Sigma_pooled) @ (mu_succ - mu_fail)
      2. L2-regularized logistic regression (lambda=1.0, lr=0.1, 1000 iters)
  - Validation: leave-one-out, per-fold standardization (no leakage).
  - Report aggregated held-out ROC-AUC for each classifier.

Decision:
  - AUC>=0.80 -> STRONG, stage real-time classifier patch
  - AUC in [0.70,0.80) -> MODERATE, defer
  - AUC<0.70 -> WEAK, elimination chain extends, schema bump required

Usage:
    python3 16_h111_multivariate_aggregates.py [out_path.json]
"""
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_DIR, PER_SAMPLE_DIR  # noqa: E402

SUMMARIES = json.load(open(DATA_DIR / "summaries.json"))

OBJECT = "u_orange"
SESSION_TAG = "20260504"
SUCCESS_THRESH_MM = 20.0
FAIL_THRESH_MM = 5.0
WINDOW_S = 5.0

TOP5_FEATURES = [
    "max_F_lat_base_N",
    "max_r_cop_m",
    "max_abs_fz_t_N",
    "time_tilt_above_1deg_s",
    "max_tilt_deg",
]


def classify(s):
    name = Path(s["csv_path"]).name
    if s["object"] != OBJECT or SESSION_TAG not in name:
        return None
    fdz = s.get("final_z_drop_mm") or 0.0
    if fdz >= SUCCESS_THRESH_MM:
        return "AUTO_success"
    if fdz < FAIL_THRESH_MM:
        return "AUTO_fail"
    return None


def per_sample_for(s):
    p = PER_SAMPLE_DIR / (Path(s["csv_path"]).stem + ".per_sample.json")
    if not p.exists():
        return None
    return json.load(open(p))


def slice_window(ps, contact_t_s, window_s=WINDOW_S):
    t = np.asarray(ps.get("t_s", []), dtype=float)
    if len(t) == 0:
        return None
    mask = (t >= contact_t_s) & (t <= contact_t_s + window_s)
    if mask.sum() < 10:
        return None
    out = {"t": t[mask]}
    for k in ("F_lat_base", "fz_t", "tilt_deg", "r_cop_m"):
        v = ps.get(k)
        if v is None:
            out[k] = None
        else:
            arr = np.asarray(v, dtype=float)
            if len(arr) != len(t):
                out[k] = None
            else:
                out[k] = arr[mask]
    return out


def max_run_length_above(values, t, thresh):
    if values is None or len(values) == 0:
        return 0.0
    above = values > thresh
    best = 0.0
    cur_start = None
    for i, hi in enumerate(above):
        if hi and cur_start is None:
            cur_start = t[i]
        elif not hi and cur_start is not None:
            best = max(best, float(t[i] - cur_start))
            cur_start = None
    if cur_start is not None:
        best = max(best, float(t[-1] - cur_start))
    return best


def aggregates_top5(win):
    F = win["F_lat_base"]
    fz = win["fz_t"]
    tilt = win["tilt_deg"]
    rcop = win["r_cop_m"]
    t = win["t"]
    return {
        "max_F_lat_base_N": float(np.nanmax(F)) if F is not None else float("nan"),
        "max_r_cop_m": float(np.nanmax(rcop)) if rcop is not None else float("nan"),
        "max_abs_fz_t_N": float(np.nanmax(np.abs(fz))) if fz is not None else float("nan"),
        "time_tilt_above_1deg_s": (
            max_run_length_above(tilt, t, 1.0) if tilt is not None else float("nan")
        ),
        "max_tilt_deg": float(np.nanmax(tilt)) if tilt is not None else float("nan"),
    }


def auc_from_scores(scores, labels):
    """ROC-AUC via Mann-Whitney U identity (no sklearn). Higher score = succ."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for x in pos:
        wins += np.sum(neg < x) + 0.5 * np.sum(neg == x)
    return float(wins / (len(pos) * len(neg)))


def standardize(X_train, X_test):
    """Per-fold z-score using training stats only."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def fit_lda(X, y):
    """Fisher LDA closed-form. Returns weight vector w (k,) such that succ
    has higher w.x than fail in expectation. Uses pinv to handle degeneracy."""
    pos = X[y == 1]
    neg = X[y == 0]
    mu_p = pos.mean(axis=0)
    mu_n = neg.mean(axis=0)
    # Pooled within-class covariance
    if len(pos) > 1:
        cov_p = np.cov(pos, rowvar=False, ddof=1)
    else:
        cov_p = np.zeros((X.shape[1], X.shape[1]))
    if len(neg) > 1:
        cov_n = np.cov(neg, rowvar=False, ddof=1)
    else:
        cov_n = np.zeros((X.shape[1], X.shape[1]))
    n1, n0 = len(pos), len(neg)
    pooled = ((n1 - 1) * cov_p + (n0 - 1) * cov_n) / max(n1 + n0 - 2, 1)
    # Ridge for numerical stability (k is small, 5)
    pooled = pooled + 1e-6 * np.eye(pooled.shape[0])
    w = np.linalg.pinv(pooled) @ (mu_p - mu_n)
    return w, float(0.5 * (w @ mu_p + w @ mu_n))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def fit_logreg(X, y, lam=1.0, lr=0.1, iters=1000):
    """L2-regularized logistic regression by gradient descent. Returns
    (beta, b)."""
    n, k = X.shape
    beta = np.zeros(k)
    b = 0.0
    yf = y.astype(float)
    for _ in range(iters):
        z = X @ beta + b
        p = sigmoid(z)
        grad_beta = X.T @ (p - yf) / n + lam * beta / n
        grad_b = float((p - yf).mean())
        beta -= lr * grad_beta
        b -= lr * grad_b
    return beta, float(b)


def loo_lda(X, y):
    n = len(y)
    scores = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xtr_raw = X[mask]
        ytr = y[mask]
        Xte_raw = X[i:i + 1]
        Xtr, Xte, _, _ = standardize(Xtr_raw, Xte_raw)
        if (ytr == 1).sum() < 2 or (ytr == 0).sum() < 2:
            scores[i] = 0.0
            continue
        w, thr = fit_lda(Xtr, ytr)
        scores[i] = float((Xte @ w)[0])
    return scores


def loo_logreg(X, y, lam=1.0, lr=0.1, iters=1000):
    n = len(y)
    scores = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xtr_raw = X[mask]
        ytr = y[mask]
        Xte_raw = X[i:i + 1]
        Xtr, Xte, _, _ = standardize(Xtr_raw, Xte_raw)
        if (ytr == 1).sum() < 2 or (ytr == 0).sum() < 2:
            scores[i] = 0.5
            continue
        beta, b = fit_logreg(Xtr, ytr, lam=lam, lr=lr, iters=iters)
        scores[i] = float(sigmoid((Xte @ beta + b)[0]))
    return scores


def fit_lda_full(X, y):
    """Fit on all data — for reporting feature contributions only, not for AUC."""
    Xz, _, mu, sd = standardize(X, X[:0])
    w, thr = fit_lda(Xz, y)
    return {
        "feature_means_train": mu.tolist(),
        "feature_stds_train": sd.tolist(),
        "lda_weight_standardized": w.tolist(),
        "lda_threshold_standardized": thr,
        "feature_names": TOP5_FEATURES,
    }


def fit_logreg_full(X, y, lam=1.0):
    Xz, _, mu, sd = standardize(X, X[:0])
    beta, b = fit_logreg(Xz, y, lam=lam)
    return {
        "feature_means_train": mu.tolist(),
        "feature_stds_train": sd.tolist(),
        "logreg_beta_standardized": beta.tolist(),
        "logreg_bias_standardized": b,
        "feature_names": TOP5_FEATURES,
    }


def main():
    succ_rows = []
    fail_rows = []
    skipped = {"no_per_sample": 0, "no_window": 0, "wrong_class": 0,
               "no_contact_t_s": 0, "nan_features": 0}

    for r in SUMMARIES:
        cls = classify(r)
        if cls is None:
            skipped["wrong_class"] += 1
            continue
        ct = r.get("contact_t_s")
        if ct is None or not np.isfinite(ct):
            skipped["no_contact_t_s"] += 1
            continue
        ps = per_sample_for(r)
        if ps is None:
            skipped["no_per_sample"] += 1
            continue
        win = slice_window(ps, float(ct))
        if win is None:
            skipped["no_window"] += 1
            continue
        ag = aggregates_top5(win)
        vec = [ag[f] for f in TOP5_FEATURES]
        if not all(np.isfinite(v) for v in vec):
            skipped["nan_features"] += 1
            continue
        row = (vec, Path(r["csv_path"]).name)
        if cls == "AUTO_success":
            succ_rows.append(row)
        else:
            fail_rows.append(row)

    print(f"AUTO_success n={len(succ_rows)}  AUTO_fail n={len(fail_rows)}")
    print(f"skipped={skipped}")

    if len(succ_rows) < 5 or len(fail_rows) < 5:
        print("ERROR: insufficient samples")
        sys.exit(1)

    X = np.asarray([r[0] for r in succ_rows] + [r[0] for r in fail_rows],
                   dtype=float)
    y = np.asarray([1] * len(succ_rows) + [0] * len(fail_rows), dtype=int)

    print("\n--- LOO LDA ---")
    lda_scores = loo_lda(X, y)
    lda_auc = auc_from_scores(lda_scores, y)
    print(f"LDA LOO AUC = {lda_auc:.4f}")

    print("\n--- LOO L2-LogReg ---")
    lr_scores = loo_logreg(X, y, lam=1.0, lr=0.1, iters=1000)
    lr_auc = auc_from_scores(lr_scores, y)
    print(f"LogReg LOO AUC = {lr_auc:.4f}")

    # Sanity: best univariate AUC on the same n=119 subset (replicates D006).
    print("\n--- Univariate AUCs (same n=119 subset, for sanity) ---")
    univ = {}
    for j, fname in enumerate(TOP5_FEATURES):
        col = X[:, j]
        a_hi = auc_from_scores(col, y)
        a_lo = auc_from_scores(-col, y)
        a = max(a_hi, a_lo)
        univ[fname] = a
        print(f"  {fname:<28} AUC={a:.4f}")

    # Feature contributions (full-data fit, for interpretation only)
    lda_full = fit_lda_full(X, y)
    lr_full = fit_logreg_full(X, y)

    # Decision
    best = max(lda_auc, lr_auc)
    if best >= 0.80:
        verdict = "STRONG"
        interpretation = (
            f"STRONG: best LOO AUC={best:.3f} >= 0.80. Multivariate signal "
            f"exists in recorded features. Stage a real-time stuck classifier "
            f"patch using {('LDA' if lda_auc >= lr_auc else 'LogReg')} weights."
        )
    elif best >= 0.70:
        verdict = "MODERATE"
        interpretation = (
            f"MODERATE: best LOO AUC={best:.3f} in [0.70, 0.80). Joint signal "
            f"exists but is too weak for a real-time gate. Defer to multivariate-"
            f"with-more-features (H112) before staging a patch."
        )
    else:
        verdict = "WEAK"
        interpretation = (
            f"WEAK: best LOO AUC={best:.3f} < 0.70. The 5-feature linear "
            f"combination does NOT discriminate. Combined with I012 (no "
            f"per-timepoint median band leaving), I013 (no at-contact xy "
            f"separation), and I014 (no univariate aggregate >= 0.65), the "
            f"elimination chain now extends through multivariate space on the "
            f"top-5 features. The success/fail discriminator is NOT in the "
            f"currently-recorded features — the discriminator must lie in "
            f"signals not yet logged. Promote G001-G004 (joint_states / "
            f"native-rate wrench / per-sample 6-axis cmd / "
            f"force_mode_controller state) from PROPOSED to REQUIRED-PRE-"
            f"CONDITION for any further discriminator-discovery iteration."
        )

    out = {
        "object": OBJECT,
        "session_tag": SESSION_TAG,
        "window_s": WINDOW_S,
        "n_AUTO_success": len(succ_rows),
        "n_AUTO_fail": len(fail_rows),
        "skipped": skipped,
        "feature_set": TOP5_FEATURES,
        "thresholds": {
            "auc_strong": 0.80,
            "auc_moderate": 0.70,
        },
        "univariate_auc_same_subset": univ,
        "lda_loo_auc": lda_auc,
        "logreg_loo_auc": lr_auc,
        "lda_full_fit": lda_full,
        "logreg_full_fit": lr_full,
        "verdict": verdict,
        "interpretation": interpretation,
        "selection_bias_note": (
            "Top-5 features were chosen on this same dataset in discovery 006, "
            "so the LOO AUC reported here is an upper bound on out-of-sample "
            "performance. A truly held-out test would need a fresh autonomous "
            "batch (needs_robot_data). Adequate for the binary 'does the joint "
            "signal exist?' question."
        ),
    }

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("metrics.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}\n")
    print(f"--- VERDICT: {verdict} ---")
    print(interpretation)


if __name__ == "__main__":
    main()
