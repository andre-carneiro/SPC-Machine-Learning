"""
E3 - Inference for AUROC differences (Reviewer 2, comment 4; Reviewer 1, comment 5).

* delong_paired: DeLong et al. (1988) test for two correlated ROC curves (same test set), using the
  fast O(n log n) algorithm of Sun & Xu (2014). Returns AUCs, difference, SE, two-sided p-value,
  95% CI and the one-sided non-inferiority p-value for margin delta (H0: AUC1 - AUC2 <= -delta).
* bootstrap_auc_diff: stratified bootstrap CI for AUC1 - AUC2.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, n) predictions with the m positives first."""
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, cov


def delong_paired(y_true, p1, p2, delta=0.0, alpha=0.05):
    y = np.asarray(y_true).ravel().astype(int)
    order = np.argsort(-y, kind="mergesort")      # positives first
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0)))
    z = diff / se if se > 0 else 0.0
    zc = stats.norm.ppf(1 - alpha / 2)
    p_two = float(2 * stats.norm.sf(abs(z))) if se > 0 else 1.0
    p_noninf = float(stats.norm.sf((diff + delta) / se)) if se > 0 else 0.0
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diff": float(diff), "se": se,
            "ci_low": float(diff - zc * se), "ci_high": float(diff + zc * se),
            "p_two_sided": p_two, "delta": delta, "p_noninferiority": p_noninf,
            "non_inferior": bool(diff - stats.norm.ppf(1 - alpha) * se > -delta)}


def bootstrap_auc_diff(y_true, p1, p2, n_boot=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).ravel().astype(int)
    p1, p2 = np.asarray(p1), np.asarray(p2)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    a1, a2, d = [], [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        x1, x2 = roc_auc_score(y[idx], p1[idx]), roc_auc_score(y[idx], p2[idx])
        a1.append(x1); a2.append(x2); d.append(x1 - x2)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"auc_1_ci": np.percentile(a1, q).tolist(), "auc_2_ci": np.percentile(a2, q).tolist(),
            "diff_ci": np.percentile(d, q).tolist()}
