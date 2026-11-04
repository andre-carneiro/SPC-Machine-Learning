"""
Vectorized Monte Carlo simulator of the EPC project (E4/E5/E9).

Faithful re-implementation of statistical_project_control/scripts/1_monte_carlo_simulator
(functions_body.py + functions_indexes.py, used to create reference_dpi.pkl), including:
  * the order of random draws per simulation: one np.random.triangular vector (durations), then one
    np.random.uniform vector (costs) -> with the same seed the draws are identical to the original;
  * the "break" in the TPD/TED loops: activities are scanned in spreadsheet order and the scan stops
    at the first activity whose start is later than the current day;
  * the ED interpolation rule (first integer day whose TPD exceeds TED; ED = TED/TPD_1 before day 1).
Only the DPI and the outcome are computed (CPI/SPI/TPI are not needed for the paper).

Supported duration distributions: triangular (original), pert (Beta-PERT with the same min/mode/max),
uniform (same min/max), gamma (shape/scale from db_original_gamma.xlsx).
"""
import os
import numpy as np
import pandas as pd

# project parameter files (network, durations, costs): copied to data/project from the original simulator repository
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "data", "project")
RAW_XLSX = os.path.join(PROJECT_DIR, "db_original.xlsx")
RAW_XLSX_GAMMA = os.path.join(PROJECT_DIR, "db_original_gamma.xlsx")
RAW_XLSX_CMP = os.path.join(PROJECT_DIR, "db_original_cmpoisson.xlsx")


def load_network(path=RAW_XLSX, sheet_name="reference"):
    """Read the activity table exactly as functions_body.import_df/adjust_df do."""
    df = pd.read_excel(path, sheet_name, header=1, dtype="str").drop("Unnamed: 0", axis=1)
    cols = ["activity", "predecessor", "min_time", "avr_time", "max_time", "expectation", "std_dev",
            "inflection", "intercept", "parameter", "avr_cost"]
    extra = list(df.columns[len(cols):])
    df.columns = cols + extra
    for c in df.columns:
        if c != "predecessor":
            df[c] = df[c].astype(float)
    df["predecessor"] = df.predecessor.astype(str).apply(lambda x: x.split(", "))
    return df.reset_index(drop=True)


def _pred_index(df):
    act_pos = {int(a): i for i, a in enumerate(df.activity)}
    preds = []
    for p in df.predecessor:
        preds.append([] if p[0] in ("nan", "0") else [act_pos[int(x)] for x in p])
    return preds


def _forward_pass(durations, preds):
    """durations: (n_sim, n_act). Start = max end of predecessors (0 if none)."""
    n_sim, n_act = durations.shape
    start = np.zeros_like(durations)
    end = np.zeros_like(durations)
    for j in range(n_act):
        if preds[j]:
            start[:, j] = end[:, preds[j]].max(axis=1)
        end[:, j] = start[:, j] + durations[:, j]
    return start, end


def _earned_curve(start, end, weight, range_time):
    """
    Cumulative earned (or planned) duration for days 0..range_time, with the original 'break' rule.
    start/end: (n_sim, n_act); weight: (n_act,) planned durations. Returns (n_sim, range_time+1).
    """
    days = np.arange(range_time + 1)[None, :, None]                      # (1, T+1, 1)
    s, e = start[:, None, :], end[:, None, :]                             # (n, 1, a)
    started = s <= days                                                   # (n, T+1, a)
    active = np.cumprod(started, axis=2).astype(bool)                     # break at first not-started activity
    frac = np.where(e <= days, 1.0, (days - s) / (e - s))
    contrib = np.where(active, frac * weight[None, None, :], 0.0)
    curve = contrib.sum(axis=2)
    curve[:, 0] = 0.0                                                     # day 0 is never computed in the original
    return curve


def _ed_dpi(ted, tpd, range_time):
    """ED and DPI for days 1..range_time (original interpolation rule)."""
    n_sim = ted.shape[0]
    tpd_1 = tpd[1:range_time + 1]                                         # TPD at days 1..T
    t = ted[:, 1:range_time + 1]                                          # TED at days 1..T, (n, T)
    greater = tpd_1[None, None, :] > t[:, :, None]                        # (n, T, T)
    found = greater.any(axis=2)
    time_aux = np.where(found, greater.argmax(axis=2) + 1, range_time)   # first day with TPD > TED
    tpd_plus = tpd[time_aux]
    tpd_prev = tpd[time_aux - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ed = np.where(time_aux == 1, t / tpd_plus, time_aux - 1 + (t - tpd_prev) / (tpd_plus - tpd_prev))
    ed = np.where(~found & (tpd_plus == tpd_prev), np.nan, ed)            # original raises -> NaN
    dpi = ed / np.arange(1, range_time + 1)[None, :]
    return ed, dpi


def _gamma_params():
    g = load_network(RAW_XLSX_GAMMA)
    return g.iloc[:, -2].values.astype(float), g.iloc[:, -1].values.astype(float)


def _cmp_draw(rng, size, k_max=400):
    """
    Conway-Maxwell-Poisson durations, replicating the original sampler (CMPoisson._random in
    functions_body.py): inverse-CDF walk starting at p0 = 1/Z with Z given by the asymptotic
    approximation Z = exp(nu*alpha) / ((2*pi*alpha)^((nu-1)/2) * sqrt(nu)), alpha = lambda^(1/nu).
    Computed in log space to avoid overflow; the RNG stream differs from the original (pymc3).
    """
    g = load_network(RAW_XLSX_CMP)
    lam, nu = g.iloc[:, -2].values.astype(float), g.iloc[:, -1].values.astype(float)
    alpha = lam ** (1 / nu)
    log_z = nu * alpha - (nu - 1) / 2 * np.log(2 * np.pi * alpha) - 0.5 * np.log(nu)
    k = np.arange(k_max + 1)[:, None]
    from scipy.special import gammaln
    log_p = k * np.log(lam)[None, :] - nu[None, :] * gammaln(k + 1) - log_z[None, :]   # (k, a)
    cdf = np.cumsum(np.exp(log_p), axis=0)
    u = rng.uniform(size=size)
    out = np.empty(size)
    for j in range(size[1]):
        out[:, j] = np.minimum(np.searchsorted(cdf[:, j], u[:, j], side="left"), k_max)
    return out


def draw_durations(df, n_sim, rng_mode="legacy", dist="triangular", seed=123):
    """
    Draw activity durations for n_sim projects. rng_mode='legacy' replays the original global-RNG call
    sequence (triangular vector, then uniform cost vector, per simulation) so seed 123 reproduces
    reference_dpi.pkl. rng_mode='fast' uses a Generator (independent replications).
    """
    a, c, b = df.min_time.values, df.avr_time.values, df.max_time.values
    n_act = len(df)
    if rng_mode == "legacy":
        # reference_dpi*.pkl = union of 100 runs: np.random.seed(s) for s = 1..100, 100 projects each
        # (monte_carlo_simulator_iterator*.py + monte_carlo_simulator_unificator.ipynb)
        if dist == "triangular":
            draw = lambda: np.random.triangular(a, c, b)
        elif dist == "gamma":
            shape, scale = _gamma_params()
            draw = lambda: np.random.gamma(shape, scale)
        else:
            raise ValueError("legacy mode reproduces only the triangular and gamma runs")
        first_seed = 0 if dist == "gamma" else 1   # gamma union used seeds 0..99, triangular 1..100
        out = np.empty((n_sim, n_act))
        for i in range(n_sim):
            if i % 100 == 0:
                np.random.seed(i // 100 + first_seed)
            out[i] = draw()
            np.random.uniform(df.intercept.values * 0.8, df.intercept.values * 1.2)  # cost draw (keeps RNG aligned)
        return out
    rng = np.random.default_rng(seed)
    size = (n_sim, n_act)
    if dist == "triangular":
        return rng.triangular(a, c, b, size=size)
    if dist == "uniform":
        return rng.uniform(a, b, size=size)
    if dist == "pert":
        alpha = 1 + 4 * (c - a) / (b - a)
        beta = 1 + 4 * (b - c) / (b - a)
        return a + (b - a) * rng.beta(alpha, beta, size=size)
    if dist == "gamma":
        shape, scale = _gamma_params()
        return rng.gamma(shape, scale, size=size)
    if dist == "cmpoisson":
        return _cmp_draw(rng, size)
    raise ValueError(dist)


def simulate(n_sim=10000, seed=1, dist="triangular", rng_mode="fast", deadline=320, range_time=330,
             batch=500, df=None):
    """
    Returns (dpi, result): dpi is a DataFrame (n_sim x range_time) with columns period_1..period_T,
    result has total_duration and late_project, in the same format as df_dpi.csv.
    """
    df = load_network() if df is None else df
    preds = _pred_index(df)
    expectation = df.expectation.values
    plan_start, plan_end = _forward_pass(expectation[None, :], preds)
    tpd = _earned_curve(plan_start, plan_end, expectation, range_time)[0]

    durations = draw_durations(df, n_sim, rng_mode=rng_mode, dist=dist, seed=seed)
    dpis, totals = [], []
    for k in range(0, n_sim, batch):
        dur = durations[k:k + batch]
        start, end = _forward_pass(dur, preds)
        ted = _earned_curve(start, end, expectation, range_time)
        _, dpi = _ed_dpi(ted, tpd, range_time)
        dpis.append(dpi)
        totals.append(end.max(axis=1))
    dpi = pd.DataFrame(np.vstack(dpis), columns=[f"period_{t}" for t in range(1, range_time + 1)])
    # after completion TED cannot be matched by TPD (ED undefined -> NaN); the dataset used in the paper
    # carries the last valid DPI forward in time
    dpi = dpi.ffill(axis=1)
    total = np.concatenate(totals)
    result = pd.DataFrame({"total_duration": total, "late_project": total > deadline})
    return dpi, result
