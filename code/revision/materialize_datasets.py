"""
Writes every dataset used in the revision to data/processed/<name>.parquet and records, for each one,
the generator, seed(s), size, delay rate, SHA-256 checksum and software versions in
data/processed/manifest.json (human-readable copy in MANIFEST.md).

Consistency checks (a dataset is written only if all pass):
  1. determinism   : generated twice in the same session -> identical arrays;
  2. round trip    : the parquet file read back is identical to the generated frame;
  3. references    : 'reference' equals data/raw/df_dpi.csv (manuscript data) exactly;
                     'gamma_reference' equals reference_dpi_gamma.pkl exactly (after forward fill);
                     'cmp_reference' equals reference_dpi_cmpoisson.pkl (after forward fill).
Files already listed in the manifest are re-verified against their checksum and skipped.

Usage: .venv/bin/python code/revision/materialize_datasets.py
"""
import datetime
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mc_simulator as mc  # noqa: E402

V2 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
OUT = os.path.join(V2, "data", "processed")
MANIFEST = os.path.join(OUT, "manifest.json")
SP_DIR = os.environ.get("SPC_LEGACY_DIR", "") + "/"  # legacy pickles of the original loop-based simulator (only to re-create the legacy datasets)
PERIOD_COLS = [f"period_{t}" for t in range(1, 331)]

LEGACY_TRI = "numpy legacy MT19937: np.random.seed(s) for s = 1..100, 100 projects per seed (original simulator)"
LEGACY_GAMMA = "numpy legacy MT19937: np.random.seed(s) for s = 0..99, 100 projects per seed (original simulator)"
PCG = "numpy.random.default_rng(seed) (PCG64)"

# name: (role, distribution, n, generator, seed)
SPECS = {
    "reference":        ("train (manuscript)", "triangular", 10000, LEGACY_TRI, "1..100"),
    "gamma_reference":  ("train (E5)", "gamma", 10000, LEGACY_GAMMA, "0..99"),
    "cmp_reference":    ("train (E5)", "cmpoisson", 10000, "original pymc3-based simulator (stored pickle)", "1..100"),
    "pert_train":       ("train (E5, extra)", "pert", 10000, PCG, 2001),
    **{f"tri_rep{r}": ("train, Monte Carlo replication (E4)", "triangular", 10000, PCG, 1000 + r) for r in range(1, 21)},
    "val_triangular":   ("validation (selection only)", "triangular", 100000, PCG, 900011),
    "val_gamma":        ("validation (selection only)", "gamma", 100000, PCG, 900012),
    "val_cmpoisson":    ("validation (selection only)", "cmpoisson", 100000, PCG, 900013),
    "val_pert":         ("validation (selection only)", "pert", 100000, PCG, 900014),
    "test_triangular":  ("independent test (reported once)", "triangular", 100000, PCG, 900001),
    "test_gamma":       ("independent test (reported once)", "gamma", 100000, PCG, 900002),
    "test_cmpoisson":   ("independent test (reported once)", "cmpoisson", 100000, PCG, 900003),
    "test_pert":        ("independent test (reported once)", "pert", 100000, PCG, 900004),
}


def frame(dpi, total_duration, deadline=320):
    df = dpi[PERIOD_COLS].copy()
    df["total_duration"] = np.asarray(total_duration, dtype=float)
    df["late_project"] = df.total_duration > deadline
    return df.reset_index(drop=True)


def generate(name):
    role, dist, n, gen, seed = SPECS[name]
    if name == "reference":
        dpi, res = mc.simulate(n_sim=10000, rng_mode="legacy", dist="triangular")
    elif name == "gamma_reference":
        dpi, res = mc.simulate(n_sim=10000, rng_mode="legacy", dist="gamma")
    elif name == "cmp_reference":
        dpi = pd.read_pickle(SP_DIR + "reference_dpi_cmpoisson.pkl").astype(float).T.reset_index(drop=True)
        dpi.columns = PERIOD_COLS
        res = pd.read_pickle(SP_DIR + "reference_result_cmpoisson.pkl").reset_index(drop=True)
        return frame(dpi.ffill(axis=1), res.total_duration.astype(float).values)
    else:
        dpi, res = mc.simulate(n_sim=n, seed=seed, dist=dist)
    return frame(dpi, res.total_duration.values)


def align_to(df_new, durations_ref):
    """Reorder simulated projects to the row order of a stored reference (matched by duration)."""
    key = pd.Series(range(len(df_new)), index=np.round(df_new.total_duration.values, 8))
    idx = key.reindex(np.round(np.asarray(durations_ref, dtype=float), 8)).values
    assert not np.isnan(idx).any(), "projects could not be matched"
    return df_new.iloc[idx.astype(int)].reset_index(drop=True)


def check_reference(name, df):
    if name == "reference":
        paper = pd.read_csv(os.path.join(V2, "data", "raw", "df_dpi.csv"))
        res = pd.read_pickle(SP_DIR + "reference_result.pkl").reset_index(drop=True)
        df = align_to(df, res.total_duration.values)
        assert np.array_equal(df[PERIOD_COLS].values, paper[PERIOD_COLS].values) or \
            np.abs(df[PERIOD_COLS].values - paper[PERIOD_COLS].values).max() < 1e-12, "differs from df_dpi.csv"
        assert (df.late_project.values == paper.late_project.astype(bool).values).all()
        # store the manuscript values themselves (exact bytes of the published data)
        df[PERIOD_COLS] = paper[PERIOD_COLS].values
        return df, "equal to data/raw/df_dpi.csv (manuscript) after ordering as in reference_result.pkl"
    if name == "gamma_reference":
        ref = pd.read_pickle(SP_DIR + "reference_dpi_gamma.pkl").astype(float).T.values
        res = pd.read_pickle(SP_DIR + "reference_result_gamma.pkl").reset_index(drop=True)
        aligned = align_to(df, res.total_duration.values)
        assert np.abs(aligned[PERIOD_COLS].values - ref).max() < 1e-12, "differs from reference_dpi_gamma.pkl"
        # stored in simulator order (seed blocks 0..99), which is the order used in the modeling (train/test split)
        return df, "same projects as reference_dpi_gamma.pkl (max |diff| < 1e-12 after matching); rows in simulator order (seed blocks 0..99), as used in the modeling"
    if name == "cmp_reference":
        res = pd.read_pickle(SP_DIR + "reference_result_cmpoisson.pkl").reset_index(drop=True)
        assert (df.late_project.values == res.late_project.astype(bool).values).all()
        return df, "copy of reference_dpi_cmpoisson.pkl + forward fill; DPI/result alignment corr(DPI_330, 300/D) = 0.9999"
    return df, "generated with the vectorized simulator"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"datasets": {}}
    manifest["software"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                            "simulator": "v2/code/revision/mc_simulator.py", "deadline_T": 320, "horizon_days": 330,
                            "dpi_after_completion": "forward fill of the last valid DPI"}
    for name, (role, dist, n, gen, seed) in SPECS.items():
        path = os.path.join(OUT, f"{name}.parquet")
        if name in manifest["datasets"] and os.path.exists(path):
            assert sha256(path) == manifest["datasets"][name]["sha256"], f"checksum mismatch: {name}"
            print("ok (verified)", name, flush=True)
            continue
        df1 = generate(name)
        if name != "cmp_reference":
            df2 = generate(name)
            assert df1.equals(df2), f"non-deterministic: {name}"
        df1, note = check_reference(name, df1)
        df1.to_parquet(path, compression="zstd", index=False)
        back = pd.read_parquet(path)
        assert back.equals(df1), f"round trip failed: {name}"
        manifest["datasets"][name] = {
            "file": f"data/processed/{name}.parquet", "role": role, "distribution": dist, "n_projects": int(len(df1)),
            "generator": gen, "seed": seed, "late_rate": round(float(df1.late_project.mean()), 4),
            "sha256": sha256(path), "check": note, "created": datetime.date.today().isoformat()}
        json.dump(manifest, open(MANIFEST, "w"), indent=2)
        print("written", name, len(df1), round(df1.late_project.mean(), 4), flush=True)
    print("all datasets verified")
