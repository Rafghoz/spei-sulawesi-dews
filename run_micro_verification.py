import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

METEO_CANDIDATES = [
    "PRECTOTCORR", "GWETTOP", "T2M_MAX", "T2M_MIN",
    "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M",
]


def find_csv(explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    for p in (
        "DATA/PROCESSED/dataset_master_nasa_csic_clean.csv",
        "data/processed/dataset_master_nasa_csic_clean.csv",
        "dataset_master_nasa_csic_clean.csv",
    ):
        if os.path.exists(p):
            return p
    found = glob.glob("**/dataset_master*.csv", recursive=True)
    if found:
        return found[0]
    raise FileNotFoundError("Tidak menemukan dataset_master_nasa_csic_clean.csv")


def build_features(df, date_col, spei_col, meteo_cols, horizon=1):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col).reset_index(drop=True)
    s = d[spei_col].astype(float)
    feats = pd.DataFrame(index=d.index)
    feats["SPEI_t"] = s
    months = d[date_col].dt.month.values
    feats["MONTH_SIN"] = np.sin(2.0 * np.pi * months / 12.0)
    feats["MONTH_COS"] = np.cos(2.0 * np.pi * months / 12.0)
    feats["SPEI_lag1"] = s.shift(1)
    feats["SPEI_lag2"] = s.shift(2)
    for w in (2, 3, 6):
        feats[f"SPEI_delta_{w}"] = s - s.shift(w)
        feats[f"SPEI_mean_{w}"] = s.rolling(w, min_periods=w).mean()
        feats[f"SPEI_ema_{w}"] = s.ewm(span=w, adjust=False).mean()
    for mc in meteo_cols:
        v = d[mc].astype(float)
        feats[f"{mc}_t"] = v
        feats[f"{mc}_lag1"] = v.shift(1)
        for w in (2, 3, 6):
            feats[f"{mc}_delta_{w}"] = v - v.shift(w)
            feats[f"{mc}_mean_{w}"] = v.rolling(w, min_periods=w).mean()
            feats[f"{mc}_ema_{w}"] = v.ewm(span=w, adjust=False).mean()
    target = s.shift(-horizon)
    valid = pd.Series(True, index=d.index)
    valid.iloc[:6] = False
    valid.iloc[-horizon:] = False
    X_all = feats.loc[valid].reset_index(drop=True)
    y_all = target.loc[valid].reset_index(drop=True).values
    dates_all = d.loc[valid, date_col].reset_index(drop=True)
    months_target = (dates_all + pd.DateOffset(months=horizon)).dt.month.values
    spei_t = X_all["SPEI_t"].values.copy()
    return X_all, y_all, months_target, spei_t


def get_splits(n_eff):
    return slice(0, 315), slice(320, 388), slice(393, n_eff)


def select_mir12(X_tr, y_tr, k=12, seed=42):
    mi = mutual_info_regression(X_tr, y_tr, n_neighbors=5, random_state=seed)
    order = np.argsort(mi)[::-1]
    return [X_tr.columns[i] for i in order[:k]]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def dm_hac_test(y_true, pred_a, pred_b, h=1, q=None):
    d = (y_true - pred_a) ** 2 - (y_true - pred_b) ** 2
    n = len(d)
    d_bar = float(np.mean(d))
    dc = d - d_bar
    gamma0 = float(np.dot(dc, dc) / n)
    if q is None:
        q = max(5, h)
    v_hac = gamma0
    for lag in range(1, min(q, n - 1) + 1):
        w = 1.0 - lag / (q + 1.0)
        v_hac += 2.0 * w * float(np.dot(dc[lag:], dc[:-lag]) / n)
    v_hac = max(v_hac, 1e-12)
    se = float(np.sqrt(v_hac / n))
    k_hln = float(np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n))
    dm_stat = (d_bar / se) * k_hln
    p_val = float(2.0 * stats.t.sf(abs(dm_stat), df=n - 1))
    return {"d_bar": round(d_bar, 5), "dm": round(float(dm_stat), 3), "p_raw": round(p_val, 4)}


def holm_adj(pvals):
    m = len(pvals)
    idx = np.argsort(pvals)
    adj = np.empty(m)
    rmax = 0.0
    for rank, i in enumerate(idx):
        rmax = max(rmax, (m - rank) * pvals[i])
        adj[i] = min(1.0, rmax)
    return [round(float(x), 4) for x in adj]


def climatology_predict(y_tr, m_tr, m_eval):
    mm = {m: float(np.mean(y_tr[m_tr == m])) if np.any(m_tr == m) else float(np.mean(y_tr)) for m in range(1, 13)}
    return np.array([mm[m] for m in m_eval])


def empirical_prob(y_val, p_val, p_eval, tau=-1.0):
    r_val = y_val - p_val
    n_val = len(r_val)
    return np.array([(1.0 + np.sum(r_val <= (tau - yp))) / (n_val + 1.0) for yp in p_eval])


def cost_metrics(y_bin, prob, thr, cl):
    alert = (prob >= thr).astype(int)
    n = len(y_bin)
    tp = int(np.sum((alert == 1) & (y_bin == 1)))
    fp = int(np.sum((alert == 1) & (y_bin == 0)))
    fn = int(np.sum((alert == 0) & (y_bin == 1)))
    s = float(np.mean(y_bin))
    e_model = (tp * cl + fp * cl + fn * 1.0) / n
    e_clim = min(cl, s)
    e_perf = s * cl
    denom = e_clim - e_perf
    rev = float((e_clim - e_model) / denom) if denom > 1e-12 else np.nan
    return e_model, e_clim, rev


def stationary_boot_idx(n, avg_block=6.0, rng=None):
    p_stop = 1.0 / avg_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        idx[t] = rng.integers(0, n) if rng.random() < p_stop else (idx[t - 1] + 1) % n
    return idx


def fit_eval_models(Xtr, ytr, str_, Xva, sva, Xte, ste, seed=42):
    specs = [
        ("Ridge_Direct", Ridge(alpha=1.0), False),
        ("MLP_Direct",   make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=0.1, max_iter=600, random_state=seed)), False),
        ("MLP_Delta",    make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=600, random_state=seed)), True),
        ("RF_Direct",    RandomForestRegressor(n_estimators=150, max_depth=3, random_state=seed, n_jobs=-1), False),
        ("RF_Delta",     RandomForestRegressor(n_estimators=150, max_depth=3, random_state=seed, n_jobs=-1), True),
    ]
    if HAS_XGB:
        specs += [
            ("XGB_Direct", xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.04, random_state=seed, verbosity=0), False),
            ("XGB_Delta",  xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.03, random_state=seed, verbosity=0), True),
        ]
    if HAS_CAT:
        specs += [
            ("Cat_Direct", CatBoostRegressor(iterations=120, depth=4, learning_rate=0.04, random_seed=seed, verbose=0), False),
            ("Cat_Delta",  CatBoostRegressor(iterations=120, depth=4, learning_rate=0.03, random_seed=seed, verbose=0), True),
        ]
    pva, pte = {"B1_Persistence": sva.copy()}, {"B1_Persistence": ste.copy()}
    for name, est, isd in specs:
        est.fit(Xtr, (ytr - str_) if isd else ytr)
        pv = est.predict(Xva)
        pt = est.predict(Xte)
        pva[name] = (sva + pv) if isd else pv
        pte[name] = (ste + pt) if isd else pt
    return pva, pte


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    csv_path = find_csv(args.csv)
    df = pd.read_csv(csv_path)
    date_col = next((c for c in df.columns if c.upper() in ("DATE", "TANGGAL", "TIME")), df.columns[0])
    spei_col = next(c for c in df.columns if c.upper() in ("SPEI_06", "SPEI06"))
    meteo_cols = [c for c in METEO_CANDIDATES if c in df.columns]

    out = {"T1_delta_cost_vs_clim_and_bestbase": {}, "T2a_rf_seeds_and_mean_seed_dm": {}, "T2b_t6_all_models_dm_vs_B0": {}}

    # =========================================================================
    # T1: Delta Cost vs E_clim & vs min(E_clim, E_persist) pada t+1, t+2, t+3 (C/L=0.20)
    # =========================================================================
    B = 2000
    cl = 0.20
    for h in (1, 2, 3):
        Xh, yh, mth, sth = build_features(df, date_col, spei_col, meteo_cols, horizon=h)
        trh, vah, teh = get_splits(len(yh))
        mirh = select_mir12(Xh.iloc[trh], yh[trh], k=12, seed=42)
        pvah, pteh = fit_eval_models(
            Xh.iloc[trh][mirh].values, yh[trh], sth[trh],
            Xh.iloc[vah][mirh].values, sth[vah],
            Xh.iloc[teh][mirh].values, sth[teh],
            seed=42,
        )
        yb_va = (yh[vah] <= -1.0).astype(int)
        yb_te = (yh[teh] <= -1.0).astype(int)
        n_te = len(yb_te)
        rng = np.random.default_rng(42 + h)
        bidxs = [stationary_boot_idx(n_te, 6.0, rng) for _ in range(B)]

        prob_va = {m: empirical_prob(yh[vah], pvah[m], pvah[m], -1.0) for m in pteh}
        prob_te = {m: empirical_prob(yh[vah], pvah[m], pteh[m], -1.0) for m in pteh}

        pstar = {}
        for m in pteh:
            best_r, bp = -1e9, cl
            for thr in np.linspace(0.05, 0.90, 86):
                _, _, rv = cost_metrics(yb_va, prob_va[m], thr, cl)
                if rv > best_r:
                    best_r, bp = rv, float(thr)
            pstar[m] = round(bp, 2)

        ep_full, ec_full, _ = cost_metrics(yb_te, prob_te["B1_Persistence"], pstar["B1_Persistence"], cl)
        best_ref_name = "B1_Persistence" if ep_full < ec_full else "B0_Climatology"

        h_out = {
            "n_test": int(n_te),
            "n_drought": int(np.sum(yb_te)),
            "E_clim": round(ec_full, 4),
            "E_persist": round(ep_full, 4),
            "best_ref": best_ref_name,
            "models": {},
        }
        for m in [k for k in ("Ridge_Direct", "RF_Delta", "Cat_Direct", "XGB_Delta", "Cat_Delta") if k in pteh]:
            em_full, _, rev_full = cost_metrics(yb_te, prob_te[m], pstar[m], cl)
            dc_clim_b, dc_pers_b, dc_best_b = [], [], []
            n_zero_denom = 0
            for bi in bidxs:
                yb_b = yb_te[bi]
                em_b, ec_b, rv_b = cost_metrics(yb_b, prob_te[m][bi], pstar[m], cl)
                ep_b, _, _ = cost_metrics(yb_b, prob_te["B1_Persistence"][bi], pstar["B1_Persistence"], cl)
                if np.isnan(rv_b):
                    n_zero_denom += 1
                dc_clim_b.append(em_b - ec_b)
                dc_pers_b.append(em_b - ep_b)
                dc_best_b.append(em_b - (ep_b if best_ref_name == "B1_Persistence" else ec_b))
            h_out["models"][m] = {
                "P_star_val": pstar[m],
                "E_model": round(em_full, 4),
                "REV_desc": round(rev_full, 3),
                "dCost_vs_Clim": round(em_full - ec_full, 4),
                "dCost_vs_Clim_CI95": [round(float(np.percentile(dc_clim_b, 2.5)), 4), round(float(np.percentile(dc_clim_b, 97.5)), 4)],
                "dCost_vs_Persist": round(em_full - ep_full, 4),
                "dCost_vs_Persist_CI95": [round(float(np.percentile(dc_pers_b, 2.5)), 4), round(float(np.percentile(dc_pers_b, 97.5)), 4)],
                "dCost_vs_BestRef": round(em_full - min(ec_full, ep_full), 4),
                "dCost_vs_BestRef_CI95": [round(float(np.percentile(dc_best_b, 2.5)), 4), round(float(np.percentile(dc_best_b, 97.5)), 4)],
                "boot_resamples_zero_drought": int(n_zero_denom),
            }
        out["T1_delta_cost_vs_clim_and_bestbase"][f"t+{h}"] = h_out

    # =========================================================================
    # T2a: Per-seed RF_Direct RMSE + DM-HAC on 10-seed average predictions (t+1)
    # =========================================================================
    X1, y1, m1, s1 = build_features(df, date_col, spei_col, meteo_cols, horizon=1)
    tr1, va1, te1 = get_splits(len(y1))
    mir1 = select_mir12(X1.iloc[tr1], y1[tr1], k=12, seed=42)
    Xtr1, Xva1, Xte1 = X1.iloc[tr1][mir1].values, X1.iloc[va1][mir1].values, X1.iloc[te1][mir1].values
    y_te1 = y1[te1]

    seeds = [0, 1, 7, 13, 21, 42, 77, 100, 123, 2026]
    rf_dir_per_seed = {}
    pred_stack = {m: [] for m in ("MLP_Direct", "MLP_Delta", "RF_Direct", "RF_Delta", "XGB_Direct", "XGB_Delta", "Cat_Direct", "Cat_Delta")}
    pte_base = None
    for sd in seeds:
        _, pte_s = fit_eval_models(Xtr1, y1[tr1], s1[tr1], Xva1, s1[va1], Xte1, s1[te1], seed=sd)
        if sd == 42:
            pte_base = pte_s
        rf_dir_per_seed[str(sd)] = round(rmse(y_te1, pte_s["RF_Direct"]), 6)
        for m in pred_stack:
            if m in pte_s:
                pred_stack[m].append(pte_s[m])

    mean_preds = {m: np.mean(v, axis=0) for m, v in pred_stack.items() if v}
    # Paired DM on 10-seed average predictions
    fams = [f for f in ("MLP", "RF", "XGB", "Cat") if f"{f}_Direct" in mean_preds]
    p_raw_pair = []
    pair_res = {}
    for f in fams:
        r = dm_hac_test(y_te1, mean_preds[f"{f}_Delta"], mean_preds[f"{f}_Direct"], h=1)
        pair_res[f] = {
            "meanpred_direct_rmse": round(rmse(y_te1, mean_preds[f"{f}_Direct"]), 4),
            "meanpred_delta_rmse": round(rmse(y_te1, mean_preds[f"{f}_Delta"]), 4),
            "dm_del_vs_dir": r["dm"], "p_raw": r["p_raw"],
            "dm_del_vs_rdir": dm_hac_test(y_te1, mean_preds[f"{f}_Delta"], pte_base["Ridge_Direct"], h=1),
            "dm_del_vs_pers": dm_hac_test(y_te1, mean_preds[f"{f}_Delta"], pte_base["B1_Persistence"], h=1),
        }
        p_raw_pair.append(r["p_raw"])
    # Include Ridge (deterministic) in m=5 Holm
    p_ridge_pair = 0.1874
    holm5 = holm_adj([p_ridge_pair] + p_raw_pair)
    for f, ph in zip(fams, holm5[1:]):
        pair_res[f]["p_holm5"] = ph

    out["T2a_rf_seeds_and_mean_seed_dm"] = {
        "rf_direct_per_seed_rmse": rf_dir_per_seed,
        "rf_direct_seed42_exact": rf_dir_per_seed["42"],
        "rf_direct_10seed_mean_exact": round(float(np.mean(list(rf_dir_per_seed.values()))), 6),
        "mean_seed_pred_dm": pair_res,
    }

    # =========================================================================
    # T2b: DM-HAC seluruh model utama pada t+6 melawan B0_Climatology + Holm
    # =========================================================================
    X6, y6, m6, s6 = build_features(df, date_col, spei_col, meteo_cols, horizon=6)
    tr6, va6, te6 = get_splits(len(y6))
    mir6 = select_mir12(X6.iloc[tr6], y6[tr6], k=12, seed=42)
    _, pte6 = fit_eval_models(
        X6.iloc[tr6][mir6].values, y6[tr6], s6[tr6],
        X6.iloc[va6][mir6].values, s6[va6],
        X6.iloc[te6][mir6].values, s6[te6],
        seed=42,
    )
    p_b0_6 = climatology_predict(y6[tr6], m6[tr6], m6[te6])
    y_te6 = y6[te6]
    r_b0_6 = rmse(y_te6, p_b0_6)
    test_m6 = [m for m in ("Ridge_Direct", "RF_Delta", "Cat_Direct", "Cat_Delta", "XGB_Delta") if m in pte6]
    p6_raw = []
    m6_res = {}
    for m in test_m6:
        rm = rmse(y_te6, pte6[m])
        dm6 = dm_hac_test(y_te6, pte6[m], p_b0_6, h=6)
        p6_raw.append(dm6["p_raw"])
        m6_res[m] = {
            "rmse": round(rm, 4),
            "ss_vs_B0_pct": round((1.0 - (rm / r_b0_6) ** 2) * 100.0, 2),
            "dm_vs_B0": dm6["dm"],
            "p_raw_vs_B0": dm6["p_raw"],
        }
    for m, ph in zip(test_m6, holm_adj(p6_raw)):
        m6_res[m]["p_holm_t6"] = ph
    out["T2b_t6_all_models_dm_vs_B0"] = {"B0_Climatology_rmse": round(r_b0_6, 4), "models": m6_res}
    out["runtime_sec"] = round(time.perf_counter() - t0, 2)

    with open("micro_results.json", "w", encoding="utf-8") as fj:
        json.dump(out, fj, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[SELESAI dalam {out['runtime_sec']} detik] Tersimpan di `micro_results.json`.")


if __name__ == "__main__":
    main()