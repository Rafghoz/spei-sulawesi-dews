import argparse
import glob
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import statsmodels
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import acf, pacf
    HAS_SM = True
except Exception:
    HAS_SM = False

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

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


METEO_CANDIDATES = [
    "PRECTOTCORR", "GWETTOP", "T2M_MAX", "T2M_MIN",
    "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M",
]

GRID_SPEI06_HINTS = {
    "G1_Palu":     ["SPEI_06", "SPEI06", "SPEI_06_G1", "G1_SPEI_06", "SPEI06_G1", "SPEI_06_PALU"],
    "G2_Parigi":   ["SPEI_06_G2", "G2_SPEI_06", "SPEI06_G2", "SPEI_06_PARIGI", "G2_PARIGI_SPEI_06"],
    "G3_Poso":     ["SPEI_06_G3", "G3_SPEI_06", "SPEI06_G3", "SPEI_06_POSO", "G3_POSO_SPEI_06"],
    "G4_Toli":     ["SPEI_06_G4", "G4_SPEI_06", "SPEI06_G4", "SPEI_06_TOLI", "SPEI_06_TOLITOLI"],
    "G5_Makassar": ["SPEI_06_G5", "G5_SPEI_06", "SPEI06_G5", "SPEI_06_MAKASSAR"],
}


def find_master_csv(explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
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
    raise FileNotFoundError("Dataset `dataset_master_nasa_csic_clean.csv` tidak ditemukan.")


def detect_date_col(df):
    for c in df.columns:
        if c.upper() in ("DATE", "TANGGAL", "TIME", "MONTH", "TIMESTAMP"):
            return c
    return df.columns[0]


def detect_spei_col(df, hints):
    cols_up = {c.upper(): c for c in df.columns}
    for h in hints:
        if h.upper() in cols_up:
            return cols_up[h.upper()]
    return None


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

    lag_only_cols = list(feats.columns)

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
    return X_all, y_all, dates_all, months_target, spei_t, lag_only_cols


def get_splits(n_eff, gap=5):
    if gap == 5:
        return slice(0, 315), slice(320, 388), slice(393, n_eff)
    # Tanpa embargo (gap=0): jumlah sampel uji sama persis (berakhir di n_eff)
    n_te = n_eff - 393
    te_start = n_eff - n_te
    va_start = te_start - 68
    return slice(0, va_start), slice(va_start, te_start), slice(te_start, n_eff)


def select_mir12(X_train, y_train, k=12, seed=42):
    mi = mutual_info_regression(X_train, y_train, n_neighbors=5, random_state=seed)
    order = np.argsort(mi)[::-1]
    cols = [X_train.columns[i] for i in order[:k]]
    return cols, {X_train.columns[i]: round(float(mi[i]), 4) for i in order[:k]}


# =============================================================================
# METRIK STATISTIK TERPADU
# =============================================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def dm_hac_test(y_true, pred_a, pred_b, h=1, q=None):
    e_a = y_true - pred_a
    e_b = y_true - pred_b
    d = e_a**2 - e_b**2
    n = len(d)
    d_bar = float(np.mean(d))
    dc = d - d_bar
    gamma0 = float(np.dot(dc, dc) / n)
    if q is None:
        q = max(5, h)
    v_hac = gamma0
    for lag in range(1, min(q, n - 1) + 1):
        w = 1.0 - lag / (q + 1.0)
        g_lag = float(np.dot(dc[lag:], dc[:-lag]) / n)
        v_hac += 2.0 * w * g_lag
    v_hac = max(v_hac, 1e-12)
    n_eff_dm = float(n * (gamma0 / v_hac)) if v_hac > 0 else float(n)
    se = float(np.sqrt(v_hac / n))
    k_hln = float(np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n))
    dm_stat = (d_bar / se) * k_hln
    p_val = float(2.0 * stats.t.sf(abs(dm_stat), df=n - 1))
    se_adj = se / k_hln
    ci_lo = float(d_bar - stats.t.ppf(0.975, df=n - 1) * se_adj)
    ci_hi = float(d_bar + stats.t.ppf(0.975, df=n - 1) * se_adj)
    return {
        "d_bar": round(d_bar, 5), "se": round(se, 5), "dm": round(float(dm_stat), 3),
        "p_raw": round(p_val, 4), "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "n_eff_dm": round(n_eff_dm, 2), "q": int(q),
    }


def tost_mse(y_true, pred_a, pred_b, margin=0.10, h=1, q=None):
    e_a = y_true - pred_a
    e_b = y_true - pred_b
    d = e_a**2 - e_b**2
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
    k_hln = float(np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n))
    se_adj = float(np.sqrt(v_hac / n)) / k_hln
    t_lo = (d_bar + margin) / se_adj
    t_hi = (margin - d_bar) / se_adj
    return round(float(max(stats.t.sf(t_lo, df=n - 1), stats.t.sf(t_hi, df=n - 1))), 4)


def holm_adjust(pvals):
    m = len(pvals)
    idx = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    rmax = 0.0
    for rank, i in enumerate(idx):
        rmax = max(rmax, (m - rank) * pvals[i])
        adj[i] = min(1.0, rmax)
    return [round(float(x), 4) for x in adj]


def christoffersen_cc(hits_bool, c0):
    hits = np.asarray(hits_bool, dtype=int)
    n = len(hits)
    n1 = int(np.sum(hits))
    n0 = n - n1
    c_hat = n1 / n
    eps = 1e-12
    c_hat_c = np.clip(c_hat, eps, 1.0 - eps)
    lr_uc = -2.0 * (
        n0 * np.log((1.0 - c0) / (1.0 - c_hat_c))
        + n1 * np.log(c0 / c_hat_c)
    )
    # Transisi orde-1: n00, n01, n10, n11
    n00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    n01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    n10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    n11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    def safe_term(k, p):
        return k * np.log(np.clip(p, eps, 1.0 - eps)) if k > 0 else 0.0
    ll_null = safe_term(n00 + n10, 1.0 - pi) + safe_term(n01 + n11, pi)
    ll_alt = (safe_term(n00, 1.0 - pi01) + safe_term(n01, pi01)
              + safe_term(n10, 1.0 - pi11) + safe_term(n11, pi11))
    lr_ind = max(0.0, -2.0 * (ll_null - ll_alt))
    lr_cc = max(0.0, lr_uc + lr_ind)
    p_cc = float(stats.chi2.sf(lr_cc, df=2))
    return {
        "picp_pct": round(100.0 * c_hat, 1),
        "n_hit": n1,
        "lr_uc": round(float(lr_uc), 3),
        "lr_ind": round(float(lr_ind), 3),
        "lr_cc": round(float(lr_cc), 3),
        "p_cc": round(p_cc, 4),
    }


def winkler_score(y_true, lo, hi, alpha):
    width = hi - lo
    pen_lo = (2.0 / alpha) * np.maximum(0.0, lo - y_true)
    pen_hi = (2.0 / alpha) * np.maximum(0.0, y_true - hi)
    return round(float(np.mean(width + pen_lo + pen_hi)), 4)


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n)) / denom
    return [round(100.0 * max(0.0, center - half), 1), round(100.0 * min(1.0, center + half), 1)]


def empirical_prob(y_val, p_val, p_eval, tau=-1.0):
    r_val = y_val - p_val
    n_val = len(r_val)
    return np.array([(1.0 + np.sum(r_val <= (tau - yp))) / (n_val + 1.0) for yp in p_eval])


def rev_and_cost(y_bin, prob, thr, cl):
    alert = (prob >= thr).astype(int)
    n = len(y_bin)
    tp = int(np.sum((alert == 1) & (y_bin == 1)))
    fp = int(np.sum((alert == 1) & (y_bin == 0)))
    fn = int(np.sum((alert == 0) & (y_bin == 1)))
    tn = int(np.sum((alert == 0) & (y_bin == 0)))
    s = float(np.mean(y_bin))
    e_model = (tp * cl + fp * cl + fn * 1.0) / n
    e_clim = min(cl, s)
    e_perf = s * cl
    denom = e_clim - e_perf
    rev = float((e_clim - e_model) / denom) if denom > 1e-12 else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "e_model": e_model, "e_clim": e_clim, "rev": rev}


def stationary_boot_idx(n, avg_block=6.0, rng=None):
    p_stop = 1.0 / avg_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        idx[t] = rng.integers(0, n) if rng.random() < p_stop else (idx[t - 1] + 1) % n
    return idx


# =============================================================================
# SATU REGISTRASI MODEL BEKU (DIGUNAKAN KONSISTEN DI SELURUH TABEL!)
# =============================================================================
def climatology_predict(y_tr, m_tr, m_eval):
    mm = {m: float(np.mean(y_tr[m_tr == m])) if np.any(m_tr == m) else float(np.mean(y_tr)) for m in range(1, 13)}
    return np.array([mm[m] for m in m_eval])


def fit_sarima_frozen(series_full, tr_end, va_sl, te_sl, horizon=1):
    """SARIMA(2,0,0)(1,0,0)_12 diestimasi sekali di Train, lalu extend kausal."""
    if not HAS_SM:
        return series_full[va_sl], series_full[te_sl]
    try:
        mod = SARIMAX(series_full[:tr_end], order=(2, 0, 0), seasonal_order=(1, 0, 0, 12),
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        ext = res.extend(series_full[tr_end:])
        if horizon == 1:
            preds = ext.fittedvalues
            p_va = preds[va_sl.start - tr_end: va_sl.stop - tr_end]
            p_te = preds[te_sl.start - tr_end: te_sl.stop - tr_end]
            # Gunakan prediksi dari origin t menuju t+1 (geser 1 sesuai target y)
            return np.asarray(p_va, dtype=float), np.asarray(p_te, dtype=float)
    except Exception:
        pass
    return series_full[va_sl], series_full[te_sl]


def make_14_models_predictions(X_tr, y_tr, s_tr, m_tr, X_va, y_va, s_va, m_va, X_te, y_te, s_te, m_te,
                               full_spei_series=None, tr_sl=None, va_sl=None, te_sl=None, seed=42,
                               ridge_alpha_dir=1.0, ridge_alpha_del=50.0, horizon=1,
                               ar_tr=None, ar_va=None, ar_te=None):
    """Melatih tepat 14 model utama secara seragam dan mengembalikan prediksi Val & Test."""
    preds_va, preds_te = {}, {}

    # B0 Climatology
    preds_va["B0_Climatology"] = climatology_predict(y_tr, m_tr, m_va)
    preds_te["B0_Climatology"] = climatology_predict(y_tr, m_tr, m_te)

    # B1 Persistence
    preds_va["B1_Persistence"] = s_va.copy()
    preds_te["B1_Persistence"] = s_te.copy()

    # B2 AR(2) selalu menggunakan [SPEI_t, SPEI_lag1] penuh (tidak bergantung pada seleksi MIR)
    if ar_tr is not None and ar_va is not None and ar_te is not None:
        m_ar2 = Ridge(alpha=1.0).fit(ar_tr, y_tr)
        preds_va["B2_AR(2)"] = m_ar2.predict(ar_va)
        preds_te["B2_AR(2)"] = m_ar2.predict(ar_te)
    else:
        ar_cols = [c for c in ("SPEI_t", "SPEI_lag1") if c in X_tr.columns]
        if ar_cols:
            m_ar2 = Ridge(alpha=1.0).fit(X_tr[ar_cols].values, y_tr)
            preds_va["B2_AR(2)"] = m_ar2.predict(X_va[ar_cols].values)
            preds_te["B2_AR(2)"] = m_ar2.predict(X_te[ar_cols].values)
        else:
            m_ar2 = Ridge(alpha=1.0).fit(s_tr.reshape(-1, 1), y_tr)
            preds_va["B2_AR(2)"] = m_ar2.predict(s_va.reshape(-1, 1))
            preds_te["B2_AR(2)"] = m_ar2.predict(s_te.reshape(-1, 1))

    # B3 SARIMA(2,0,0)(1,0,0)_12 (diestimasi murni pada Train)
    if HAS_SM and full_spei_series is not None and horizon == 1:
        try:
            # Target pada indeks i adalah s[i + 1]; jadi latih pada s[0..315]
            s_train_raw = full_spei_series[:tr_sl.stop + 1]
            mod = SARIMAX(s_train_raw, order=(2, 0, 0), seasonal_order=(1, 0, 0, 12),
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            ext = res.extend(full_spei_series[tr_sl.stop + 1:])
            all_pred = np.concatenate([res.fittedvalues, ext.fittedvalues])
            # Prediksi untuk target y[i] = full_spei_series[i + 1] ada di all_pred[i + 1]
            preds_va["B3_SARIMA"] = all_pred[va_sl.start + 1: va_sl.stop + 1]
            preds_te["B3_SARIMA"] = all_pred[te_sl.start + 1: te_sl.stop + 1]
        except Exception:
            preds_va["B3_SARIMA"] = preds_va["B2_AR(2)"].copy()
            preds_te["B3_SARIMA"] = preds_te["B2_AR(2)"].copy()
    else:
        preds_va["B3_SARIMA"] = preds_va["B2_AR(2)"].copy()
        preds_te["B3_SARIMA"] = preds_te["B2_AR(2)"].copy()

    Xtr_v, Xva_v, Xte_v = X_tr.values, X_va.values, X_te.values

    ml_specs = [
        ("Ridge_Direct", Ridge(alpha=ridge_alpha_dir), False),
        ("Ridge_Delta",  Ridge(alpha=ridge_alpha_del), True),
        ("MLP_Direct",   make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=0.1, max_iter=600, random_state=seed)), False),
        ("MLP_Delta",    make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=600, random_state=seed)), True),
        ("RF_Direct",    RandomForestRegressor(n_estimators=150, max_depth=3, random_state=seed, n_jobs=-1), False),
        ("RF_Delta",     RandomForestRegressor(n_estimators=150, max_depth=3, random_state=seed, n_jobs=-1), True),
    ]
    if HAS_XGB:
        ml_specs += [
            ("XGB_Direct", xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.04, random_state=seed, verbosity=0), False),
            ("XGB_Delta",  xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.03, random_state=seed, verbosity=0), True),
        ]
    if HAS_CAT:
        ml_specs += [
            ("Cat_Direct", CatBoostRegressor(iterations=120, depth=4, learning_rate=0.04, random_seed=seed, verbose=0), False),
            ("Cat_Delta",  CatBoostRegressor(iterations=120, depth=4, learning_rate=0.03, random_seed=seed, verbose=0), True),
        ]

    for name, est, is_del in ml_specs:
        y_fit = (y_tr - s_tr) if is_del else y_tr
        est.fit(Xtr_v, y_fit)
        pva = est.predict(Xva_v)
        pte = est.predict(Xte_v)
        preds_va[name] = (s_va + pva) if is_del else pva
        preds_te[name] = (s_te + pte) if is_del else pte

    return preds_va, preds_te


def main():
    parser = argparse.ArgumentParser(description="Pipa Terpadu Tunggal Revisi Techno.Com")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--masks", type=int, default=100)
    parser.add_argument("--bench_reps", type=int, default=10)
    args = parser.parse_args()

    t_global = time.perf_counter()
    csv_path = find_master_csv(args.csv)
    df = pd.read_csv(csv_path)
    date_col = detect_date_col(df)
    spei_col = detect_spei_col(df, GRID_SPEI06_HINTS["G1_Palu"])
    meteo_cols = [c for c in METEO_CANDIDATES if c in df.columns]
    print(f"[INFO] Dataset={csv_path} ({len(df)} rows) | SPEI={spei_col} | Meteo={len(meteo_cols)}")

    # =========================================================================
    # A. DATA UTAMA t+1 & SELEKSI FITUR MIR-12 (SINGLE SOURCE OF TRUTH)
    # =========================================================================
    X_all, y_all, dates_all, mt_all, st_all, lag14_cols = build_features(df, date_col, spei_col, meteo_cols, horizon=1)
    tr, va, te = get_splits(len(y_all), gap=5)
    mir12_cols, mir12_scores = select_mir12(X_all.iloc[tr], y_all[tr], k=12, seed=42)
    full_spei_aligned = np.concatenate([st_all, [y_all[-1]]])

    # Statistik partisi (Tabel 2) + ACF/PACF + CI 95%
    table2_stats = {}
    for pname, sl in [("Train", tr), ("Val", va), ("Test", te)]:
        yv = y_all[sl]
        sv = st_all[sl]
        n_p = len(yv)
        a_vals = [round(float(v), 4) for v in acf(sv, nlags=12, fft=False)] if HAS_SM else []
        ci_acf6 = round(float(1.96 / np.sqrt(max(n_p - 6, 1))), 4)
        table2_stats[pname] = {
            "n": int(n_p),
            "mean": round(float(np.mean(yv)), 4),
            "sd": round(float(np.std(yv, ddof=1)), 4),
            "min": round(float(np.min(yv)), 4),
            "max": round(float(np.max(yv)), 4),
            "n_mod": int(np.sum(yv <= -1.0)),
            "rate_mod_pct": round(100.0 * float(np.mean(yv <= -1.0)), 1),
            "n_sev": int(np.sum(yv <= -1.5)),
            "acf_lag1": a_vals[1] if len(a_vals) > 1 else None,
            "acf_lag6": a_vals[6] if len(a_vals) > 6 else None,
            "acf6_wn_ci95_half": ci_acf6,
            "acf_0_to_12": a_vals,
        }

    # =========================================================================
    # B. BENCHMARK RUNTIME NYATA PADA PERANGKAT LOKAL (M11 FIX!)
    # =========================================================================
    bench_times = []
    for _ in range(args.bench_reps):
        t0 = time.perf_counter()
        make_14_models_predictions(
            X_all.iloc[tr][mir12_cols], y_all[tr], st_all[tr], mt_all[tr],
            X_all.iloc[va][mir12_cols], y_all[va], st_all[va], mt_all[va],
            X_all.iloc[te][mir12_cols], y_all[te], st_all[te], mt_all[te],
            full_spei_series=full_spei_aligned, tr_sl=tr, va_sl=va, te_sl=te, seed=42,
        )
        bench_times.append(time.perf_counter() - t0)

    # =========================================================================
    # C. TABEL 3: 14 MODEL UTAMA + DM-HAC HOLM + 10-SEED + TOST
    # =========================================================================
    pva_14, pte_14 = make_14_models_predictions(
        X_all.iloc[tr][mir12_cols], y_all[tr], st_all[tr], mt_all[tr],
        X_all.iloc[va][mir12_cols], y_all[va], st_all[va], mt_all[va],
        X_all.iloc[te][mir12_cols], y_all[te], st_all[te], mt_all[te],
        full_spei_series=full_spei_aligned, tr_sl=tr, va_sl=va, te_sl=te, seed=42,
    )
    y_te = y_all[te]
    y_va = y_all[va]

    model_order = list(pte_14.keys())
    # DM vs Persistence (m=13) & vs Ridge_Direct (m=13)
    p_vs_pers_raw, dm_vs_pers = [], {}
    p_vs_rdir_raw, dm_vs_rdir = [], {}
    m_pers_list = [m for m in model_order if m != "B1_Persistence"]
    m_rdir_list = [m for m in model_order if m != "Ridge_Direct"]

    for m in m_pers_list:
        r = dm_hac_test(y_te, pte_14[m], pte_14["B1_Persistence"], h=1)
        dm_vs_pers[m] = r
        p_vs_pers_raw.append(r["p_raw"])
    p_pers_holm = dict(zip(m_pers_list, holm_adjust(p_vs_pers_raw)))

    for m in m_rdir_list:
        r = dm_hac_test(y_te, pte_14[m], pte_14["Ridge_Direct"], h=1)
        dm_vs_rdir[m] = r
        p_vs_rdir_raw.append(r["p_raw"])
    p_rdir_holm = dict(zip(m_rdir_list, holm_adjust(p_vs_rdir_raw)))

    # 10-seed stability pada model stokastik
    seeds_10 = [0, 1, 7, 13, 21, 42, 77, 100, 123, 2026]
    seed_rmses = {m: [] for m in model_order if m.startswith(("MLP_", "RF_", "XGB_", "Cat_"))}
    for sd in seeds_10:
        _, pte_sd = make_14_models_predictions(
            X_all.iloc[tr][mir12_cols], y_all[tr], st_all[tr], mt_all[tr],
            X_all.iloc[va][mir12_cols], y_all[va], st_all[va], mt_all[va],
            X_all.iloc[te][mir12_cols], y_all[te], st_all[te], mt_all[te],
            full_spei_series=None, seed=sd,
        )
        for m in seed_rmses:
            seed_rmses[m].append(rmse(y_te, pte_sd[m]))

    table3_14models = {}
    for m in model_order:
        rm = round(rmse(y_te, pte_14[m]), 4)
        ss = round((1.0 - (rm / rmse(y_te, pte_14["B1_Persistence"])) ** 2) * 100.0, 2)
        entry = {
            "rmse_seed42": rm,
            "ss_mse_vs_persist_pct": ss,
            "dm_vs_persist": dm_vs_pers[m]["dm"] if m in dm_vs_pers else None,
            "p_raw_vs_persist": dm_vs_pers[m]["p_raw"] if m in dm_vs_pers else None,
            "p_holm_vs_persist": p_pers_holm[m] if m in p_pers_holm else None,
            "dm_vs_rdir": dm_vs_rdir[m]["dm"] if m in dm_vs_rdir else None,
            "p_raw_vs_rdir": dm_vs_rdir[m]["p_raw"] if m in dm_vs_rdir else None,
            "p_holm_vs_rdir": p_rdir_holm[m] if m in p_rdir_holm else None,
            "dbar_vs_rdir": dm_vs_rdir[m]["d_bar"] if m in dm_vs_rdir else 0.0,
            "ci95_vs_rdir": dm_vs_rdir[m]["ci95"] if m in dm_vs_rdir else [0.0, 0.0],
            "tost_005_vs_rdir": tost_mse(y_te, pte_14[m], pte_14["Ridge_Direct"], 0.05) if m != "Ridge_Direct" else 0.0,
            "tost_010_vs_rdir": tost_mse(y_te, pte_14[m], pte_14["Ridge_Direct"], 0.10) if m != "Ridge_Direct" else 0.0,
            "tost_015_vs_rdir": tost_mse(y_te, pte_14[m], pte_14["Ridge_Direct"], 0.15) if m != "Ridge_Direct" else 0.0,
        }
        if m in seed_rmses:
            entry["seed10_mean"] = round(float(np.mean(seed_rmses[m])), 4)
            entry["seed10_sd"] = round(float(np.std(seed_rmses[m], ddof=1)), 4)
        table3_14models[m] = entry

    # Paired Direct vs. Delta (m=5)
    fams_5 = ["Ridge", "MLP", "RF", "XGB", "Cat"]
    p_pair_raw, pair_dm = [], {}
    for fam in fams_5:
        if f"{fam}_Direct" in pte_14 and f"{fam}_Delta" in pte_14:
            r = dm_hac_test(y_te, pte_14[f"{fam}_Delta"], pte_14[f"{fam}_Direct"], h=1)
            pair_dm[fam] = r
            p_pair_raw.append(r["p_raw"])
    p_pair_holm = dict(zip(pair_dm.keys(), holm_adjust(p_pair_raw)))
    table4_paired_main = {
        fam: {
            "direct_rmse": round(rmse(y_te, pte_14[f"{fam}_Direct"]), 4),
            "delta_rmse": round(rmse(y_te, pte_14[f"{fam}_Delta"]), 4),
            "d_bar": pair_dm[fam]["d_bar"],
            "ci95": pair_dm[fam]["ci95"],
            "dm": pair_dm[fam]["dm"],
            "p_raw": pair_dm[fam]["p_raw"],
            "p_holm5": p_pair_holm[fam],
        }
        for fam in pair_dm
    }

    # =========================================================================
    # D. TABEL 4 LANJUTAN: KAPASITAS TINGGI (DIRECT vs DELTA) & RESIDUAL-ON-RIDGE
    #    + ABLASI EMBARGO VS. NO-EMBARGO (M1 / N3!)
    # =========================================================================
    Xtr_v, Xva_v, Xte_v = X_all.iloc[tr][mir12_cols].values, X_all.iloc[va][mir12_cols].values, X_all.iloc[te][mir12_cols].values
    y_tr, s_tr, s_va, s_te = y_all[tr], st_all[tr], st_all[va], st_all[te]

    high_cap_specs = [
        ("Ridge_alpha0.01", Ridge(alpha=0.01), Ridge(alpha=0.01)),
        ("RF_d8_n300", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
                       RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)),
        ("RF_unb_n300", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
                        RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)),
    ]
    if HAS_XGB:
        high_cap_specs.append((
            "XGB_d5_n300",
            xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42, verbosity=0),
            xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42, verbosity=0),
        ))
    if HAS_CAT:
        high_cap_specs.append((
            "Cat_d6_iter400",
            CatBoostRegressor(iterations=400, depth=6, learning_rate=0.03, random_seed=42, verbose=0),
            CatBoostRegressor(iterations=400, depth=6, learning_rate=0.03, random_seed=42, verbose=0),
        ))

    high_cap_out = {}
    high_cap_preds_va, high_cap_preds_te = {}, {}
    for lbl, e_dir, e_del in high_cap_specs:
        e_dir.fit(Xtr_v, y_tr)
        p_d_va, p_d_te = e_dir.predict(Xva_v), e_dir.predict(Xte_v)
        e_del.fit(Xtr_v, y_tr - s_tr)
        p_l_va, p_l_te = s_va + e_del.predict(Xva_v), s_te + e_del.predict(Xte_v)
        high_cap_preds_va[f"{lbl}_Dir"] = p_d_va
        high_cap_preds_te[f"{lbl}_Dir"] = p_d_te
        dm_dl = dm_hac_test(y_te, p_l_te, p_d_te, h=1)
        dm_dr = dm_hac_test(y_te, p_d_te, pte_14["Ridge_Direct"], h=1)
        dm_lr = dm_hac_test(y_te, p_l_te, pte_14["Ridge_Direct"], h=1)
        high_cap_out[lbl] = {
            "direct_rmse": round(rmse(y_te, p_d_te), 4),
            "delta_rmse": round(rmse(y_te, p_l_te), 4),
            "dm_del_vs_dir": dm_dl["dm"], "p_raw_del_vs_dir": dm_dl["p_raw"],
            "dm_dir_vs_rdir": dm_dr["dm"], "p_raw_dir_vs_rdir": dm_dr["p_raw"],
            "dm_del_vs_rdir": dm_lr["dm"], "p_raw_del_vs_rdir": dm_lr["p_raw"],
            "tost_010_dir_vs_rdir": tost_mse(y_te, p_d_te, pte_14["Ridge_Direct"], 0.10),
            "tost_010_del_vs_rdir": tost_mse(y_te, p_l_te, pte_14["Ridge_Direct"], 0.10),
        }

    # Residual-on-Ridge
    r_base = Ridge(alpha=1.0).fit(Xtr_v, y_tr)
    res_tr = y_tr - r_base.predict(Xtr_v)
    r_te_base = r_base.predict(Xte_v)
    res_ridge_out = {}
    rr_specs = [("RF_on_RidgeRes", RandomForestRegressor(n_estimators=150, max_depth=3, random_state=42, n_jobs=-1))]
    if HAS_XGB:
        rr_specs.append(("XGB_on_RidgeRes", xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.03, random_state=42, verbosity=0)))
    if HAS_CAT:
        rr_specs.append(("Cat_on_RidgeRes", CatBoostRegressor(iterations=120, depth=4, learning_rate=0.03, random_seed=42, verbose=0)))
    rr_p_raw = []
    for rname, rest in rr_specs:
        rest.fit(Xtr_v, res_tr)
        p_comb = r_te_base + rest.predict(Xte_v)
        dm_r = dm_hac_test(y_te, p_comb, pte_14["Ridge_Direct"], h=1)
        rr_p_raw.append(dm_r["p_raw"])
        res_ridge_out[rname] = {
            "rmse": round(rmse(y_te, p_comb), 4),
            "dm_vs_rdir": dm_r["dm"], "p_raw_vs_rdir": dm_r["p_raw"],
            "tost_010_vs_rdir": tost_mse(y_te, p_comb, pte_14["Ridge_Direct"], 0.10),
        }
    for k, ph in zip(res_ridge_out.keys(), holm_adjust(rr_p_raw)):
        res_ridge_out[k]["p_holm3_vs_rdir"] = ph

    # Ablasi Fitur MIR-12 vs. Lag-Only (14)
    _, pte_lag14 = make_14_models_predictions(
        X_all.iloc[tr][lag14_cols], y_all[tr], st_all[tr], mt_all[tr],
        X_all.iloc[va][lag14_cols], y_all[va], st_all[va], mt_all[va],
        X_all.iloc[te][lag14_cols], y_all[te], st_all[te], mt_all[te],
        full_spei_series=None, seed=42,
    )
    abl_models = ["Ridge_Direct", "MLP_Delta", "RF_Delta", "XGB_Delta", "Cat_Delta"]
    abl_models = [m for m in abl_models if m in pte_14]
    abl_p = []
    abl_dm = {}
    for m in abl_models:
        r = dm_hac_test(y_te, pte_14[m], pte_lag14[m], h=1)
        abl_dm[m] = r
        abl_p.append(r["p_raw"])
    abl_holm = dict(zip(abl_models, holm_adjust(abl_p)))
    table5_ablation = {
        m: {
            "mir12_rmse": round(rmse(y_te, pte_14[m]), 4),
            "lag14_rmse": round(rmse(y_te, pte_lag14[m]), 4),
            "dm_hac": abl_dm[m]["dm"],
            "p_raw": abl_dm[m]["p_raw"],
            "p_holm5": abl_holm[m],
        }
        for m in abl_models
    }

    # Ablasi Embargo (gap=5) vs Tanpa Embargo (gap=0) pada Train/Val/Test
    tr0, va0, te0 = get_splits(len(y_all), gap=0)
    mir12_0, _ = select_mir12(X_all.iloc[tr0], y_all[tr0], k=12, seed=42)
    pva_0, pte_0 = make_14_models_predictions(
        X_all.iloc[tr0][mir12_0], y_all[tr0], st_all[tr0], mt_all[tr0],
        X_all.iloc[va0][mir12_0], y_all[va0], st_all[va0], mt_all[va0],
        X_all.iloc[te0][mir12_0], y_all[te0], st_all[te0], mt_all[te0],
        full_spei_series=None, seed=42,
        ar_tr=X_all.iloc[tr0][["SPEI_t", "SPEI_lag1"]].values,
        ar_va=X_all.iloc[va0][["SPEI_t", "SPEI_lag1"]].values,
        ar_te=X_all.iloc[te0][["SPEI_t", "SPEI_lag1"]].values,
    )
    embargo_ablation = {
        m: {
            "val_rmse_gap5": round(rmse(y_va, pva_14[m]), 4),
            "val_rmse_gap0": round(rmse(y_all[va0], pva_0[m]), 4),
            "test_rmse_gap5": round(rmse(y_te, pte_14[m]), 4),
            "test_rmse_gap0": round(rmse(y_all[te0], pte_0[m]), 4),
        }
        for m in ["Ridge_Direct", "RF_Delta", "XGB_Delta", "Cat_Delta"] if m in pte_14
    }

    # =========================================================================
    # E. TABEL 6 & 7: MULTI-HORIZON (t+1..t+6), ROLLING-ORIGIN (W1–W4), & 5 GRID
    # =========================================================================
    table6_horizons = {}
    h_preds_store = {}
    for h in (1, 2, 3, 6):
        Xh, yh, _, mth, sth, _ = build_features(df, date_col, spei_col, meteo_cols, horizon=h)
        trh, vah, teh = get_splits(len(yh), gap=5)
        mirh, _ = select_mir12(Xh.iloc[trh], yh[trh], k=12, seed=42)
        pvah, pteh = make_14_models_predictions(
            Xh.iloc[trh][mirh], yh[trh], sth[trh], mth[trh],
            Xh.iloc[vah][mirh], yh[vah], sth[vah], mth[vah],
            Xh.iloc[teh][mirh], yh[teh], sth[teh], mth[teh],
            full_spei_series=None, seed=42, horizon=h,
            ar_tr=Xh.iloc[trh][["SPEI_t", "SPEI_lag1"]].values,
            ar_va=Xh.iloc[vah][["SPEI_t", "SPEI_lag1"]].values,
            ar_te=Xh.iloc[teh][["SPEI_t", "SPEI_lag1"]].values,
        )
        h_preds_store[h] = (yh[vah], pvah, yh[teh], pteh)
        rb0 = round(rmse(yh[teh], pteh["B0_Climatology"]), 4)
        rb1 = round(rmse(yh[teh], pteh["B1_Persistence"]), 4)
        rbest = min(rb0, rb1)
        bname = "B0_Climatology" if rb0 < rb1 else "B1_Persistence"
        rrdir = round(rmse(yh[teh], pteh["Ridge_Direct"]), 4)
        rcdir = round(rmse(yh[teh], pteh["Cat_Direct"]), 4) if "Cat_Direct" in pteh else None
        rcdel = round(rmse(yh[teh], pteh["Cat_Delta"]), 4) if "Cat_Delta" in pteh else None
        rxdel = round(rmse(yh[teh], pteh["XGB_Delta"]), 4) if "XGB_Delta" in pteh else None
        sd_yh = round(float(np.std(yh[teh], ddof=1)), 4)
        dm_r_b0 = dm_hac_test(yh[teh], pteh["Ridge_Direct"], pteh["B0_Climatology"], h=h)
        dm_cd_b0 = dm_hac_test(yh[teh], pteh["Cat_Delta"], pteh["B0_Climatology"], h=h) if "Cat_Delta" in pteh else None
        dm_cd_dir = dm_hac_test(yh[teh], pteh["Cat_Delta"], pteh["Cat_Direct"], h=h) if "Cat_Delta" in pteh else None
        table6_horizons[f"t+{h}"] = {
            "n_test": int(len(yh[teh])),
            "test_sd": sd_yh,
            "sqrt2_sd": round(float(np.sqrt(2.0) * sd_yh), 4),
            "B0_Climatology": rb0, "B1_Persistence": rb1,
            "best_base": bname, "best_base_rmse": rbest,
            "Ridge_Direct": rrdir,
            "SS_RDir_vs_BestBase_pct": round((1.0 - (rrdir / rbest) ** 2) * 100.0, 2),
            "Cat_Direct": rcdir, "Cat_Delta": rcdel,
            "SS_CatDel_vs_BestBase_pct": round((1.0 - (rcdel / rbest) ** 2) * 100.0, 2) if rcdel else None,
            "XGB_Delta": rxdel,
            "SS_XGBDel_vs_BestBase_pct": round((1.0 - (rxdel / rbest) ** 2) * 100.0, 2) if rxdel else None,
            "DM_RDir_vs_B0": [dm_r_b0["dm"], dm_r_b0["p_raw"]],
            "DM_CatDel_vs_B0": [dm_cd_b0["dm"], dm_cd_b0["p_raw"]] if dm_cd_b0 else None,
            "DM_CatDel_vs_CatDir": [dm_cd_dir["dm"], dm_cd_dir["p_raw"]] if dm_cd_dir else None,
        }

    # Rolling-origin 4 jendela (W1-W4)
    win_defs = [
        ("W1_2010_13", slice(0, 240), slice(245, 280), slice(285, 320)),
        ("W2_2013_16", slice(0, 280), slice(285, 320), slice(325, 360)),
        ("W3_2017_20", slice(0, 325), slice(330, 365), slice(370, 405)),
        ("W4_2021_24", slice(0, 380), slice(385, 420), slice(425, 460)),
    ]
    table7_windows = {}
    for wname, wtr, wva, wte in win_defs:
        mirw, _ = select_mir12(X_all.iloc[wtr], y_all[wtr], k=12, seed=42)
        _, ptew = make_14_models_predictions(
            X_all.iloc[wtr][mirw], y_all[wtr], st_all[wtr], mt_all[wtr],
            X_all.iloc[wva][mirw], y_all[wva], st_all[wva], mt_all[wva],
            X_all.iloc[wte][mirw], y_all[wte], st_all[wte], mt_all[wte],
            full_spei_series=None, seed=42,
            ar_tr=X_all.iloc[wtr][["SPEI_t", "SPEI_lag1"]].values,
            ar_va=X_all.iloc[wva][["SPEI_t", "SPEI_lag1"]].values,
            ar_te=X_all.iloc[wte][["SPEI_t", "SPEI_lag1"]].values,
        )
        rp = round(rmse(y_all[wte], ptew["B1_Persistence"]), 4)
        rr = round(rmse(y_all[wte], ptew["Ridge_Direct"]), 4)
        rcd = round(rmse(y_all[wte], ptew["Cat_Direct"]), 4) if "Cat_Direct" in ptew else None
        rcl = round(rmse(y_all[wte], ptew["Cat_Delta"]), 4) if "Cat_Delta" in ptew else None
        dm_w = dm_hac_test(y_all[wte], ptew["Cat_Delta"], ptew["Cat_Direct"], h=1) if "Cat_Delta" in ptew else None
        table7_windows[wname] = {
            "n_test": int(len(y_all[wte])),
            "B1_Persistence": rp,
            "Ridge_Direct": rr, "SS_RDir_pct": round((1.0 - (rr / rp) ** 2) * 100.0, 1),
            "Cat_Direct": rcd, "Cat_Delta": rcl,
            "SS_CatDel_pct": round((1.0 - (rcl / rp) ** 2) * 100.0, 1) if rcl else None,
            "DM_CatDel_vs_CatDir": [dm_w["dm"], dm_w["p_raw"]] if dm_w else None,
        }

    # Lima sel grid (Lag-14) jika kolom tersedia
    table7_grids = {}
    for gname, hints in GRID_SPEI06_HINTS.items():
        gcol = detect_spei_col(df, hints)
        if gcol is None:
            continue
        Xg, yg, _, mtg, stg, lag14g = build_features(df, date_col, gcol, [], horizon=1)
        trg, vag, teg = get_splits(len(yg), gap=5)
        _, pteg = make_14_models_predictions(
            Xg.iloc[trg][lag14g], yg[trg], stg[trg], mtg[trg],
            Xg.iloc[vag][lag14g], yg[vag], stg[vag], mtg[vag],
            Xg.iloc[teg][lag14g], yg[teg], stg[teg], mtg[teg],
            full_spei_series=None, seed=42,
        )
        rp = round(rmse(yg[teg], pteg["B1_Persistence"]), 4)
        rb0g = round(rmse(yg[teg], pteg["B0_Climatology"]), 4)
        rr = round(rmse(yg[teg], pteg["Ridge_Direct"]), 4)
        rcd = round(rmse(yg[teg], pteg["Cat_Direct"]), 4) if "Cat_Direct" in pteg else None
        rcl = round(rmse(yg[teg], pteg["Cat_Delta"]), 4) if "Cat_Delta" in pteg else None
        dm_g = dm_hac_test(yg[teg], pteg["Cat_Delta"], pteg["Cat_Direct"], h=1) if "Cat_Delta" in pteg else None
        table7_grids[gname] = {
            "col": gcol,
            "test_sd": round(float(np.std(yg[teg], ddof=1)), 2),
            "rho1": round(float(np.corrcoef(yg[teg][:-1], yg[teg][1:])[0, 1]), 3),
            "B0_Climatology": rb0g, "B1_Persistence": rp,
            "Ridge_Direct": rr, "SS_RDir_pct": round((1.0 - (rr / rp) ** 2) * 100.0, 1),
            "Cat_Direct": rcd, "Cat_Delta": rcl,
            "SS_CatDel_pct": round((1.0 - (rcl / rp) ** 2) * 100.0, 1) if rcl else None,
            "DM_CatDel_vs_CatDir": [dm_g["dm"], dm_g["p_raw"]] if dm_g else None,
        }

    # =========================================================================
    # F. TABEL 8: OUTAGE (100 MASK) & END-OF-SERIES LATENCY (3 MODEL + CI)
    # =========================================================================
    Xtr_df, Xte_df = X_all.iloc[tr][mir12_cols].copy(), X_all.iloc[te][mir12_cols].copy()
    exo_ok = ("PRECTOTCORR_t" in X_all.columns and "GWETTOP_t" in X_all.columns)
    if exo_ok:
        mu_p, sd_p = X_all.iloc[tr]["PRECTOTCORR_t"].mean(), X_all.iloc[tr]["PRECTOTCORR_t"].std(ddof=0)
        mu_g, sd_g = X_all.iloc[tr]["GWETTOP_t"].mean(), X_all.iloc[tr]["GWETTOP_t"].std(ddof=0)
        z_tr = 0.5 * ((X_all.iloc[tr]["PRECTOTCORR_t"].values - mu_p) / sd_p + (X_all.iloc[tr]["GWETTOP_t"].values - mu_g) / sd_g)
        z_te = 0.5 * ((X_all.iloc[te]["PRECTOTCORR_t"].values - mu_p) / sd_p + (X_all.iloc[te]["GWETTOP_t"].values - mu_g) / sd_g)
    else:
        z_tr, z_te = np.zeros(len(y_tr)), np.zeros(len(y_te))

    l1_tr, l2_tr = X_all.iloc[tr]["SPEI_lag1"].values, X_all.iloc[tr]["SPEI_lag2"].values
    coef_ar2, *_ = np.linalg.lstsq(np.column_stack([l1_tr, l2_tr]), s_tr, rcond=None)
    coef_ar2z, *_ = np.linalg.lstsq(np.column_stack([l1_tr, l2_tr, z_tr]), s_tr, rcond=None)

    mice_cols = ["SPEI_t", "SPEI_lag1", "SPEI_lag2"] + [f"{m}_t" for m in meteo_cols if f"{m}_t" in X_all.columns]
    mice_imp = IterativeImputer(estimator=BayesianRidge(), max_iter=15, random_state=42).fit(X_all.iloc[tr][mice_cols].values)

    outage_models = {"Ridge_Direct": (Ridge(alpha=1.0).fit(Xtr_df.values, y_tr), False)}
    if HAS_CAT:
        outage_models["CatBoost_Direct"] = (CatBoostRegressor(iterations=120, depth=4, learning_rate=0.04, random_seed=42, verbose=0).fit(Xtr_df.values, y_tr), False)
        outage_models["CatBoost_Delta"] = (CatBoostRegressor(iterations=120, depth=4, learning_rate=0.03, random_seed=42, verbose=0).fit(Xtr_df.values, y_tr - s_tr), True)

    clean_out_rmse = {
        mn: round(rmse(y_te, (s_te + md.predict(Xte_df.values)) if isd else md.predict(Xte_df.values)), 4)
        for mn, (md, isd) in outage_models.items()
    }
    n_te = len(y_te)
    l1_te_true = X_all.iloc[te]["SPEI_lag1"].values
    l2_te_true = X_all.iloc[te]["SPEI_lag2"].values
    mice_base_te = X_all.iloc[te][mice_cols].values
    s_pre6 = st_all[te.start - 6: te.start]

    def rec_series(mask_b, meth):
        sh = s_te.copy()
        for t in range(n_te):
            if not mask_b[t]:
                continue
            p1 = sh[t - 1] if t >= 1 else l1_te_true[t]
            p2 = sh[t - 2] if t >= 2 else (l1_te_true[t - 1] if t == 1 else l2_te_true[t])
            if meth == "LOCF":
                sh[t] = p1
            elif meth == "AR2":
                sh[t] = coef_ar2[0] * p1 + coef_ar2[1] * p2
            elif meth == "AR2_Z":
                sh[t] = coef_ar2z[0] * p1 + coef_ar2z[1] * p2 + coef_ar2z[2] * z_te[t]
            else:
                rm = mice_base_te[t].copy()
                rm[0], rm[1], rm[2] = np.nan, p1, p2
                sh[t] = mice_imp.transform(rm.reshape(1, -1))[0, 0]
        return sh

    def rebuild_X(s_rec):
        sc = pd.Series(np.concatenate([s_pre6, s_rec]))
        Xm = Xte_df.copy()
        if "SPEI_t" in Xm: Xm["SPEI_t"] = sc.iloc[6:].values
        if "SPEI_lag1" in Xm: Xm["SPEI_lag1"] = sc.shift(1).iloc[6:].values
        if "SPEI_lag2" in Xm: Xm["SPEI_lag2"] = sc.shift(2).iloc[6:].values
        for w in (2, 3, 6):
            if f"SPEI_delta_{w}" in Xm: Xm[f"SPEI_delta_{w}"] = (sc - sc.shift(w)).iloc[6:].values
            if f"SPEI_mean_{w}" in Xm: Xm[f"SPEI_mean_{w}"] = sc.rolling(w, min_periods=w).mean().iloc[6:].values
            if f"SPEI_ema_{w}" in Xm: Xm[f"SPEI_ema_{w}"] = sc.ewm(span=w, adjust=False).mean().iloc[6:].values
        return Xm.values

    rng = np.random.default_rng(42)
    table8_outage = {
        "clean_rmse": clean_out_rmse,
        "ols_ar2": [round(float(x), 4) for x in coef_ar2],
        "ols_ar2z": [round(float(x), 4) for x in coef_ar2z],
        "scenarios": {},
        "end_of_series_latency": {},
    }
    for sc_name in ("MCAR_10", "MCAR_20", "MCAR_30", "MAR_Dry", "Block_3M"):
        masks = []
        for _ in range(args.masks):
            if sc_name == "MCAR_10": m = rng.random(n_te) < 0.10
            elif sc_name == "MCAR_20": m = rng.random(n_te) < 0.20
            elif sc_name == "MCAR_30": m = rng.random(n_te) < 0.30
            elif sc_name == "MAR_Dry": m = rng.random(n_te) < np.where(l1_te_true <= -0.5, 0.35, 0.08)
            else:
                m = np.zeros(n_te, dtype=bool)
                for si in rng.choice(np.arange(0, n_te - 3), size=4, replace=False):
                    m[si:si + 3] = True
            if not m.any(): m[rng.integers(0, n_te)] = True
            masks.append(m)

        sc_res = {}
        for mn, (md, isd) in outage_models.items():
            pm = {meth: [] for meth in ("LOCF", "AR2", "AR2_Z", "MICE_Lag")}
            for mk in masks:
                for meth in pm:
                    sr = rec_series(mk, meth)
                    Xr = rebuild_X(sr)
                    pr = (sr + md.predict(Xr)) if isd else md.predict(Xr)
                    pm[meth].append(rmse(y_te, pr))
            locf_a, ar2z_a, mice_a = np.array(pm["LOCF"]), np.array(pm["AR2_Z"]), np.array(pm["MICE_Lag"])
            infl_l = max(float(np.mean(locf_a)) - clean_out_rmse[mn], 1e-9)
            diff_ma = mice_a - ar2z_a
            sc_res[mn] = {
                meth: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4)}
                for meth, v in pm.items()
            }
            sc_res[mn]["rel_red_AR2Z_vs_LOCF_pct"] = round((1.0 - float(np.mean(ar2z_a)) / float(np.mean(locf_a))) * 100.0, 2)
            sc_res[mn]["rel_red_MICE_vs_LOCF_pct"] = round((1.0 - float(np.mean(mice_a)) / float(np.mean(locf_a))) * 100.0, 2)
            sc_res[mn]["infl_red_MICE_vs_LOCF_pct"] = round(((float(np.mean(locf_a)) - float(np.mean(mice_a))) / infl_l) * 100.0, 2)
            sc_res[mn]["paired_MICE_minus_AR2Z_ci95"] = [
                round(float(np.percentile(diff_ma, 2.5)), 4),
                round(float(np.percentile(diff_ma, 97.5)), 4),
            ]
        table8_outage["scenarios"][sc_name] = sc_res

    for lk in (1, 2, 3):
        l_preds = {mn: {} for mn in outage_models}
        l_out = {}
        for mn, (md, isd) in outage_models.items():
            l_out[mn] = {}
            for meth in ("LOCF", "AR2", "AR2_Z", "MICE_Lag"):
                pk = np.empty(n_te)
                for t in range(n_te):
                    mk = np.zeros(n_te, dtype=bool)
                    mk[max(0, t - lk + 1):t + 1] = True
                    sr = rec_series(mk, meth)
                    Xr = rebuild_X(sr)
                    pk[t] = (sr[t] + md.predict(Xr[t:t+1])[0]) if isd else md.predict(Xr[t:t+1])[0]
                l_preds[mn][meth] = pk
                l_out[mn][meth] = round(rmse(y_te, pk), 4)
            l_out[mn]["rel_red_AR2Z_vs_LOCF_pct"] = round((1.0 - l_out[mn]["AR2_Z"] / l_out[mn]["LOCF"]) * 100.0, 2)
            l_out[mn]["rel_red_MICE_vs_LOCF_pct"] = round((1.0 - l_out[mn]["MICE_Lag"] / l_out[mn]["LOCF"]) * 100.0, 2)
        if "CatBoost_Direct" in l_preds and "CatBoost_Delta" in l_preds:
            dm_cd_lat = dm_hac_test(y_te, l_preds["CatBoost_Direct"]["AR2_Z"], l_preds["CatBoost_Delta"]["AR2_Z"], h=1)
            l_out["DM_CatDir_vs_CatDel_AR2Z"] = {"dm": dm_cd_lat["dm"], "p_raw": dm_cd_lat["p_raw"], "ci95_mse": dm_cd_lat["ci95"]}
        table8_outage["end_of_series_latency"][f"delay_{lk}m"] = l_out

    # =========================================================================
    # G. TABEL 9 & 10: KALIBRASI CHRISTOFFERSEN, WINKLER, BSS MULTI-REF,
    #    RELIABILITY BINS, EXPECTED COST, REV, & HAC SENSITIVITY (SAME PIPELINE!)
    # =========================================================================
    rng_b = np.random.default_rng(42)
    B = 2000
    boot_te_idx = [stationary_boot_idx(len(y_te), 6.0, rng_b) for _ in range(B)]

    ybin_te_10 = (y_te <= -1.0).astype(int)
    ybin_te_15 = (y_te <= -1.5).astype(int)
    ybin_va_10 = (y_va <= -1.0).astype(int)
    bs_ref_te10 = float(np.mean((float(np.mean(ybin_te_10)) - ybin_te_10) ** 2))
    bs_ref_va10 = float(np.mean((float(np.mean(ybin_va_10)) - ybin_te_10) ** 2))
    bs_ref_tr10 = float(np.mean((float(np.mean(y_tr <= -1.0)) - ybin_te_10) ** 2))
    bs_ref_te15 = float(np.mean((float(np.mean(ybin_te_15)) - ybin_te_15) ** 2))

    eval_models_9 = [m for m in ("B1_Persistence", "B2_AR(2)", "Ridge_Direct", "MLP_Delta", "RF_Delta", "XGB_Delta", "Cat_Delta") if m in pte_14]
    table9_cal = {}
    reliability_bins = {}
    edges = np.linspace(0.0, 1.0, 6)

    for m in eval_models_9:
        r_va = y_va - pva_14[m]
        r_te = y_te - pte_14[m]
        sd_va = float(np.std(r_va, ddof=1))
        abs_va = np.sort(np.abs(r_va))
        n_va = len(abs_va)

        # Shapiro-Wilk & Excess Kurtosis
        sw_w, sw_p = stats.shapiro(r_va)
        ek = float(stats.kurtosis(r_va, fisher=True))

        cal_e = {"shapiro_W": round(float(sw_w), 4), "shapiro_p": round(float(sw_p), 4), "excess_kurtosis": round(ek, 2)}
        for lvl, z_val in ((80, stats.norm.ppf(0.90)), (95, stats.norm.ppf(0.975))):
            alpha = 1.0 - lvl / 100.0
            # Gaussian
            lo_g, hi_g = pte_14[m] - z_val * sd_va, pte_14[m] + z_val * sd_va
            cc_g = christoffersen_cc((y_te >= lo_g) & (y_te <= hi_g), lvl / 100.0)
            wi_g = winkler_score(y_te, lo_g, hi_g, alpha)
            # Split Conformal
            k_ord = min(n_va, int(np.ceil((n_va + 1) * (1.0 - alpha)))) - 1
            q_c = float(abs_va[k_ord])
            lo_c, hi_c = pte_14[m] - q_c, pte_14[m] + q_c
            cc_c = christoffersen_cc((y_te >= lo_c) & (y_te <= hi_c), lvl / 100.0)
            wi_c = winkler_score(y_te, lo_c, hi_c, alpha)
            cal_e[f"gauss_{lvl}"] = {**cc_g, "winkler": wi_g}
            cal_e[f"conf_{lvl}"] = {**cc_c, "winkler": wi_c, "q_halfwidth": round(q_c, 4)}

        # BSS
        p_te_10 = empirical_prob(y_va, pva_14[m], pte_14[m], tau=-1.0)
        p_te_15 = empirical_prob(y_va, pva_14[m], pte_14[m], tau=-1.5)
        bs10 = float(np.mean((p_te_10 - ybin_te_10) ** 2))
        bs15 = float(np.mean((p_te_15 - ybin_te_15) ** 2))
        bss_boot = []
        for bidx in boot_te_idx:
            yb = ybin_te_10[bidx]
            br = float(np.mean((float(np.mean(yb)) - yb) ** 2))
            if br > 1e-12:
                bss_boot.append(100.0 * (1.0 - float(np.mean((p_te_10[bidx] - yb) ** 2)) / br))
        cal_e["BS_10"] = round(bs10, 4)
        cal_e["BSS_test10_pct"] = round(100.0 * (1.0 - bs10 / bs_ref_te10), 1)
        cal_e["BSS_test10_ci95"] = [round(float(np.percentile(bss_boot, 2.5)), 1), round(float(np.percentile(bss_boot, 97.5)), 1)]
        cal_e["BSS_val10_pct"] = round(100.0 * (1.0 - bs10 / bs_ref_va10), 1)
        cal_e["BSS_train10_pct"] = round(100.0 * (1.0 - bs10 / bs_ref_tr10), 1)
        cal_e["BSS_test15_pct"] = round(100.0 * (1.0 - bs15 / bs_ref_te15), 1)
        table9_cal[m] = cal_e

        # Reliability bins
        bins_m = []
        for b in range(5):
            lo_e, hi_e = edges[b], edges[b + 1]
            mk = (p_te_10 >= lo_e) & (p_te_10 <= hi_e if b == 4 else p_te_10 < hi_e)
            cnt = int(np.sum(mk))
            if cnt > 0:
                bins_m.append({"bin": f"[{lo_e:.1f},{hi_e:.1f}]", "n_b": cnt,
                               "mean_p": round(float(np.mean(p_te_10[mk])), 3),
                               "obs_freq": round(float(np.mean(ybin_te_10[mk])), 3)})
        reliability_bins[m] = bins_m

    # HAC bandwidth sensitivity pada model tabel 3 yang SAMA PERSIS
    hac_sens = {}
    for m in [x for x in ("Cat_Delta", "XGB_Delta", "RF_Delta") if x in pte_14]:
        hac_sens[m] = {}
        for q in (3, 5, 8, 10):
            rp = dm_hac_test(y_te, pte_14[m], pte_14["B1_Persistence"], h=1, q=q)
            rr = dm_hac_test(y_te, pte_14[m], pte_14["Ridge_Direct"], h=1, q=q)
            hac_sens[m][f"q={q}"] = {
                "vs_Persist": [rp["dm"], rp["p_raw"], rp["n_eff_dm"]],
                "vs_RDir": [rr["dm"], rr["p_raw"], rr["n_eff_dm"]],
            }

    # Tabel 10: Expected Cost & REV pada t+1 (C/L 0.20, 0.30, 0.50, 0.60) & t+2, t+3 (C/L 0.20)
    # Menggunakan N_test aktual tiap horizon (68, 67, 66) dari PIPELINE YANG SAMA PERSIS!
    table10_cost_rev = {}
    for h, cl_list in [(1, [0.20, 0.30, 0.50, 0.60]), (2, [0.20]), (3, [0.20])]:
        yva_h, pva_h, yte_h, pte_h = h_preds_store[h]
        yb_va = (yva_h <= -1.0).astype(int)
        yb_te = (yte_h <= -1.0).astype(int)
        n_te_h = len(yb_te)
        rng_h = np.random.default_rng(42 + h)
        bidx_h = [stationary_boot_idx(n_te_h, 6.0, rng_h) for _ in range(B)]

        prob_va_h = {m: empirical_prob(yva_h, pva_h[m], pva_h[m], -1.0) for m in pte_h}
        prob_te_h = {m: empirical_prob(yva_h, pva_h[m], pte_h[m], -1.0) for m in pte_h}

        for cl in cl_list:
            key_hc = f"t+{h}_CL{cl:.2f}"
            table10_cost_rev[key_hc] = {"n_test": int(n_te_h), "n_drought_test": int(np.sum(yb_te)), "models": {}}
            # Pilih P*_val optimal pada Validasi (grid 0.05..0.90)
            pstar = {}
            for m in pte_h:
                best_r, bp = -1e9, cl
                for thr in np.linspace(0.05, 0.90, 86):
                    rv = rev_and_cost(yb_va, prob_va_h[m], thr, cl)["rev"]
                    if rv > best_r:
                        best_r, bp = rv, float(thr)
                pstar[m] = round(bp, 2)

            ref_pers = rev_and_cost(yb_te, prob_te_h["B1_Persistence"], pstar["B1_Persistence"], cl)
            for m in [x for x in ("B1_Persistence", "Ridge_Direct", "RF_Delta", "Cat_Direct", "XGB_Delta", "Cat_Delta") if x in pte_h]:
                res_m = rev_and_cost(yb_te, prob_te_h[m], pstar[m], cl)
                rev_b, dc_b = [], []
                for bi in bidx_h:
                    rm_b = rev_and_cost(yb_te[bi], prob_te_h[m][bi], pstar[m], cl)
                    rp_b = rev_and_cost(yb_te[bi], prob_te_h["B1_Persistence"][bi], pstar["B1_Persistence"], cl)
                    if not np.isnan(rm_b["rev"]):
                        rev_b.append(rm_b["rev"])
                    dc_b.append(rm_b["e_model"] - rp_b["e_model"])
                n_ev = res_m["tp"] + res_m["fn"]
                table10_cost_rev[key_hc]["models"][m] = {
                    "P_star_val": pstar[m],
                    "tp_fp_fn": f"{res_m['tp']}/{res_m['fp']}/{res_m['fn']}",
                    "pod_pct": round(100.0 * res_m["tp"] / max(n_ev, 1), 1),
                    "pod_wilson95": wilson_ci(res_m["tp"], n_ev),
                    "e_cost": round(res_m["e_model"], 4),
                    "e_clim": round(res_m["e_clim"], 4),
                    "delta_cost_vs_pers": round(res_m["e_model"] - ref_pers["e_model"], 4),
                    "delta_cost_ci95": [round(float(np.percentile(dc_b, 2.5)), 4), round(float(np.percentile(dc_b, 97.5)), 4)],
                    "rev": round(res_m["rev"], 3),
                    "rev_ci95": [round(float(np.percentile(rev_b, 2.5)), 3), round(float(np.percentile(rev_b, 97.5)), 3)] if rev_b else [None, None],
                }

    # Kurva REV penuh C/L 0.1..0.9 untuk Gambar 5(b)
    rev_curves = {}
    for h in (1, 2):
        yva_h, pva_h, yte_h, pte_h = h_preds_store[h]
        yb_va = (yva_h <= -1.0).astype(int)
        yb_te = (yte_h <= -1.0).astype(int)
        pva_m = {m: empirical_prob(yva_h, pva_h[m], pva_h[m], -1.0) for m in pte_h}
        pte_m = {m: empirical_prob(yva_h, pva_h[m], pte_h[m], -1.0) for m in pte_h}
        rev_curves[f"t+{h}"] = {}
        for m in [x for x in ("B1_Persistence", "Ridge_Direct", "RF_Delta", "XGB_Delta", "Cat_Delta") if x in pte_h]:
            curve_m = []
            for cl in [round(0.1 * k, 1) for k in range(1, 10)]:
                best_r, bp = -1e9, cl
                for thr in np.linspace(0.05, 0.90, 86):
                    rv = rev_and_cost(yb_va, pva_m[m], thr, cl)["rev"]
                    if rv > best_r:
                        best_r, bp = rv, float(thr)
                rv_te = rev_and_cost(yb_te, pte_m[m], bp, cl)["rev"]
                curve_m.append(round(rv_te, 3) if not np.isnan(rv_te) else None)
            rev_curves[f"t+{h}"][m] = curve_m

    # =========================================================================
    # H. SIMPAN KELUARAN TERPADU
    # =========================================================================
    report = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count_logical": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2) if HAS_PSUTIL else "unknown",
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__ if HAS_SM else "not_installed",
            "xgboost": xgb.__version__ if HAS_XGB else "not_installed",
            "catboost": __import__("catboost").__version__ if HAS_CAT else "not_installed",
        },
        "runtime_benchmark_14models": {
            "reps": args.bench_reps,
            "mean_sec": round(float(np.mean(bench_times)), 2),
            "sd_sec": round(float(np.std(bench_times, ddof=1)), 2) if len(bench_times) > 1 else 0.0,
        },
        "mir12_cols": mir12_cols,
        "table2_stats": table2_stats,
        "table3_14models": table3_14models,
        "table4_paired_main": table4_paired_main,
        "table4_high_capacity": high_cap_out,
        "table4_residual_on_ridge": res_ridge_out,
        "table5_ablation_mir12_vs_lag14": table5_ablation,
        "embargo_vs_no_embargo": embargo_ablation,
        "table6_horizons": table6_horizons,
        "table7_windows": table7_windows,
        "table7_grids": table7_grids,
        "table8_outage": table8_outage,
        "table9_calibration": table9_cal,
        "reliability_bins": reliability_bins,
        "hac_bandwidth_sensitivity": hac_sens,
        "table10_cost_rev": table10_cost_rev,
        "rev_curves_01_09": rev_curves,
        "total_runtime_sec": round(time.perf_counter() - t_global, 2),
    }

    with open("unified_results.json", "w", encoding="utf-8") as fj:
        json.dump(report, fj, indent=2, ensure_ascii=False)

    print(f"[SELESAI] unified_results.json tersimpan dalam {report['total_runtime_sec']} detik.")
    print(f"  Runtime 14-model (n={args.bench_reps}): {report['runtime_benchmark_14models']['mean_sec']} ± {report['runtime_benchmark_14models']['sd_sec']} s")
    print(f"  Cat_Delta t+1 (SINGLE UNIFIED VALUE): Tabel 3 = {table3_14models.get('Cat_Delta', {}).get('rmse_seed42')}, "
          f"Tabel 6 = {table6_horizons['t+1']['Cat_Delta']}, Tabel 8 clean = {clean_out_rmse.get('CatBoost_Delta')}")
    print(f"  Embargo vs No-Embargo: {embargo_ablation}")


if __name__ == "__main__":
    main()