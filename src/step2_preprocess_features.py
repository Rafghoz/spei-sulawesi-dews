# ==============================================================================
# STEP 2: PRAPEMROSESAN, REKAYASA FITUR JENDELA, & PEMBAGIAN DATA DINAMIS
# File: src/step2_preprocess_features.py
# ==============================================================================
import os
from functools import partial
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression

SENSOR_COLS = [
    "ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN",
    "RH2M", "WS2M", "GWETTOP", "PRECTOTCORR"
]

def engineer_features(
    df_in: pd.DataFrame,
    sp_col: str = "SPEI_06",
    w: int = 6,
    horizon: int = 1
):
    """
    Membentuk himpunan fitur Lag-Only (8 fitur) dan kandidat Lag+Meteo (34 fitur)
    yang konsisten dengan pipa Google Colab V25.
    """
    d = df_in.copy()
    d["DATE"] = pd.to_datetime(d["DATE"])
    d = d.sort_values("DATE").reset_index(drop=True)
    d["MONTH"] = d["DATE"].dt.month

    # 1. Komponen siklus musiman
    d["MONTH_SIN"] = np.sin(2.0 * np.pi * d["MONTH"] / 12.0)
    d["MONTH_COS"] = np.cos(2.0 * np.pi * d["MONTH"] / 12.0)

    # 2. Lag autoregresif untuk AR(2) dan Lag-Only
    d["lag1_ar"] = d[sp_col].shift(1).bfill()
    d["lag2_ar"] = d[sp_col].shift(2).bfill()

    # 3. Kandidat fitur Lag+Meteo (34 fitur)
    meteo_all_cols = list(SENSOR_COLS) + [sp_col, "MONTH_SIN", "MONTH_COS"]
    for c in ["PRECTOTCORR", "GWETTOP", "T2M_MAX", "RH2M", "ALLSKY_SFC_SW_DWN", sp_col]:
        d[f"{c}_lag1"] = d[c].shift(1).bfill()
        d[f"{c}_delta_w"] = (d[c] - d[c].shift(w)).bfill()
        d[f"{c}_mean_w"] = d[c].rolling(w, min_periods=1).mean().bfill()
        d[f"{c}_ema_w"] = d[c].ewm(span=w, adjust=False).mean().bfill()
        meteo_all_cols += [f"{c}_lag1", f"{c}_delta_w", f"{c}_mean_w", f"{c}_ema_w"]

    # 4. Himpunan fitur Lag-Only (8 fitur univariat)
    lag_only_cols = [
        sp_col, "lag1_ar", "lag2_ar",
        f"{sp_col}_delta_w", f"{sp_col}_mean_w", f"{sp_col}_ema_w",
        "MONTH_SIN", "MONTH_COS"
    ]

    # 5. Target Y_{t+h}
    tg_col_t1 = f"TARGET_{sp_col}_t1"
    if horizon == 1 and tg_col_t1 in d.columns:
        d["Y_TARGET"] = d[tg_col_t1]
    else:
        d["Y_TARGET"] = d[sp_col].shift(-horizon)

    # Potong 6 bulan awal (burn-in jendela maksimum w=6)
    d_eff_all = d.iloc[6:].reset_index(drop=True)
    # Baris efektif yang sudah memiliki nilai observasi Y_{t+h}
    d_eff_valid = d_eff_all.dropna(subset=["Y_TARGET"]).reset_index(drop=True)
    # Baris paling akhir (bulan berjalan terkini) untuk proyeksi ke depan
    latest_row = d_eff_all.iloc[[-1]].reset_index(drop=True)

    return d_eff_valid, latest_row, lag_only_cols, meteo_all_cols


def get_dynamic_splits(n_eff: int, gap: int = 5, horizon: int = 1):
    """
    Menghasilkan indeks pembagian Latih, Validasi, dan Uji berjeda 5 bulan:
    - Jika data berukuran standar (n_eff = 461 pada h=1), gunakan pembagian eksak Colab:
      Latih = 0..314 (315 bln), Val = 320..387 (68 bln), Uji = 393..n_eff (68 bln pada h=1).
    - Jika data mentah bertambah atau berkurang, proporsi disesuaikan secara otomatis.
    """
    expected_std_len = 461 - (horizon - 1)
    if n_eff == expected_std_len and gap == 5:
        n_tr = 315
        n_va = 68
    else:
        usable = max(30, n_eff - 2 * gap)
        n_va = max(12, int(round(usable * 0.1508)))
        n_te_est = max(12, int(round(usable * 0.1508)))
        n_tr = usable - n_va - n_te_est

    tr_sl = slice(0, n_tr)
    va_sl = slice(n_tr + gap, n_tr + gap + n_va)
    te_sl = slice(n_tr + gap + n_va + gap, n_eff)

    split_meta = {
        "n_eff": n_eff,
        "n_train": n_tr,
        "n_val": n_va,
        "n_test": max(0, n_eff - (n_tr + 2 * gap + n_va)),
        "gap": gap
    }
    return tr_sl, va_sl, te_sl, split_meta


def select_k12_mi_features(
    df_eff: pd.DataFrame,
    tr_sl: slice,
    cand_cols: list,
    target_fit: np.ndarray,
    sp_col: str
) -> list:
    """Memilih 12 fitur terbaik Mutual Information hanya pada Data Latih (bebas kebocoran data)."""
    selector = SelectKBest(score_func=partial(mutual_info_regression, random_state=42), k=12)
    selector.fit(df_eff.iloc[tr_sl][cand_cols], target_fit)
    f_k12 = [cand_cols[i] for i in selector.get_support(indices=True)]
    if sp_col not in f_k12:
        f_k12[0] = sp_col
    return f_k12


def reconstruct_missing_anchor(
    spei_lags: list,
    prectot_mm: float,
    gwettop_frac: float,
    rh2m_pct: float,
    method: str = "MICE+Lag"
):
    """Merekonstruksi jangkar SPEI_t secara sekuensial apabila sensor bulan berjalan hilang (Outage)."""
    valid_lags = [v for v in spei_lags if v is not None and not np.isnan(v)]
    if len(valid_lags) == 0:
        return 0.0, "Fallback Klimatologi (0,00)"
    if method == "LOCF (Statis)":
        return float(valid_lags[0]), "Imputasi Statis LOCF"
    elif method == "Kalman State-Space AR(2)":
        l1 = valid_lags[0]
        l2 = valid_lags[1] if len(valid_lags) > 1 else l1
        est = 0.72 * l1 + 0.14 * l2
        return float(np.clip(est, -3.5, 3.5)), "Rekonstruksi Rekursif Kalman AR(2)"
    else:
        l1 = valid_lags[0]
        l2 = valid_lags[1] if len(valid_lags) > 1 else l1
        p_anom = (prectot_mm - 140.0) / 72.4
        g_anom = (gwettop_frac - 0.62) / 0.12
        r_anom = (rh2m_pct - 78.0) / 5.5
        est = 0.68 * l1 + 0.12 * l2 + 0.10 * g_anom + 0.06 * p_anom + 0.04 * r_anom
        return float(np.clip(est, -3.5, 3.5)), "Rekonstruksi Sekuensial MICE+Lag"


if __name__ == "__main__":
    master_path = "DATA/PROCESSED/dataset_master_nasa_csic_clean.csv"
    if not os.path.exists(master_path):
        from src.step1_extract_raw import run_step1_pipeline
        df_m = run_step1_pipeline()
    else:
        df_m = pd.read_csv(master_path)

    df_eff, latest, l_cols, m_cols = engineer_features(df_m, sp_col="SPEI_06", w=6, horizon=1)
    tr_s, va_s, te_s, meta = get_dynamic_splits(len(df_eff), gap=5, horizon=1)
    print("=== [SELESAI STEP 2] VERIFIKASI PRAPEMROSESAN & SPLIT DINAMIS ===")
    print(f"Total baris efektif (N_eff) : {meta['n_eff']} bulan")
    print(f"Pembagian Data              : Latih={meta['n_train']} | Val={meta['n_val']} | Uji={meta['n_test']} (Jeda={meta['gap']} bln)")
    print(f"Jumlah fitur Lag-Only       : {len(l_cols)} fitur")
    print(f"Jumlah kandidat Lag+Meteo   : {len(m_cols)} fitur")