# ==============================================================================
# STEP 3: PELATIHAN 14 MODEL, UJI DM-HAC HOLM, & GENERATOR BULETIN DEWS
# File: src/step3_train_and_evaluate.py
# ==============================================================================
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2, kurtosis, norm, shapiro, skew, t as student_t
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.step2_preprocess_features import (
    engineer_features,
    get_dynamic_splits,
    select_k12_mi_features
)

warnings.filterwarnings("ignore")
np.random.seed(42)


def diebold_mariano_hac(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, max_lag: int = 5):
    """Menghitung statistik uji Diebold-Mariano dengan varians HAC Newey-West."""
    d = ((y_true - pred1) ** 2) - ((y_true - pred2) ** 2)
    n = len(d)
    d_bar = float(np.mean(d))
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    d_c = d - d_bar
    g0 = np.sum(d_c ** 2) / n
    g_sum = sum(
        2.0 * (1.0 - k / (max_lag + 1.0)) * (np.sum(d_c[k:] * d_c[:-k]) / n)
        for k in range(1, max_lag + 1)
    )
    var_hac = max(1e-12, (g0 + g_sum) / n)
    dm_stat = d_bar / np.sqrt(var_hac)
    p_val = 2.0 * (1.0 - student_t.cdf(abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_val)

def evaluate_picp_stage1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma_val: float,
    nominal_level: float = 0.80
) -> dict:
    """
    Menghitung PICP empiris, jumlah hit (N_hit/N_test), SE_iid, autokorelasi lag-1
    indikator cakupan (rho1), N_eff_cov, SE_corr, dan uji Kupiec POF (LR_POF, p_value).
    """
    n_test = len(y_true)
    alpha_tail = (1.0 - nominal_level) / 2.0
    z_crit = norm.ppf(1.0 - alpha_tail)

    low_b = y_pred - z_crit * sigma_val
    up_b = y_pred + z_crit * sigma_val

    ind_hit = ((y_true >= low_b) & (y_true <= up_b)).astype(float)
    n_hit = int(np.sum(ind_hit))
    picp_emp = float(n_hit / n_test)

    # 1. Galat Standar Binomial i.i.d. terhadap target nominal
    se_iid = float(np.sqrt(nominal_level * (1.0 - nominal_level) / n_test))

    # 2. Autokorelasi lag-1 pada indikator pelanggaran/cakupan & sampel efektif
    if np.std(ind_hit) > 1e-12 and n_test > 2:
        rho1 = float(np.corrcoef(ind_hit[1:], ind_hit[:-1])[0, 1])
        rho1_pos = max(0.0, min(0.95, rho1))  # Penalti konservatif jika berkorelasi positif
    else:
        rho1 = 0.0
        rho1_pos = 0.0

    n_eff_cov = float(n_test * (1.0 - rho1_pos) / (1.0 + rho1_pos))
    se_corr = float(np.sqrt(nominal_level * (1.0 - nominal_level) / max(1.0, n_eff_cov)))

    # 3. Uji Kupiec POF (Proportion of Failures Likelihood Ratio Test, df=1)
    n_fail = n_test - n_hit
    if 0 < n_hit < n_test:
        ll_null = n_fail * np.log(1.0 - nominal_level) + n_hit * np.log(nominal_level)
        ll_alt = n_fail * np.log(1.0 - picp_emp) + n_hit * np.log(picp_emp)
        lr_pof = float(-2.0 * (ll_null - ll_alt))
        p_kupiec = float(1.0 - chi2.cdf(lr_pof, df=1))
    else:
        lr_pof = np.nan
        p_kupiec = 0.0

    return {
        "N_Hit": f"{n_hit}/{n_test}",
        "PICP_80 (%)": round(picp_emp * 100.0, 2),
        "SE_iid (pp)": round(se_iid * 100.0, 2),
        "Rho1_Hit": round(rho1, 3),
        "N_eff_Cov": round(n_eff_cov, 1),
        "SE_corr (pp)": round(se_corr * 100.0, 2),
        "Kupiec_LR": round(lr_pof, 4),
        "Kupiec_p": round(p_kupiec, 4),
    }

def compute_winkler_interval_score(y_true: np.ndarray, low_b: np.ndarray, up_b: np.ndarray, alpha: float) -> float:
    """
    Menghitung rata-rata Winkler Interval Score (Gneiting & Raftery, 2007) pada tingkat signifikansi alpha.
    Semakin rendah nilainya, semakin baik keseimbangan antara ketajaman (lebar pita) dan kalibrasi.
    """
    width = up_b - low_b
    under_pen = (2.0 / alpha) * (low_b - y_true) * (y_true < low_b).astype(float)
    over_pen  = (2.0 / alpha) * (y_true - up_b) * (y_true > up_b).astype(float)
    return float(np.mean(width + under_pen + over_pen))


def evaluate_multilevel_calibration_stage2(
    y_val: np.ndarray,
    pred_val: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    model_name: str,
    levels: tuple = (0.50, 0.80, 0.90, 0.95)
) -> pd.DataFrame:
    """
    Mengevaluasi kurva kalibrasi interval prediksi pada beberapa tingkat nominal (50%, 80%, 90%, 95%)
    dengan membandingkan:
      1. Parametrik Gaussian Homoskedastik (sigma_val = RMSE_val)
      2. Non-Parametrik Split Conformal Prediction (kuantil empiris terkoreksi sampel-hingga |(y - y_hat)_val|)
    """
    n_val = len(y_val)
    n_test = len(y_test)
    abs_res_val = np.sort(np.abs(y_val - pred_val))
    sigma_val = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))

    records = []
    for c_nom in levels:
        alpha = 1.0 - c_nom
        se_iid = float(np.sqrt(c_nom * (1.0 - c_nom) / n_test) * 100.0)

        # --- METODE 1: Parametrik Gaussian Homoskedastik ---
        z_crit = float(norm.ppf(1.0 - alpha / 2.0))
        hw_gauss = z_crit * sigma_val
        low_g, up_g = pred_test - hw_gauss, pred_test + hw_gauss
        diag_g = evaluate_picp_stage1(y_test, pred_test, sigma_val, nominal_level=c_nom)
        is_gauss = compute_winkler_interval_score(y_test, low_g, up_g, alpha=alpha)

        # --- METODE 2: Split Conformal Prediction (Lei et al., 2018) ---
        k_idx = int(min(n_val, np.ceil((n_val + 1) * c_nom))) - 1  # indeks 0-based
        q_conf = float(abs_res_val[k_idx])
        low_c, up_c = pred_test - q_conf, pred_test + q_conf
        hits_c = int(np.sum((y_test >= low_c) & (y_test <= up_c)))
        picp_c = float(hits_c / n_test)
        # Gunakan rasio q_conf / z_crit sebagai sigma ekuivalen agar bisa diuji Kupiec POF lewat fungsi Tahap 1
        diag_c = evaluate_picp_stage1(y_test, pred_test, q_conf / z_crit, nominal_level=c_nom)
        is_conf = compute_winkler_interval_score(y_test, low_c, up_c, alpha=alpha)

        records.append({
            "Model": model_name,
            "Target_Nominal (%)": int(round(c_nom * 100)),
            "SE_iid (pp)": round(se_iid, 2),
            # Hasil Gaussian
            "Gauss_HalfWidth (±)": round(hw_gauss, 4),
            "Gauss_MPIW": round(2.0 * hw_gauss, 4),
            "Gauss_Hit": diag_g["N_Hit"],
            "Gauss_PICP (%)": diag_g["PICP_80 (%)"],
            "Gauss_Kupiec_p": diag_g["Kupiec_p"],
            "Gauss_Winkler_IS": round(is_gauss, 4),
            # Hasil Split Conformal
            "Conf_Quantile_q (±)": round(q_conf, 4),
            "Conf_MPIW": round(2.0 * q_conf, 4),
            "Conf_Hit": f"{hits_c}/{n_test}",
            "Conf_PICP (%)": round(picp_c * 100.0, 2),
            "Conf_Kupiec_p": diag_c["Kupiec_p"],
            "Conf_Winkler_IS": round(is_conf, 4),
        })

    return pd.DataFrame(records)


def evaluate_tail_and_brier_stage3_4(
    y_val: np.ndarray,
    pred_val: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    model_name: str
) -> dict:
    """
    Melakukan audit mendalam untuk Tahap 3 & Tahap 4:
      1. Rasio inflasi RMSE_test / RMSE_val (indikator post-selection optimism / pergeseran rezim).
      2. Uji Shapiro-Wilk, Skewness, dan Excess Kurtosis pada residual validasi & uji.
      3. Proporsi pelanggaran ekor bawah (y < L_80, target 10%) vs ekor atas (y > U_80, target 10%).
      4. Cakupan PICP_80 khusus pada bulan-bulan kekeringan aktual (y_test <= -1.0).
      5. Brier Score (BS) dan Brier Skill Score (BSS) untuk P(Y <= -1.0) dan P(Y <= -1.5),
         membandingkan probabilitas Gaussian vs Distribusi Empiris Residual Validasi.
    """
    res_val = y_val - pred_val
    res_test = y_test - pred_test

    sigma_val = float(np.sqrt(np.mean(res_val ** 2)))
    sigma_test = float(np.sqrt(np.mean(res_test ** 2)))
    sigma_ratio = float(sigma_test / max(1e-12, sigma_val))

    # 1. Uji Normalitas Shapiro-Wilk & Momen Orde Tinggi pada Residual Uji
    shapiro_w, shapiro_p = shapiro(res_test)
    sk_test = float(skew(res_test, bias=False))
    kurt_test = float(kurtosis(res_test, fisher=True, bias=False))

    # 2. Asimetri Pelanggaran Interval 80% (Target nominal masing-masing ekor = 10.0%)
    z80 = float(norm.ppf(0.90))
    low80 = pred_test - z80 * sigma_val
    up80 = pred_test + z80 * sigma_val

    viol_low_pct = float(np.mean(y_test < low80) * 100.0)
    viol_up_pct = float(np.mean(y_test > up80) * 100.0)

    # 3. Cakupan Bersyarat Khusus Bulan Kering Aktual (y_test <= -1.0)
    dry_mask = y_test <= -1.0
    n_dry = int(np.sum(dry_mask))
    if n_dry > 0:
        dry_hits = int(np.sum(
            (y_test[dry_mask] >= low80[dry_mask])
            & (y_test[dry_mask] <= up80[dry_mask])
        ))
        picp80_dry_pct = float(dry_hits / n_dry * 100.0)
        dry_hit_str = f"{dry_hits}/{n_dry} ({picp80_dry_pct:.1f}%)"
    else:
        dry_hit_str = "0/0"

    # 4. Brier Score (BS) & Brier Skill Score (BSS) pada Ambang tau = -1.0 dan tau = -1.5
    brier_metrics = {}
    for tau, lbl in [(-1.0, "Mod10"), (-1.5, "Sev15")]:
        obs_event = (y_test <= tau).astype(float)
        p_clim = float(np.mean(obs_event))
        bs_clim = float(np.mean((p_clim - obs_event) ** 2))

        # Probabilitas Gaussian
        p_gauss = norm.cdf((tau - pred_test) / sigma_val)
        bs_gauss = float(np.mean((p_gauss - obs_event) ** 2))
        bss_gauss = (
            float((1.0 - bs_gauss / max(1e-12, bs_clim)) * 100.0)
            if bs_clim > 0 else 0.0
        )

        # Probabilitas Empiris Residual Validasi (Bebas Asumsi Normal)
        p_emp = np.array([np.mean(res_val <= (tau - yh)) for yh in pred_test])
        bs_emp = float(np.mean((p_emp - obs_event) ** 2))
        bss_emp = (
            float((1.0 - bs_emp / max(1e-12, bs_clim)) * 100.0)
            if bs_clim > 0 else 0.0
        )

        brier_metrics[f"N_Event_{lbl}"] = int(np.sum(obs_event))
        brier_metrics[f"BS_Gauss_{lbl}"] = round(bs_gauss, 4)
        brier_metrics[f"BSS_Gauss_{lbl} (%)"] = round(bss_gauss, 2)
        brier_metrics[f"BS_Emp_{lbl}"] = round(bs_emp, 4)
        brier_metrics[f"BSS_Emp_{lbl} (%)"] = round(bss_emp, 2)

    return {
        "Model": model_name,
        "Sigma_Val": round(sigma_val, 4),
        "Sigma_Test": round(sigma_test, 4),
        "Ratio_Test_Val": round(sigma_ratio, 3),
        "Skew_Test": round(sk_test, 3),
        "ExKurt_Test": round(kurt_test, 3),
        "Shapiro_W": round(float(shapiro_w), 4),
        "Shapiro_p": round(float(shapiro_p), 4),
        "Viol_Low10 (%)": round(viol_low_pct, 2),
        "Viol_Up10 (%)": round(viol_up_pct, 2),
        "PICP80_DryMonths": dry_hit_str,
        **brier_metrics
    }


def compute_contingency_metrics(obs_dry: np.ndarray, alarm_triggered: np.ndarray) -> dict:
    """
    Menghitung metrik verifikasi peringatan dini dari matriks kontingensi 2x2:
      - Hits (TP), False Alarms (FP), Misses (FN), Correct Negatives (TN)
      - POD / Hit Rate = TP / (TP + FN)
      - FAR (False Alarm Ratio) = FP / (TP + FP)
      - POFD (Probability of False Detection) = FP / (FP + TN)
      - CSI (Critical Success Index / Threat Score) = TP / (TP + FP + FN)
      - F1-Score = 2*TP / (2*TP + FP + FN)
    """
    tp = int(np.sum((obs_dry == 1) & (alarm_triggered == 1)))
    fp = int(np.sum((obs_dry == 0) & (alarm_triggered == 1)))
    fn = int(np.sum((obs_dry == 1) & (alarm_triggered == 0)))
    tn = int(np.sum((obs_dry == 0) & (alarm_triggered == 0)))

    pod = float(tp / max(1, tp + fn)) * 100.0
    far = float(fp / max(1, tp + fp)) * 100.0
    pofd = float(fp / max(1, fp + tn)) * 100.0
    csi = float(tp / max(1, tp + fp + fn)) * 100.0
    f1 = float(2.0 * tp / max(1, 2 * tp + fp + fn)) * 100.0

    return {
        "TP_Hit": tp,
        "FP_FalseAlarm": fp,
        "FN_Miss": fn,
        "TN_CorrectNeg": tn,
        "POD_HitRate (%)": round(pod, 2),
        "FAR_FalseAlarmRatio (%)": round(far, 2),
        "POFD_FalseDetRate (%)": round(pofd, 2),
        "CSI_ThreatScore (%)": round(csi, 2),
        "F1_Score (%)": round(f1, 2),
    }


def evaluate_dews_contingency_stage5(
    y_val: np.ndarray,
    pred_val: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    model_name: str,
    tau_mod: float = -1.0
) -> pd.DataFrame:
    """
    Membandingkan kinerja operasional 3 rezim pemicu peringatan terhadap kejadian aktual (y_test <= tau_mod):
      1. Pemicu Probabilistik WASPADA+SIAGA: P(Y <= -1.0) >= 30% (Cost-Loss C/L = 0.30)
      2. Pemicu Probabilistik SIAGA (Ketat) : P(Y <= -1.0) >= 60% (Cost-Loss C/L = 0.60)
      3. Pemicu Titik Deterministik (Lama)  : y_hat <= -1.0 (setara P >= 50%)
    """
    sigma_val = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))
    prob_mod = norm.cdf((tau_mod - pred_test) / sigma_val) * 100.0
    obs_dry = (y_test <= tau_mod).astype(int)

    rules = [
        ("1_Prob_Watch_Alert (P >= 30%)", (prob_mod >= 30.0).astype(int)),
        ("2_Prob_Point_50 (y_hat <= -1.0 / P >= 50%)", (pred_test <= tau_mod).astype(int)),
        ("3_Prob_Strict_Alert (P >= 60%)", (prob_mod >= 60.0).astype(int)),
    ]

    rows = []
    for rule_name, trig in rules:
        m_dict = compute_contingency_metrics(obs_dry, trig)
        rows.append({
            "Model": model_name,
            "Aturan_Pemicu": rule_name,
            **m_dict
        })
    return pd.DataFrame(rows)


def get_tuned_builders(sp_col: str = "SPEI_06", feat_scheme: str = "Lag+Meteo"):
    """Mengembalikan konfigurasi hiperparameter hasil 3-Fold CV (Tabel Suplemen S4)."""
    if sp_col == "SPEI_06" and feat_scheme == "Lag+Meteo":
        return {
            "Ridge_Direct": (6, "Direct", lambda: Ridge(alpha=1.0)),
            "Ridge_Delta":  (2, "Delta",  lambda: Ridge(alpha=50.0)),
            "MLP_Direct":   (3, "Direct", lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=0.1, max_iter=500, random_state=42))),
            "MLP_Delta":    (6, "Delta",  lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=500, random_state=42))),
            "RF_Direct":    (3, "Direct", lambda: RandomForestRegressor(n_estimators=150, max_depth=3, max_features=0.75, random_state=42, n_jobs=-1)),
            "RF_Delta":     (6, "Delta",  lambda: RandomForestRegressor(n_estimators=150, max_depth=3, max_features=0.50, random_state=42, n_jobs=-1)),
            "XGB_Direct":   (3, "Direct", lambda: XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.04, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=1.5, random_state=42, n_jobs=-1)),
            "XGB_Delta":    (6, "Delta",  lambda: XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=3.0, random_state=42, n_jobs=-1)),
            "Cat_Direct":   (2, "Direct", lambda: CatBoostRegressor(iterations=120, depth=4, learning_rate=0.04, l2_leaf_reg=3.0, random_state=42, verbose=0)),
            "Cat_Delta":    (6, "Delta",  lambda: CatBoostRegressor(iterations=120, depth=4, learning_rate=0.03, l2_leaf_reg=5.0, random_state=42, verbose=0)),
        }
    elif sp_col == "SPEI_06" and feat_scheme == "Lag-Only":
        return {
            "Ridge_Direct": (3, "Direct", lambda: Ridge(alpha=10.0)),
            "Ridge_Delta":  (3, "Delta",  lambda: Ridge(alpha=50.0)),
            "MLP_Direct":   (2, "Direct", lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.1, max_iter=500, random_state=42))),
            "MLP_Delta":    (3, "Delta",  lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=500, random_state=42))),
            "RF_Direct":    (3, "Direct", lambda: RandomForestRegressor(n_estimators=150, max_depth=3, max_features=0.75, random_state=42, n_jobs=-1)),
            "RF_Delta":     (6, "Delta",  lambda: RandomForestRegressor(n_estimators=150, max_depth=3, max_features=0.50, random_state=42, n_jobs=-1)),
            "XGB_Direct":   (2, "Direct", lambda: XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.04, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=1.5, random_state=42, n_jobs=-1)),
            "XGB_Delta":    (6, "Delta",  lambda: XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=3.0, random_state=42, n_jobs=-1)),
            "Cat_Direct":   (2, "Direct", lambda: CatBoostRegressor(iterations=120, depth=3, learning_rate=0.04, l2_leaf_reg=5.0, random_state=42, verbose=0)),
            "Cat_Delta":    (6, "Delta",  lambda: CatBoostRegressor(iterations=120, depth=4, learning_rate=0.03, l2_leaf_reg=5.0, random_state=42, verbose=0)),
        }
    else:
        return {
            "Ridge_Direct": (3, "Direct", lambda: Ridge(alpha=1.0)),
            "Ridge_Delta":  (3, "Delta",  lambda: Ridge(alpha=1.0)),
            "MLP_Direct":   (6, "Direct", lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=500, random_state=42))),
            "MLP_Delta":    (3, "Delta",  lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=500, random_state=42))),
            "RF_Direct":    (2, "Direct", lambda: RandomForestRegressor(n_estimators=150, max_depth=3, max_features=0.50, random_state=42, n_jobs=-1)),
            "RF_Delta":     (3, "Delta",  lambda: RandomForestRegressor(n_estimators=150, max_depth=3, max_features=0.50, random_state=42, n_jobs=-1)),
            "XGB_Direct":   (2, "Direct", lambda: XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=1.5, random_state=42, n_jobs=-1)),
            "XGB_Delta":    (3, "Delta",  lambda: XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=3.0, random_state=42, n_jobs=-1)),
            "Cat_Direct":   (2, "Direct", lambda: CatBoostRegressor(iterations=120, depth=3, learning_rate=0.04, l2_leaf_reg=3.0, random_state=42, verbose=0)),
            "Cat_Delta":    (3, "Delta",  lambda: CatBoostRegressor(iterations=120, depth=3, learning_rate=0.03, l2_leaf_reg=5.0, random_state=42, verbose=0)),
        }


def run_step3_evaluation(
    df_master: pd.DataFrame,
    sp_col: str = "SPEI_06",
    feat_scheme: str = "Lag+Meteo",
    horizon: int = 1,
    include_sarima: bool = True
):
    """
    Melatih seluruh model secara dinamis di atas df_master dan menghitung metrik evaluasi
    serta tabel buletin peringatan dini.
    """
    d_base, latest_row, _, _ = engineer_features(df_master, sp_col=sp_col, w=6, horizon=horizon)
    tr_sl, va_sl, te_sl, split_meta = get_dynamic_splits(len(d_base), gap=5, horizon=horizon)

    y_tr = d_base.iloc[tr_sl]["Y_TARGET"].values
    y_va = d_base.iloc[va_sl]["Y_TARGET"].values
    y_te = d_base.iloc[te_sl]["Y_TARGET"].values

    sp_va = d_base.iloc[va_sl][sp_col].values
    sp_te = d_base.iloc[te_sl][sp_col].values
    dates_te = d_base.iloc[te_sl]["DATE"].values

    val_preds = {}
    test_preds = {}
    future_preds = {}

    # B0: Climatology
    clim_map = d_base.iloc[tr_sl].groupby("MONTH")["Y_TARGET"].mean().to_dict()
    val_preds["B0_Climatology"] = d_base.iloc[va_sl]["MONTH"].map(clim_map).values
    test_preds["B0_Climatology"] = d_base.iloc[te_sl]["MONTH"].map(clim_map).values
    future_preds["B0_Climatology"] = float(latest_row["MONTH"].map(clim_map).iloc[0])

    # B1: Persistence
    val_preds["B1_Persistence"] = sp_va.copy()
    test_preds["B1_Persistence"] = sp_te.copy()
    future_preds["B1_Persistence"] = float(latest_row[sp_col].iloc[0])

    # B2: AR(2) Linear Ridge
    ar2_mod = Ridge(alpha=1.0).fit(d_base.iloc[tr_sl][[sp_col, "lag1_ar"]], y_tr)
    val_preds["B2_AR(2)"] = ar2_mod.predict(d_base.iloc[va_sl][[sp_col, "lag1_ar"]])
    test_preds["B2_AR(2)"] = ar2_mod.predict(d_base.iloc[te_sl][[sp_col, "lag1_ar"]])
    future_preds["B2_AR(2)"] = float(ar2_mod.predict(latest_row[[sp_col, "lag1_ar"]])[0])

    # B3: SARIMA(2,0,0)(1,0,0)_12
    if include_sarima:
        try:
            sar_mod = SARIMAX(
                d_base[sp_col].iloc[tr_sl].values,
                order=(2, 0, 0),
                seasonal_order=(1, 0, 0, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            sar_res = sar_mod.apply(d_base[sp_col].values)
            va_indices = range(va_sl.start, va_sl.stop)
            te_indices = range(te_sl.start, te_sl.stop)
            val_preds["B3_SARIMA"] = np.array([
                sar_res.get_prediction(start=idx + 1, end=idx + horizon, dynamic=0).predicted_mean[-1]
                for idx in va_indices
            ])
            test_preds["B3_SARIMA"] = np.array([
                sar_res.get_prediction(start=idx + 1, end=idx + horizon, dynamic=0).predicted_mean[-1]
                for idx in te_indices
            ])
            future_preds["B3_SARIMA"] = float(test_preds["B3_SARIMA"][-1])
        except Exception:
            pass

    # 10 Model Machine Learning (5 keluarga x Direct/Delta)
    builders = get_tuned_builders(sp_col=sp_col, feat_scheme=feat_scheme)
    for m_name, (w_opt, mode, builder_fn) in builders.items():
        d_w, latest_w, l_cols, m_cols = engineer_features(df_master, sp_col=sp_col, w=w_opt, horizon=horizon)
        sp_tr_w = d_w.iloc[tr_sl][sp_col].values
        sp_va_w = d_w.iloc[va_sl][sp_col].values
        sp_te_w = d_w.iloc[te_sl][sp_col].values
        sp_lat_w = float(latest_w[sp_col].iloc[0])

        t_fit = y_tr if mode == "Direct" else (y_tr - sp_tr_w)
        use_cols = select_k12_mi_features(d_w, tr_sl, m_cols, t_fit, sp_col) if feat_scheme == "Lag+Meteo" else l_cols

        reg = builder_fn()
        reg.fit(d_w.iloc[tr_sl][use_cols], t_fit)

        p_va = reg.predict(d_w.iloc[va_sl][use_cols])
        p_te = reg.predict(d_w.iloc[te_sl][use_cols])
        p_lat = float(reg.predict(latest_w[use_cols])[0])

        if mode == "Delta":
            p_va = sp_va_w + p_va
            p_te = sp_te_w + p_te
            p_lat = sp_lat_w + p_lat

        val_preds[m_name] = p_va
        test_preds[m_name] = p_te
        future_preds[m_name] = p_lat

    # Hitung Metrik & Uji DM-HAC + Koreksi Holm
    mse_pers = float(np.mean((y_te - test_preds["B1_Persistence"]) ** 2))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    q_lag = max(5, horizon)

    model_order = [m for m in [
        "B0_Climatology", "B1_Persistence", "B2_AR(2)", "B3_SARIMA",
        "Ridge_Direct", "Ridge_Delta", "MLP_Direct", "MLP_Delta",
        "RF_Direct", "RF_Delta", "XGB_Direct", "XGB_Delta", "Cat_Direct", "Cat_Delta"
    ] if m in test_preds]

    rows = []
    for m_name in model_order:
        p_va = val_preds[m_name]
        p_te = test_preds[m_name]
        rmse_va = float(np.sqrt(np.mean((y_va - p_va) ** 2)))
        rmse_te = float(np.sqrt(np.mean((y_te - p_te) ** 2)))
        bias_te = float(np.mean(p_te - y_te))
        mse_te = float(np.mean((y_te - p_te) ** 2))
        r2_te = float(1.0 - np.sum((y_te - p_te) ** 2) / ss_tot)
        ss_pers = float((1.0 - mse_te / mse_pers) * 100.0)

        dm_p, pr_p = diebold_mariano_hac(y_te, p_te, test_preds["B1_Persistence"], max_lag=q_lag)
        dm_a, pr_a = diebold_mariano_hac(y_te, p_te, test_preds["B2_AR(2)"], max_lag=q_lag)
        dm_rdl, pr_rdl = diebold_mariano_hac(y_te, p_te, test_preds["Ridge_Delta"], max_lag=q_lag)
        dm_rdr, pr_rdr = diebold_mariano_hac(y_te, p_te, test_preds["Ridge_Direct"], max_lag=q_lag)

        picp_diag = evaluate_picp_stage1(y_te, p_te, rmse_va, nominal_level=0.80)

        rows.append({
            "Model": m_name,
            "Val_RMSE": round(rmse_va, 4),
            "Test_RMSE": round(rmse_te, 4),
            "Test_Bias": round(bias_te, 4),
            "Test_R2": round(r2_te, 4),
            "SS_vs_Pers (%)": round(ss_pers, 2),
            "DM_vs_Pers": round(dm_p, 2),
            "p_raw_Pers": pr_p,
            "DM_vs_AR2": round(dm_a, 2),
            "p_raw_AR2": pr_a,
            "DM_vs_RidgeDelta": round(dm_rdl, 2),
            "p_raw_RidgeDelta": pr_rdl,
            "DM_vs_RidgeDirect": round(dm_rdr, 2),
            "p_raw_RidgeDirect": pr_rdr,
            "N_Hit_80": picp_diag["N_Hit"],
            "PICP_80 (%)": picp_diag["PICP_80 (%)"],
            "SE_iid (pp)": picp_diag["SE_iid (pp)"],
            "SE_corr (pp)": picp_diag["SE_corr (pp)"],
            "Kupiec_p": picp_diag["Kupiec_p"],
            "Next_Forecast_t_plus_h": round(future_preds[m_name], 4)
        })

    df_eval = pd.DataFrame(rows)

    # Koreksi Holm (m = 13 non-referensi) untuk masing-masing baseline
    for ref_name, p_col, holm_col in [
        ("B1_Persistence", "p_raw_Pers", "p_Holm13_Pers"),
        ("B2_AR(2)", "p_raw_AR2", "p_Holm13_AR2"),
        ("Ridge_Delta", "p_raw_RidgeDelta", "p_Holm13_RidgeDelta"),
        ("Ridge_Direct", "p_raw_RidgeDirect", "p_Holm13_RidgeDirect")
    ]:
        mask = df_eval["Model"] != ref_name
        _, p_holm, _, _ = multipletests(df_eval.loc[mask, p_col].values, method="holm")
        df_eval[holm_col] = 0.0
        df_eval.loc[mask, holm_col] = np.round(p_holm, 4)
        df_eval[p_col] = np.round(df_eval[p_col], 4)

    # Susun DataFrame Prediksi Periode Uji
    df_preds = pd.DataFrame({
        "DATE_CURRENT_t": pd.to_datetime(dates_te),
        "DATE_TARGET": pd.to_datetime(dates_te) + pd.DateOffset(months=horizon),
        "SPEI_t_Anchor": np.round(sp_te, 4),
        "Y_True": np.round(y_te, 4)
    })
    for m_name in model_order:
        df_preds[f"Pred_{m_name}"] = np.round(test_preds[m_name], 4)

    # Bangun dan simpan kurva kalibrasi multi-kuantil untuk seluruh model yang dievaluasi.
    multilevel_dfs = []
    for m_name in model_order:
        df_ml = evaluate_multilevel_calibration_stage2(
            y_val=y_va,
            pred_val=val_preds[m_name],
            y_test=y_te,
            pred_test=test_preds[m_name],
            model_name=m_name,
            levels=(0.50, 0.80, 0.90, 0.95)
        )
        multilevel_dfs.append(df_ml)
    df_multilevel = pd.concat(multilevel_dfs, ignore_index=True)
    os.makedirs("DATA/PROCESSED", exist_ok=True)
    df_multilevel.to_csv(
        f"DATA/PROCESSED/Kalibrasi_MultiKuantil_{sp_col}_t{horizon}.csv",
        index=False
    )

    tail_rows = []
    for m_name in model_order:
        tail_rows.append(
            evaluate_tail_and_brier_stage3_4(
                y_val=y_va,
                pred_val=val_preds[m_name],
                y_test=y_te,
                pred_test=test_preds[m_name],
                model_name=m_name
            )
        )
    df_tail_brier = pd.DataFrame(tail_rows)
    df_tail_brier.to_csv(
        f"DATA/PROCESSED/Audit_Ekor_dan_Brier_{sp_col}_t{horizon}.csv",
        index=False
    )

    dews_cont_dfs = []
    for m_name in model_order:
        dews_cont_dfs.append(
            evaluate_dews_contingency_stage5(
                y_val=y_va,
                pred_val=val_preds[m_name],
                y_test=y_te,
                pred_test=test_preds[m_name],
                model_name=m_name,
                tau_mod=-1.0
            )
        )
    df_dews_cont = pd.concat(dews_cont_dfs, ignore_index=True)
    df_dews_cont.to_csv(
        f"DATA/PROCESSED/Evaluasi_Kontingensi_DEWS_{sp_col}_t{horizon}.csv",
        index=False
    )

    return df_eval, df_preds, split_meta, latest_row


if __name__ == "__main__":
    master_path = "DATA/PROCESSED/dataset_master_nasa_csic_clean.csv"
    df_m = pd.read_csv(master_path)

    print("Menjalankan evaluasi penuh 14 model pada SPEI_06 (Lag+Meteo, t+1)...")
    df_eval, df_preds, split_meta, _ = run_step3_evaluation(
        df_m, sp_col="SPEI_06", feat_scheme="Lag+Meteo", horizon=1, include_sarima=True
    )

    os.makedirs("DATA/PROCESSED", exist_ok=True)
    eval_out = "DATA/PROCESSED/Evaluasi_14_Model_SPEI06_t1.csv"
    df_eval.to_csv(eval_out, index=False)

    # Simpan juga Buletin Peringatan Dini untuk Cat_Delta
    sigma_cat = float(df_eval.loc[df_eval["Model"] == "Cat_Delta", "Val_RMSE"].iloc[0])
    z80 = norm.ppf(0.90)
    y_hat_cat = df_preds["Pred_Cat_Delta"].values
    y_true = df_preds["Y_True"].values
    prob_mod = norm.cdf((-1.0 - y_hat_cat) / sigma_cat) * 100.0
    prob_sev = norm.cdf((-1.5 - y_hat_cat) / sigma_cat) * 100.0

    df_bulletin = pd.DataFrame({
        "DATE_TARGET_t1": df_preds["DATE_TARGET"],
        "SPEI_t_Anchor": df_preds["SPEI_t_Anchor"],
        "Y_True_SPEI06_t1": y_true,
        "Y_Pred_CatDelta_t1": y_hat_cat,
        "PICP80_Low": np.round(y_hat_cat - z80 * sigma_cat, 4),
        "PICP80_High": np.round(y_hat_cat + z80 * sigma_cat, 4),
        "Prob_Drought_Mod_Pct": np.round(prob_mod, 2),
        "Prob_Drought_Sev_Pct": np.round(prob_sev, 2),
        "Early_Warning_Status": [
            "SIAGA (MERAH)" if pm >= 60.0
            else ("WASPADA (KUNING)" if pm >= 30.0 else "NORMAL (HIJAU)")
            for pm in prob_mod
        ],
        "Actual_Drought_Event": np.where(y_true <= -1.0, "KERING (<= -1,0)", "TIDAK KERING")
    })
    bul_out = "DATA/PROCESSED/Buletin_Peringatan_Dini_SPEI06_CatDelta_V24.csv"
    df_bulletin.to_csv(bul_out, index=False)

    print("\n=== [SELESAI STEP 3] HASIL EVALUASI LOKAL vs GOOGLE COLAB (TABEL 2) ===")
    cols_show = ["Model", "Val_RMSE", "Test_RMSE", "Test_R2", "SS_vs_Pers (%)", "DM_vs_Pers", "p_Holm13_Pers", "DM_vs_RidgeDelta", "p_Holm13_RidgeDelta"]
    print(df_eval[cols_show].to_string(index=False))

    df_ml_show = pd.read_csv("DATA/PROCESSED/Kalibrasi_MultiKuantil_SPEI_06_t1.csv")
    focus_models = ["B1_Persistence", "B2_AR(2)", "Ridge_Direct", "XGB_Delta", "Cat_Delta"]
    print("\n=== [SELESAI TAHAP 2] KURVA KALIBRASI MULTI-KUANTIL (GAUSSIAN vs SPLIT CONFORMAL) ===")
    print(df_ml_show[df_ml_show["Model"].isin(focus_models)].to_string(index=False))

    df_tb_show = pd.read_csv("DATA/PROCESSED/Audit_Ekor_dan_Brier_SPEI_06_t1.csv")
    print("\n=== [SELESAI TAHAP 3 & 4] AUDIT NORMALITAS RESIDUAL, ASIMETRI EKOR, & BRIER SKILL SCORE ===")
    print(df_tb_show[df_tb_show["Model"].isin(focus_models)].to_string(index=False))

    df_cont_show = pd.read_csv("DATA/PROCESSED/Evaluasi_Kontingensi_DEWS_SPEI_06_t1.csv")
    print("\n=== [SELESAI TAHAP 5] VERIFIKASI HIT RATE (POD), FALSE ALARM (FAR), & CSI PER ATURAN PEMICU ===")
    print(df_cont_show[df_cont_show["Model"].isin(focus_models)].to_string(index=False))