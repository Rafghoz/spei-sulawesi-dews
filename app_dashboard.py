# ==============================================================================
# FILE FINAL: app_dashboard.py (TERINTEGRASI PENUH DENGAN AUDIT TAHAP 1 s.d. 5)
# Lokasi file: spei_palu_local/app_dashboard.py
# Cara menjalankan: streamlit run app_dashboard.py
# ==============================================================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

from src.step1_extract_raw import run_step1_pipeline
from src.step2_preprocess_features import reconstruct_missing_anchor
from src.step3_train_and_evaluate import run_step3_evaluation

st.set_page_config(
    page_title="DEWS Sulawesi - Audit Probabilistik & Pipa Dinamis SPEI",
    page_icon="🌤️",
    layout="wide"
)

MASTER_CSV_PATH = "DATA/PROCESSED/dataset_master_nasa_csic_clean.csv"
NASA_RAW_DIR    = "DATA/NASA POWER"
CSIC_RAW_DIR    = "DATA/CSIC Global SPEI Database"

GRID_LABELS = {
    "G1_Palu":     "G1: Lembah Palu (-0,75° LS, 119,75° BT) [Fokus Utama]",
    "G2_Parigi":   "G2: Parigi (-0,75° LS, 120,25° BT)",
    "G3_Poso":     "G3: Poso (-1,25° LS, 120,75° BT)",
    "G4_ToliToli": "G4: Toli-Toli (+1,25° LU, 120,75° BT)",
    "G5_Makassar": "G5: Makassar (-5,25° LS, 119,25° BT)",
}


def get_data_signature():
    """Mendeteksi perubahan berkas di folder DATA/ untuk memicu pembaruan otomatis."""
    tracked_files = []
    if os.path.exists(NASA_RAW_DIR):
        tracked_files.extend(glob.glob(os.path.join(NASA_RAW_DIR, "*.csv")))
    if os.path.exists(CSIC_RAW_DIR):
        tracked_files.extend(glob.glob(os.path.join(CSIC_RAW_DIR, "*.nc")))
    if os.path.exists(MASTER_CSV_PATH):
        tracked_files.append(MASTER_CSV_PATH)
    sig = []
    for fpath in sorted(tracked_files):
        try:
            st_info = os.stat(fpath)
            sig.append((fpath, round(st_info.st_mtime, 2), st_info.st_size))
        except OSError:
            pass
    return tuple(sig)


@st.cache_data(show_spinner="Menjalankan pipa terstruktur (Step 1–3) & audit probabilistik (Tahap 1–5)...")
def load_and_run_pipeline(
    signature,
    grid_key: str,
    base_scale: str,
    feat_scheme: str,
    horizon: int,
    include_sarima: bool
):
    if not os.path.exists(MASTER_CSV_PATH):
        df_master = run_step1_pipeline(NASA_RAW_DIR, CSIC_RAW_DIR, "DATA/PROCESSED")
    else:
        df_master = pd.read_csv(MASTER_CSV_PATH)
        df_master["DATE"] = pd.to_datetime(df_master["DATE"])

    df_work = df_master.copy()
    if grid_key != "G1_Palu":
        col_grid = f"{base_scale}_{grid_key}"
        if col_grid in df_work.columns:
            df_work[base_scale] = df_work[col_grid]
            df_work[f"TARGET_{base_scale}_t1"] = df_work[base_scale].shift(-1)

    df_eval, df_preds, split_meta, latest_row = run_step3_evaluation(
        df_master=df_work,
        sp_col=base_scale,
        feat_scheme=feat_scheme,
        horizon=horizon,
        include_sarima=include_sarima
    )
    return df_work, df_eval, df_preds, split_meta, latest_row


def build_dews_bulletin(
    df_preds: pd.DataFrame,
    model_name: str,
    half_width_pi: float,
    sigma_val: float,
    tau_mod: float = -1.0,
    tau_sev: float = -1.5,
    alert_prob_thr: float = 50.0,
    watch_prob_thr: float = 30.0
) -> pd.DataFrame:
    """
    Menghitung metrik probabilistik dan status peringatan dini MURNI BERBASIS PROBABILITAS
    Cost-Loss Ratio (tanpa redundansi operator 'ATAU' pada prediksi titik):
      - SIAGA (MERAH)   : P(SPEI <= tau_mod) >= alert_prob_thr (default 50%)
      - WASPADA (KUNING): watch_prob_thr <= P(SPEI <= tau_mod) < alert_prob_thr (default 30% <= P < 50%)
      - NORMAL (HIJAU)  : P(SPEI <= tau_mod) < watch_prob_thr (default < 30%)
    """
    y_hat = df_preds[f"Pred_{model_name}"].values
    y_true = df_preds["Y_True"].values

    ci_low = y_hat - half_width_pi
    ci_high = y_hat + half_width_pi
    prob_mod = norm.cdf((tau_mod - y_hat) / sigma_val) * 100.0
    prob_sev = norm.cdf((tau_sev - y_hat) / sigma_val) * 100.0

    statuses = []
    sub_levels = []
    for pm in prob_mod:
        if pm >= alert_prob_thr:
            statuses.append("SIAGA (MERAH)")
            sub_levels.append("SIAGA TINGGI (P ≥ 60%)" if pm >= 60.0 else f"SIAGA ({alert_prob_thr:.0f}% ≤ P < 60%)")
        elif pm >= watch_prob_thr:
            statuses.append("WASPADA (KUNING)")
            sub_levels.append(f"WASPADA ({watch_prob_thr:.0f}% ≤ P < {alert_prob_thr:.0f}%)")
        else:
            statuses.append("NORMAL (HIJAU)")
            sub_levels.append(f"NORMAL (P < {watch_prob_thr:.0f}%)")

    return pd.DataFrame({
        "DATE_TARGET": pd.to_datetime(df_preds["DATE_TARGET"]),
        "SPEI_t_Anchor": np.round(df_preds["SPEI_t_Anchor"], 4),
        "Y_True": np.round(y_true, 4),
        "Y_Pred": np.round(y_hat, 4),
        "PI_Low": np.round(ci_low, 4),
        "PI_High": np.round(ci_high, 4),
        "Prob_Mod_Pct": np.round(prob_mod, 2),
        "Prob_Sev_Pct": np.round(prob_sev, 2),
        "Status_DEWS": statuses,
        "Detail_SubLevel": sub_levels,
        "Aktual_Kering": np.where(y_true <= tau_mod, "KERING", "NORMAL")
    })


# ------------------------------------------------------------------------------
# JUDUL & SIDEBAR PENGATURAN
# ------------------------------------------------------------------------------
st.title("🌤️ Sistem Peringatan Dini Kekeringan (DEWS) & Audit Probabilistik SPEI Sulawesi")
st.caption(
    "Penulis: Rafli Ghozali, I Ketut Divta Suryawan, Akbar Tri Wicaksono (Universitas Tadulako) — "
    "Pipa Lokal Terstruktur dengan Audit Kupiec POF, Split Conformal, Shapiro-Wilk, Brier Skill Score, & Matriks Kontingensi Cost-Loss"
)

with st.sidebar:
    st.header("🔄 1. Sinkronisasi Data Mentah")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Cek CSV", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_btn2:
        if st.button("⚙️ Ekstrak Mentah", use_container_width=True):
            with st.spinner("Mengekstrak ulang file .nc & CSV harian NASA..."):
                run_step1_pipeline(NASA_RAW_DIR, CSIC_RAW_DIR, "DATA/PROCESSED")
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.header("🎛️ 2. Filter Wilayah, Indeks, & Model")
    grid_sel = st.selectbox(
        "Pilih Sel Grid Sulawesi:",
        options=list(GRID_LABELS.keys()),
        format_func=lambda k: GRID_LABELS[k],
        index=0
    )
    sp_sel = st.radio("Pilih Skala Akumulasi SPEI:", ["SPEI_06", "SPEI_03"], index=0, horizontal=True)

    if grid_sel != "G1_Palu":
        st.info("ℹ️ Sel grid G2–G5 menggunakan skema univariat Lag-Only.")
        feat_sel = "Lag-Only"
    else:
        feat_sel = st.radio("Skema Himpunan Fitur:", ["Lag+Meteo", "Lag-Only"], index=0, horizontal=True)

    horizon_sel = st.selectbox(
        "Horizon Peramalan (t+h):",
        options=[1, 2, 3, 6],
        format_func=lambda h: f"t+{h} Bulan ke Depan",
        index=0
    )
    inc_sarima = st.checkbox("Sertakan Model B3_SARIMA", value=(horizon_sel == 1))

    sig_current = get_data_signature()
    df_work, df_eval, df_preds, split_meta, latest_row = load_and_run_pipeline(
        sig_current, grid_sel, sp_sel, feat_sel, horizon_sel, inc_sarima
    )

    model_options = df_eval["Model"].tolist()
    def_idx = model_options.index("Cat_Delta") if "Cat_Delta" in model_options else 0
    model_sel = st.selectbox("Pilih Model Fokus Peringatan Dini:", model_options, index=def_idx)
    tau_in = st.slider("Ambang Kekeringan Sedang (τ):", min_value=-2.0, max_value=-0.5, value=-1.0, step=0.05)

    st.markdown("---")
    st.header("📐 3. Metode Interval & Ambang Cost-Loss")
    pi_method = st.radio(
        "Metode Interval Prediksi:",
        ["Parametrik Gaussian", "Split Conformal (Non-Parametrik)"],
        index=0
    )
    nom_level_sel = st.select_slider(
        "Tingkat Cakupan Nominal (1-α):",
        options=[50, 80, 90, 95],
        value=80
    )
    watch_thr = st.slider("Ambang Probabilitas WASPADA (C/L Watch %):", 15.0, 45.0, 30.0, 5.0)
    alert_thr = st.slider("Ambang Probabilitas SIAGA (C/L Alert %):", 45.0, 75.0, 50.0, 5.0)

    sigma_model = float(df_eval.loc[df_eval["Model"] == model_sel, "Val_RMSE"].iloc[0])
    z_nom = float(norm.ppf(1.0 - (1.0 - nom_level_sel / 100.0) / 2.0))
    hw_selected = z_nom * sigma_model

    # Jika pengguna memilih Split Conformal, ambil kuantil empiris q_{1-alpha} dari berkas Tahap 2
    ml_csv_path = f"DATA/PROCESSED/Kalibrasi_MultiKuantil_{sp_sel}_t{horizon_sel}.csv"
    if pi_method == "Split Conformal (Non-Parametrik)" and os.path.exists(ml_csv_path):
        df_ml_tmp = pd.read_csv(ml_csv_path)
        sub_q = df_ml_tmp[(df_ml_tmp["Model"] == model_sel) & (df_ml_tmp["Target_Nominal (%)"] == nom_level_sel)]
        if len(sub_q) > 0:
            hw_selected = float(sub_q["Conf_Quantile_q (±)"].iloc[0])

    df_dews_all = build_dews_bulletin(
        df_preds=df_preds,
        model_name=model_sel,
        half_width_pi=hw_selected,
        sigma_val=sigma_model,
        tau_mod=tau_in,
        alert_prob_thr=alert_thr,
        watch_prob_thr=watch_thr
    )

    min_date = df_dews_all["DATE_TARGET"].min().to_pydatetime()
    max_date = df_dews_all["DATE_TARGET"].max().to_pydatetime()
    date_rng = st.date_input(
        "Rentang Tanggal Evaluasi Uji:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    status_choices = ["NORMAL (HIJAU)", "WASPADA (KUNING)", "SIAGA (MERAH)"]
    status_sel = st.multiselect("Filter Status Peringatan:", options=status_choices, default=status_choices)

# Terapkan filter tanggal dan status
df_dews_filtered = df_dews_all.copy()
if isinstance(date_rng, (list, tuple)) and len(date_rng) == 2:
    d0, d1 = pd.to_datetime(date_rng[0]), pd.to_datetime(date_rng[1])
    df_dews_filtered = df_dews_filtered[(df_dews_filtered["DATE_TARGET"] >= d0) & (df_dews_filtered["DATE_TARGET"] <= d1)]
df_dews_filtered = df_dews_filtered[df_dews_filtered["Status_DEWS"].isin(status_sel)].reset_index(drop=True)

# ------------------------------------------------------------------------------
# RINGKASAN METRIK UTAMA DI BAGIAN ATAS
# ------------------------------------------------------------------------------
row_m = df_eval[df_eval["Model"] == model_sel].iloc[0]
next_y_hat = float(row_m["Next_Forecast_t_plus_h"])
next_prob_mod = float(norm.cdf((tau_in - next_y_hat) / sigma_model) * 100.0)

hits_mask = (df_dews_all["Y_True"] >= df_dews_all["PI_Low"]) & (df_dews_all["Y_True"] <= df_dews_all["PI_High"])
n_hits_curr = int(np.sum(hits_mask))
n_total_curr = len(df_dews_all)
picp_actual = float(n_hits_curr / max(1, n_total_curr) * 100.0)
se_iid_curr = float(np.sqrt((nom_level_sel / 100.0) * (1.0 - nom_level_sel / 100.0) / max(1, n_total_curr)) * 100.0)

# Hitung POD & CSI secara real-time untuk model & ambang yang dipilih
obs_dry_vec = (df_dews_all["Y_True"] <= tau_in).astype(int)
trig_watch_vec = (df_dews_all["Prob_Mod_Pct"] >= watch_thr).astype(int)
tp_w = int(np.sum((obs_dry_vec == 1) & (trig_watch_vec == 1)))
fp_w = int(np.sum((obs_dry_vec == 0) & (trig_watch_vec == 1)))
fn_w = int(np.sum((obs_dry_vec == 1) & (trig_watch_vec == 0)))
pod_w = tp_w / max(1, tp_w + fn_w) * 100.0
csi_w = tp_w / max(1, tp_w + fp_w + fn_w) * 100.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(
    "Total Data & N_eff",
    f"{len(df_work)} Bln (N_eff={split_meta['n_eff']})",
    f"Latih={split_meta['n_train']} | Val={split_meta['n_val']} | Uji={split_meta['n_test']}"
)
m2.metric(
    f"Test RMSE ({model_sel})",
    f"{row_m['Test_RMSE']:.4f}",
    f"Val RMSE (σ) = {row_m['Val_RMSE']:.4f}"
)
m3.metric(
    f"PICP {nom_level_sel}% ({pi_method.split()[0]})",
    f"{picp_actual:.1f}% ({n_hits_curr}/{n_total_curr})",
    f"Target {nom_level_sel}% (SE_iid = ±{se_iid_curr:.2f} pp)"
)
m4.metric(
    f"Hit Rate / POD (P ≥ {watch_thr:.0f}%)",
    f"{pod_w:.1f}% ({tp_w}/{tp_w+fn_w})",
    f"CSI = {csi_w:.1f}% | False Alarm = {fp_w}"
)
m5.metric(
    f"Proyeksi Berikutnya (t+{horizon_sel})",
    f"{next_y_hat:+.3f}",
    f"Peluang Kering: {next_prob_mod:.1f}%"
)

# ------------------------------------------------------------------------------
# EMPAT TAB UTAMA DASBOR
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Tab 1: Peringatan Dini Murni Probabilistik (Cost-Loss)",
    "📊 Tab 2: Audit Metodologis Tahap 1–5 (Multi-Kuantil, Ekor, Brier, & Kontingensi)",
    "📡 Tab 3: Simulasi Sensor Outage (MICE+Lag vs Kalman AR(2))",
    "🛠️ Tab 4: Editor Data Dinamis (Tambah/Hapus Baris & Auto-Update)"
])

with tab1:
    last_date_str = pd.to_datetime(latest_row["DATE"].iloc[0]).strftime("%Y-%m")
    target_date_str = (pd.to_datetime(latest_row["DATE"].iloc[0]) + pd.DateOffset(months=horizon_sel)).strftime("%Y-%m")
    if next_prob_mod >= alert_thr:
        st_lbl, bg_col = f"SIAGA KEKERINGAN (MERAH) [P ≥ {alert_thr:.0f}%]", "#cb181d"
    elif next_prob_mod >= watch_thr:
        st_lbl, bg_col = f"WASPADA / WATCH (KUNING) [{watch_thr:.0f}% ≤ P < {alert_thr:.0f}%]", "#d95f0e"
    else:
        st_lbl, bg_col = f"NORMAL / AMAN (HIJAU) [P < {watch_thr:.0f}%]", "#006d2c"

    st.markdown(
        f"""
        <div style="background-color:{bg_col};padding:13px 18px;border-radius:8px;color:white;margin-bottom:12px;">
            <h4 style="margin:0;color:white;">🚨 PROYEKSI PERINGATAN DINI PERIODE BERIKUTNYA ({target_date_str} dari Jangkar {last_date_str}): {st_lbl}</h4>
            <p style="margin:5px 0 0 0;font-size:14px;">
                Model: <b>{model_sel}</b> | Proyeksi {sp_sel}: <b>{next_y_hat:+.3f}</b> |
                Interval {pi_method} {nom_level_sel}% (±{hw_selected:.3f}): <b>[{next_y_hat - hw_selected:+.2f}, {next_y_hat + hw_selected:+.2f}]</b> |
                Peluang Kering (≤ {tau_in:.2f}): <b>{next_prob_mod:.1f}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(df_dews_filtered) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 6.0), dpi=200, sharex=True)
        dt_plot = df_dews_filtered["DATE_TARGET"]
        ax1.fill_between(
            dt_plot,
            df_dews_filtered["PI_Low"],
            df_dews_filtered["PI_High"],
            color="#6baed6",
            alpha=0.32,
            label=f"Pita Interval Prediksi {nom_level_sel}% ({pi_method}, ±{hw_selected:.2f})"
        )
        ax1.plot(dt_plot, df_dews_filtered["Y_True"], "k-o", lw=1.5, markersize=3.5, label=f"Observasi Aktual {sp_sel}")
        ax1.plot(dt_plot, df_dews_filtered["Y_Pred"], "r--s", lw=1.5, markersize=3.5, label=f"Prediksi {model_sel}")
        ax1.axhline(tau_in, color="#cb181d", ls=":", lw=1.3, label=f"Ambang Kering (τ = {tau_in:.2f})")
        ax1.set_ylabel(f"Skor-Z {sp_sel}")
        ax1.set_title(f"Verifikasi Deret Waktu Peringatan Dini pada Periode Uji ({len(df_dews_filtered)} Bulan Terfilter)")
        ax1.legend(loc="upper left", ncol=4, fontsize=8)
        ax1.grid(True, ls=":", alpha=0.6)

        ax2.bar(
            dt_plot,
            df_dews_filtered["Prob_Mod_Pct"],
            width=20,
            color="#fe9929",
            edgecolor="black",
            lw=0.4,
            label=f"Peluang {sp_sel} ≤ {tau_in:.2f} (%)"
        )
        ax2.axhline(alert_thr, color="#cb181d", ls="--", lw=1.1, label=f"Ambang Siaga Merah (P ≥ {alert_thr:.0f}%)")
        ax2.axhline(watch_thr, color="#d95f0e", ls=":", lw=1.1, label=f"Ambang Waspada Kuning (P ≥ {watch_thr:.0f}%)")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Probabilitas (%)")
        ax2.legend(loc="upper left", ncol=3, fontsize=8)
        ax2.grid(True, ls=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader(f"📋 Tabel Buletin Peringatan Dini Murni Probabilistik ({len(df_dews_filtered)} Bulan)")
    st.dataframe(df_dews_filtered, use_container_width=True)
    st.download_button(
        "⬇️ Unduh Buletin Terfilter (.CSV)",
        data=df_dews_filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"Buletin_DEWS_{grid_sel}_{sp_sel}_{model_sel}_t{horizon_sel}.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("📊 1. Tabel Evaluasi Regresi, Uji DM-HAC Holm, & Kupiec POF (Tahap 1)")
    st.caption("Mencakup jumlah bulan tercakup (N_Hit_80), Galat Standar Binomial (SE_iid & SE_corr), dan p-value Uji Kupiec POF.")
    st.dataframe(df_eval, use_container_width=True)

    ml_path = f"DATA/PROCESSED/Kalibrasi_MultiKuantil_{sp_sel}_t{horizon_sel}.csv"
    if os.path.exists(ml_path):
        st.subheader("📈 2. Kurva Kalibrasi Multi-Kuantil 50%, 80%, 90%, 95%: Gaussian vs. Split Conformal (Tahap 2)")
        st.caption("Membandingkan lebar interval (MPIW), cakupan empiris (PICP), uji Kupiec POF, dan Winkler Interval Score lintas 4 tingkat nominal.")
        st.dataframe(pd.read_csv(ml_path), use_container_width=True)

    tb_path = f"DATA/PROCESSED/Audit_Ekor_dan_Brier_{sp_sel}_t{horizon_sel}.csv"
    if os.path.exists(tb_path):
        st.subheader("🔍 3. Audit Normalitas Shapiro-Wilk, Kurtosis Ekor, & Brier Skill Score (Tahap 3 & 4)")
        st.caption("Menguji normalitas residual, asimetri pelanggaran ekor bawah/atas, cakupan khusus bulan kering (y <= -1,0), dan Brier Skill Score.")
        st.dataframe(pd.read_csv(tb_path), use_container_width=True)

    ct_path = f"DATA/PROCESSED/Evaluasi_Kontingensi_DEWS_{sp_sel}_t{horizon_sel}.csv"
    if os.path.exists(ct_path):
        st.subheader("🎯 4. Evaluasi Kontingensi Operasional DEWS: POD (Hit Rate), FAR, & CSI (Tahap 5)")
        st.caption("Membandingkan performa klasifikasi peringatan dini pada ambang probabilistik P >= 30% (Waspada+Siaga), P >= 50% (Siaga), dan P >= 60% (Siaga Tinggi).")
        st.dataframe(pd.read_csv(ct_path), use_container_width=True)

with tab3:
    st.subheader("📡 Simulasi Interaktif Kegagalan Sensor Jangkar SPEI_t (Outage Failover)")
    c_out1, c_out2, c_out3 = st.columns(3)
    with c_out1:
        imp_method = st.selectbox("Pilih Metode Rekonstruksi Jangkar:", ["MICE+Lag", "Kalman State-Space AR(2)", "LOCF (Statis)"], index=0)
        sp_lag1 = st.number_input("Nilai SPEI_{t-1} (1 Bulan Lalu):", value=float(latest_row["lag1_ar"].iloc[0]), step=0.05)
        sp_lag2 = st.number_input("Nilai SPEI_{t-2} (2 Bulan Lalu):", value=float(latest_row["lag2_ar"].iloc[0]), step=0.05)
    with c_out2:
        prec_sim = st.number_input("Curah Hujan PRECTOTCORR (mm/bln):", value=float(latest_row["PRECTOTCORR"].iloc[0]), step=5.0)
        gwet_sim = st.slider("Kelembapan Tanah GWETTOP (0–1):", 0.10, 0.95, float(latest_row["GWETTOP"].iloc[0]), 0.01)
        rh_sim   = st.slider("Kelembapan Udara RH2M (%):", 50.0, 95.0, float(latest_row["RH2M"].iloc[0]), 0.5)
    with c_out3:
        reconstructed_anchor, desc_imp = reconstruct_missing_anchor([sp_lag1, sp_lag2], prec_sim, gwet_sim, rh_sim, method=imp_method)
        true_latest_anchor = float(latest_row[sp_sel].iloc[0])
        st.metric(
            "Jangkar SPEI_t Hasil Rekonstruksi",
            f"{reconstructed_anchor:+.4f}",
            f"Selisih vs Aktual ({true_latest_anchor:+.3f}): {reconstructed_anchor - true_latest_anchor:+.4f}"
        )
        st.info(f"Metode aktif: **{desc_imp}**")

with tab4:
    st.subheader("🛠️ Editor Data Mentah / Master (Otomatis Memperbarui Seluruh Model & Dasbor)")
    df_raw_edit = pd.read_csv(MASTER_CSV_PATH)
    edited_master = st.data_editor(df_raw_edit, num_rows="dynamic", use_container_width=True, height=400)
    if st.button("💾 Simpan Perubahan Data & Perbarui Dasbor Otomatis", type="primary"):
        edited_master["DATE"] = pd.to_datetime(edited_master["DATE"])
        edited_master = edited_master.sort_values("DATE").reset_index(drop=True)
        edited_master["YEAR"] = edited_master["DATE"].dt.year
        edited_master["MONTH"] = edited_master["DATE"].dt.month
        edited_master["TARGET_SPEI_06_t1"] = edited_master["SPEI_06"].shift(-1)
        edited_master["TARGET_SPEI_03_t1"] = edited_master["SPEI_03"].shift(-1)
        edited_master.to_csv(MASTER_CSV_PATH, index=False)
        st.cache_data.clear()
        st.success(f"Berhasil menyimpan {len(edited_master)} baris! Memperbarui model dan grafik...")
        st.rerun()