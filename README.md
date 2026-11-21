# SPEI Multi-Horizon Drought Forecasting & Probabilistic Cost-Loss DEWS (Sulawesi Grids)

Repositori kode dan data terproses untuk naskah:
> **Evaluasi Komparatif Peramalan Indeks Kekeringan SPEI Multi-Horizon dan Prototipe Peringatan Dini Probabilistik Cost-Loss pada Grid Sulawesi**
> Rafli Ghozali, I Ketut Divta Suryawan, Akbar Tri Wicaksono — Program Studi Magister Teknik Informatika, Universitas Tadulako

## Struktur Proyek
- `run_unified_pipeline.py` — Pipa komputasi tunggal terpadu (14 model kapasitas dangkal, kapasitas tinggi, 10-seed, Residual-on-Ridge, ablasi fitur & embargo, multi-horizon, rolling-origin, outage/latency 100 mask, kalibrasi Christoffersen & Winkler, BSS, Expected Cost, dan REV).
- `run_micro_verification.py` — Skrip verifikasi mikro ($\Delta\text{Cost}$ vs. $E_{\text{clim}}$ dan $\min(E_{\text{clim}}, E_{\text{persist}})$, rincian 10-seed RF-Direct, DM-HAC prediksi rerata 10-seed, serta uji DM-HAC 5 model pada $t+6$).
- `app_dashboard.py` — Prototipe aplikasi web interaktif DEWS 4-tab berbasis Streamlit.
- `src/` — Modul terpisah ekstraksi NetCDF/CSV (`step1_extract_raw.py`), prapemrosesan fitur kausal (`step2_preprocess_features.py`), dan pelatihan/evaluasi (`step3_train_and_evaluate.py`).
- `Data/PROCESSED/` — Dataset bulanan bersih (`dataset_master_nasa_csic_clean.csv`, `csic_spei_5grids_monthly.csv`, `nasa_power_palu_monthly.csv`).
- `unified_results.json` & `micro_results.json` — Keluaran terpadu seluruh tabel pada naskah.

## Sumber Data Mentah
- **CSIC Global SPEI Database v2.11.0** (`spei06.nc`, `spei03.nc`): unduh dari [https://spei.csic.es](https://spei.csic.es) lalu simpan di `Data/CSIC Global SPEI Database/` (dikecualikan dari repositori melalui `.gitignore`).
- **NASA POWER Release 8.0.1** (data harian G1 Palu): tersedia pada `Data/NASA POWER/` dan dapat diakses di [https://power.larc.nasa.gov](https://power.larc.nasa.gov).

## Cara Menjalankan
```bash
# 1. Buat lingkungan virtual & pasang dependensi
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# 2. Jalankan pipa evaluasi tunggal terpadu
python run_unified_pipeline.py

# 3. Jalankan skrip verifikasi mikro (ΔCost, 10-seed, t+6)
python run_micro_verification.py

# 4. Jalankan prototipe dasbor Streamlit
streamlit run app_dashboard.py
```

## Lingkungan Uji Referensi
Python 3.10.7, Windows 10 (Intel x86_64, 4 logical cores), NumPy 2.2.6, Pandas 2.3.3, Scikit-Learn 1.7.2, SciPy 1.15.3, Statsmodels 0.15.0, XGBoost 3.2.0, CatBoost 1.2.10.