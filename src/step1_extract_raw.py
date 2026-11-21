# ==============================================================================
# STEP 1: EKSTRAKSI NETCDF (.NC) CSIC SPEIBASE & AGREGASI HARIAN NASA POWER
# File: src/step1_extract_raw.py
# ==============================================================================
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

SENSOR_COLS = [
    "ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN",
    "RH2M", "WS2M", "GWETTOP", "PRECTOTCORR"
]

# Koordinat 5 Sel Grid Sulawesi (Resolusi 0.5 derajat CSIC SPEIbase)
SULAWESI_GRIDS = {
    "G1_Palu":     {"lat": -0.75, "lon": 119.75},
    "G2_Parigi":   {"lat": -0.75, "lon": 120.25},
    "G3_Poso":     {"lat": -1.25, "lon": 120.75},
    "G4_ToliToli": {"lat":  1.25, "lon": 120.75},
    "G5_Makassar": {"lat": -5.25, "lon": 119.25},
}


def process_nasa_daily_to_monthly(nasa_folder: str = "DATA/NASA POWER") -> pd.DataFrame:
    """
    Membaca CSV harian NASA POWER dari folder DATA/NASA POWER, membersihkan kode -999,
    dan mengagregasi menjadi data bulanan.
    """
    csv_files = glob.glob(os.path.join(nasa_folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Tidak ditemukan file CSV di dalam folder '{nasa_folder}'")
    nasa_csv_path = csv_files[0]

    # Cari baris akhir header (-END HEADER-)
    skip_rows = 0
    with open(nasa_csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if "END HEADER" in line.upper():
                skip_rows = idx + 1
                break

    df_daily = pd.read_csv(nasa_csv_path, skiprows=skip_rows)
    df_daily.columns = [c.strip().upper() for c in df_daily.columns]

    # Konversi YEAR dan DOY (Day of Year) ke tanggal kalender
    if "DOY" in df_daily.columns:
        df_daily["DATE_DAILY"] = pd.to_datetime(
            df_daily["YEAR"].astype(int).astype(str) + "-" + df_daily["DOY"].astype(int).astype(str),
            format="%Y-%j"
        )
    elif "MO" in df_daily.columns and "DY" in df_daily.columns:
        df_daily["DATE_DAILY"] = pd.to_datetime(
            dict(year=df_daily["YEAR"], month=df_daily["MO"], day=df_daily["DY"])
        )
    else:
        df_daily["DATE_DAILY"] = pd.to_datetime(df_daily["DATE"])

    # Bersihkan nilai -999 menjadi NaN
    for col in SENSOR_COLS:
        df_daily[col] = pd.to_numeric(df_daily[col], errors="coerce")
        df_daily.loc[df_daily[col] <= -900, col] = np.nan

    # Buang hari-hari di ujung akhir tahun berjalan jika seluruh sensor bernilai NaN (-999)
    df_daily = df_daily.dropna(subset=SENSOR_COLS, how="all").copy()

    # Interpolasi linier untuk hari yang terselip NaN di tengah deret
    df_daily[SENSOR_COLS] = df_daily[SENSOR_COLS].interpolate(method="linear").bfill().ffill()

    # Buat kolom DATE bulanan (awal bulan YYYY-MM-01) dan hitung jumlah hari valid per bulan
    df_daily["DATE"] = df_daily["DATE_DAILY"].dt.to_period("M").dt.to_timestamp()
    day_counts = df_daily.groupby("DATE")["DATE_DAILY"].count()
    complete_months = day_counts[day_counts >= 28].index  # Hanya gunakan bulan yang datanya lengkap (>=28 hari)
    df_daily = df_daily[df_daily["DATE"].isin(complete_months)].copy()

    # Agregasi Harian ke Bulanan:
    # PRECTOTCORR dijumlahkan (mm/bulan), variabel lain dirata-rata
    agg_rules = {
        "ALLSKY_SFC_SW_DWN": "mean",
        "T2M_MAX": "mean",
        "T2M_MIN": "mean",
        "RH2M": "mean",
        "WS2M": "mean",
        "GWETTOP": "mean",
        "PRECTOTCORR": "sum"
    }
    df_monthly = df_daily.groupby("DATE", as_index=False).agg(agg_rules)
    df_monthly["YEAR"] = df_monthly["DATE"].dt.year
    df_monthly["MONTH"] = df_monthly["DATE"].dt.month
    return df_monthly.sort_values("DATE").reset_index(drop=True)


def extract_csic_nc_files(csic_folder: str = "DATA/CSIC Global SPEI Database") -> pd.DataFrame:
    """
    Mengekstrak spei01.nc, spei03.nc, dan spei06.nc untuk G1 Lembah Palu dan 4 grid Sulawesi lainnya.
    """
    nc_map = {
        "SPEI_01": os.path.join(csic_folder, "spei01.nc"),
        "SPEI_03": os.path.join(csic_folder, "spei03.nc"),
        "SPEI_06": os.path.join(csic_folder, "spei06.nc"),
    }

    dfs_per_scale = []
    for sp_label, nc_path in nc_map.items():
        if not os.path.exists(nc_path):
            continue
        ds = xr.open_dataset(nc_path)
        var_name = "spei" if "spei" in ds.data_vars else list(ds.data_vars)[0]

        # Ekstrak G1 Lembah Palu sebagai kolom utama SPEI_01, SPEI_03, SPEI_06
        g1 = SULAWESI_GRIDS["G1_Palu"]
        sub_g1 = ds[var_name].sel(lat=g1["lat"], lon=g1["lon"], method="nearest").to_dataframe().reset_index()
        sub_g1["DATE"] = pd.to_datetime(sub_g1["time"]).dt.to_period("M").dt.to_timestamp()
        df_scale = sub_g1[["DATE", var_name]].rename(columns={var_name: sp_label})

        # Jika skala SPEI_06 atau SPEI_03, ekstrak juga G2 s.d. G5 untuk evaluasi multi-grid
        if sp_label in ["SPEI_06", "SPEI_03"]:
            for g_key, coords in SULAWESI_GRIDS.items():
                if g_key == "G1_Palu":
                    continue
                sub_g = ds[var_name].sel(lat=coords["lat"], lon=coords["lon"], method="nearest").to_dataframe().reset_index()
                sub_g["DATE"] = pd.to_datetime(sub_g["time"]).dt.to_period("M").dt.to_timestamp()
                col_g_name = f"{sp_label}_{g_key}"
                df_scale = pd.merge(
                    df_scale,
                    sub_g[["DATE", var_name]].rename(columns={var_name: col_g_name}),
                    on="DATE",
                    how="left"
                )
        ds.close()
        dfs_per_scale.append(df_scale)

    if not dfs_per_scale:
        raise FileNotFoundError(f"Tidak ditemukan file .nc di dalam folder '{csic_folder}'")

    df_csic = dfs_per_scale[0]
    for df_next in dfs_per_scale[1:]:
        df_csic = pd.merge(df_csic, df_next, on="DATE", how="outer")

    return df_csic.sort_values("DATE").reset_index(drop=True)


def run_step1_pipeline(
    nasa_folder: str = "DATA/NASA POWER",
    csic_folder: str = "DATA/CSIC Global SPEI Database",
    out_folder: str = "DATA/PROCESSED"
) -> pd.DataFrame:
    """Menjalankan ekstraksi penuh dari folder mentah dan merakit dataset_master_nasa_csic_clean.csv."""
    os.makedirs(out_folder, exist_ok=True)

    print("1. Mengagregasi data harian NASA POWER ke bulanan...")
    df_nasa_m = process_nasa_daily_to_monthly(nasa_folder)
    df_nasa_m.to_csv(os.path.join(out_folder, "nasa_power_palu_monthly.csv"), index=False)

    print("2. Mengekstrak file NetCDF (.nc) CSIC SPEIbase untuk Sulawesi...")
    df_csic_m = extract_csic_nc_files(csic_folder)
    df_csic_m.to_csv(os.path.join(out_folder, "csic_spei_5grids_monthly.csv"), index=False)

    print("3. Menggabungkan NASA POWER & CSIC SPEIbase berdasarkan tanggal bulanan (DATE)...")
    df_master = pd.merge(df_nasa_m, df_csic_m, on="DATE", how="inner").sort_values("DATE").reset_index(drop=True)

    # Interpolasi jika ada sel kosong di tengah deret
    for c in ["SPEI_03", "SPEI_06"]:
        if c in df_master.columns:
            df_master[c] = df_master[c].interpolate(method="linear").bfill().ffill()

    # Bentuk kolom target t+1
    df_master["TARGET_SPEI_06_t1"] = df_master["SPEI_06"].shift(-1)
    df_master["TARGET_SPEI_03_t1"] = df_master["SPEI_03"].shift(-1)
    if "SPEI_01" in df_master.columns:
        df_master["TARGET_SPEI_01_t1"] = df_master["SPEI_01"].shift(-1)

    master_path = os.path.join(out_folder, "dataset_master_nasa_csic_clean.csv")
    df_master.to_csv(master_path, index=False)

    print(f"[SELESAI STEP 1] File tersimpan di: {master_path}")
    print(f"Rentang Waktu Irisan Valid: {df_master['DATE'].min().strftime('%Y-%m')} s.d. {df_master['DATE'].max().strftime('%Y-%m')} ({len(df_master)} bulan)")
    print(f"Rata-rata PRECTOTCORR: {df_master['PRECTOTCORR'].mean():.2f} ± {df_master['PRECTOTCORR'].std(ddof=1):.2f} mm/bulan")
    return df_master


if __name__ == "__main__":
    run_step1_pipeline()# ==============================================================================
# STEP 1: EKSTRAKSI NETCDF (.NC) CSIC SPEIBASE & AGREGASI HARIAN NASA POWER
# File: src/step1_extract_raw.py
# ==============================================================================
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

SENSOR_COLS = [
    "ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN",
    "RH2M", "WS2M", "GWETTOP", "PRECTOTCORR"
]

# Koordinat 5 Sel Grid Sulawesi (Resolusi 0.5 derajat CSIC SPEIbase)
SULAWESI_GRIDS = {
    "G1_Palu":     {"lat": -0.75, "lon": 119.75},
    "G2_Parigi":   {"lat": -0.75, "lon": 120.25},
    "G3_Poso":     {"lat": -1.25, "lon": 120.75},
    "G4_ToliToli": {"lat":  1.25, "lon": 120.75},
    "G5_Makassar": {"lat": -5.25, "lon": 119.25},
}


def process_nasa_daily_to_monthly(nasa_folder: str = "DATA/NASA POWER") -> pd.DataFrame:
    """
    Membaca CSV harian NASA POWER dari folder DATA/NASA POWER, membersihkan kode -999,
    dan mengagregasi menjadi data bulanan.
    """
    csv_files = glob.glob(os.path.join(nasa_folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Tidak ditemukan file CSV di dalam folder '{nasa_folder}'")
    nasa_csv_path = csv_files[0]

    # Cari baris akhir header (-END HEADER-)
    skip_rows = 0
    with open(nasa_csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if "END HEADER" in line.upper():
                skip_rows = idx + 1
                break

    df_daily = pd.read_csv(nasa_csv_path, skiprows=skip_rows)
    df_daily.columns = [c.strip().upper() for c in df_daily.columns]

    # Konversi YEAR dan DOY (Day of Year) ke tanggal kalender
    if "DOY" in df_daily.columns:
        df_daily["DATE_DAILY"] = pd.to_datetime(
            df_daily["YEAR"].astype(int).astype(str) + "-" + df_daily["DOY"].astype(int).astype(str),
            format="%Y-%j"
        )
    elif "MO" in df_daily.columns and "DY" in df_daily.columns:
        df_daily["DATE_DAILY"] = pd.to_datetime(
            dict(year=df_daily["YEAR"], month=df_daily["MO"], day=df_daily["DY"])
        )
    else:
        df_daily["DATE_DAILY"] = pd.to_datetime(df_daily["DATE"])

    # Bersihkan nilai -999 menjadi NaN
    for col in SENSOR_COLS:
        df_daily[col] = pd.to_numeric(df_daily[col], errors="coerce")
        df_daily.loc[df_daily[col] <= -900, col] = np.nan

    # Buang hari-hari di ujung akhir tahun berjalan jika seluruh sensor bernilai NaN (-999)
    df_daily = df_daily.dropna(subset=SENSOR_COLS, how="all").copy()

    # Interpolasi linier untuk hari yang terselip NaN di tengah deret
    df_daily[SENSOR_COLS] = df_daily[SENSOR_COLS].interpolate(method="linear").bfill().ffill()

    # Buat kolom DATE bulanan (awal bulan YYYY-MM-01) dan hitung jumlah hari valid per bulan
    df_daily["DATE"] = df_daily["DATE_DAILY"].dt.to_period("M").dt.to_timestamp()
    day_counts = df_daily.groupby("DATE")["DATE_DAILY"].count()
    complete_months = day_counts[day_counts >= 28].index  # Hanya gunakan bulan yang datanya lengkap (>=28 hari)
    df_daily = df_daily[df_daily["DATE"].isin(complete_months)].copy()

    # Agregasi Harian ke Bulanan:
    # PRECTOTCORR dijumlahkan (mm/bulan), variabel lain dirata-rata
    agg_rules = {
        "ALLSKY_SFC_SW_DWN": "mean",
        "T2M_MAX": "mean",
        "T2M_MIN": "mean",
        "RH2M": "mean",
        "WS2M": "mean",
        "GWETTOP": "mean",
        "PRECTOTCORR": "sum"
    }
    df_monthly = df_daily.groupby("DATE", as_index=False).agg(agg_rules)
    df_monthly["YEAR"] = df_monthly["DATE"].dt.year
    df_monthly["MONTH"] = df_monthly["DATE"].dt.month
    return df_monthly.sort_values("DATE").reset_index(drop=True)


def extract_csic_nc_files(csic_folder: str = "DATA/CSIC Global SPEI Database") -> pd.DataFrame:
    """
    Mengekstrak spei01.nc, spei03.nc, dan spei06.nc untuk G1 Lembah Palu dan 4 grid Sulawesi lainnya.
    """
    nc_map = {
        "SPEI_01": os.path.join(csic_folder, "spei01.nc"),
        "SPEI_03": os.path.join(csic_folder, "spei03.nc"),
        "SPEI_06": os.path.join(csic_folder, "spei06.nc"),
    }

    dfs_per_scale = []
    for sp_label, nc_path in nc_map.items():
        if not os.path.exists(nc_path):
            continue
        ds = xr.open_dataset(nc_path)
        var_name = "spei" if "spei" in ds.data_vars else list(ds.data_vars)[0]

        # Ekstrak G1 Lembah Palu sebagai kolom utama SPEI_01, SPEI_03, SPEI_06
        g1 = SULAWESI_GRIDS["G1_Palu"]
        sub_g1 = ds[var_name].sel(lat=g1["lat"], lon=g1["lon"], method="nearest").to_dataframe().reset_index()
        sub_g1["DATE"] = pd.to_datetime(sub_g1["time"]).dt.to_period("M").dt.to_timestamp()
        df_scale = sub_g1[["DATE", var_name]].rename(columns={var_name: sp_label})

        # Jika skala SPEI_06 atau SPEI_03, ekstrak juga G2 s.d. G5 untuk evaluasi multi-grid
        if sp_label in ["SPEI_06", "SPEI_03"]:
            for g_key, coords in SULAWESI_GRIDS.items():
                if g_key == "G1_Palu":
                    continue
                sub_g = ds[var_name].sel(lat=coords["lat"], lon=coords["lon"], method="nearest").to_dataframe().reset_index()
                sub_g["DATE"] = pd.to_datetime(sub_g["time"]).dt.to_period("M").dt.to_timestamp()
                col_g_name = f"{sp_label}_{g_key}"
                df_scale = pd.merge(
                    df_scale,
                    sub_g[["DATE", var_name]].rename(columns={var_name: col_g_name}),
                    on="DATE",
                    how="left"
                )
        ds.close()
        dfs_per_scale.append(df_scale)

    if not dfs_per_scale:
        raise FileNotFoundError(f"Tidak ditemukan file .nc di dalam folder '{csic_folder}'")

    df_csic = dfs_per_scale[0]
    for df_next in dfs_per_scale[1:]:
        df_csic = pd.merge(df_csic, df_next, on="DATE", how="outer")

    return df_csic.sort_values("DATE").reset_index(drop=True)


def run_step1_pipeline(
    nasa_folder: str = "DATA/NASA POWER",
    csic_folder: str = "DATA/CSIC Global SPEI Database",
    out_folder: str = "DATA/PROCESSED"
) -> pd.DataFrame:
    """Menjalankan ekstraksi penuh dari folder mentah dan merakit dataset_master_nasa_csic_clean.csv."""
    os.makedirs(out_folder, exist_ok=True)

    print("1. Mengagregasi data harian NASA POWER ke bulanan...")
    df_nasa_m = process_nasa_daily_to_monthly(nasa_folder)
    df_nasa_m.to_csv(os.path.join(out_folder, "nasa_power_palu_monthly.csv"), index=False)

    print("2. Mengekstrak file NetCDF (.nc) CSIC SPEIbase untuk Sulawesi...")
    df_csic_m = extract_csic_nc_files(csic_folder)
    df_csic_m.to_csv(os.path.join(out_folder, "csic_spei_5grids_monthly.csv"), index=False)

    print("3. Menggabungkan NASA POWER & CSIC SPEIbase berdasarkan tanggal bulanan (DATE)...")
    df_master = pd.merge(df_nasa_m, df_csic_m, on="DATE", how="inner").sort_values("DATE").reset_index(drop=True)

    # Interpolasi jika ada sel kosong di tengah deret
    for c in ["SPEI_03", "SPEI_06"]:
        if c in df_master.columns:
            df_master[c] = df_master[c].interpolate(method="linear").bfill().ffill()

    # Bentuk kolom target t+1
    df_master["TARGET_SPEI_06_t1"] = df_master["SPEI_06"].shift(-1)
    df_master["TARGET_SPEI_03_t1"] = df_master["SPEI_03"].shift(-1)
    if "SPEI_01" in df_master.columns:
        df_master["TARGET_SPEI_01_t1"] = df_master["SPEI_01"].shift(-1)

    master_path = os.path.join(out_folder, "dataset_master_nasa_csic_clean.csv")
    df_master.to_csv(master_path, index=False)

    print(f"[SELESAI STEP 1] File tersimpan di: {master_path}")
    print(f"Rentang Waktu Irisan Valid: {df_master['DATE'].min().strftime('%Y-%m')} s.d. {df_master['DATE'].max().strftime('%Y-%m')} ({len(df_master)} bulan)")
    print(f"Rata-rata PRECTOTCORR: {df_master['PRECTOTCORR'].mean():.2f} ± {df_master['PRECTOTCORR'].std(ddof=1):.2f} mm/bulan")
    return df_master


if __name__ == "__main__":
    run_step1_pipeline()