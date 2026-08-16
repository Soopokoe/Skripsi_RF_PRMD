# MotionLens RF — Kinect UI-PRMD

Repository ringan untuk klasifikasi kualitas gerakan **Correct/Incorrect** dari
video menggunakan MediaPipe Pose dan Random Forest. Model, calibrator, dan
threshold dilatih dari skeleton Kinect UI-PRMD. Video upload hanya digunakan
untuk inferensi dan tidak mengubah model.

## Gerakan aktif

| Kode | Gerakan |
|---|---|
| `m01` | Deep Squat |
| `m04` | Side Lunge |
| `m05` | Sit to Stand |
| `m07` | Standing Shoulder Abduction |

Hurdle Step (`m02`) tidak disertakan pada kode gerakan aktif, data fitur,
indeks model, model RF, calibrator, maupun antarmuka.

## Menjalankan secara lokal

Gunakan Python 3.12. Dari PowerShell pada root repository:

```powershell
.\setup_venv_windows.ps1
```

Aplikasi Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Buka `http://127.0.0.1:8501`.

Aplikasi Flask:

```powershell
.\.venv\Scripts\python.exe rf_kinect_domain_randomized\app.py
```

Buka `http://127.0.0.1:5001`.

## Deployment Streamlit Community Cloud

1. Push seluruh isi repository ini ke GitHub.
2. Buat aplikasi baru di Streamlit Community Cloud.
3. Pilih repository dan branch `main`.
4. Isi **Main file path** dengan `streamlit_app.py`.
5. Gunakan Python 3.12 jika pilihan versi Python ditampilkan, lalu deploy.

`packages.txt` dan `.streamlit/config.toml` sudah disertakan. Dataset mentah,
video, `.venv`, cache, serta file hasil analisis tidak perlu diunggah.

## Struktur file yang diunggah

```text
Skripsi_RF_PRMD/
|-- README.md
|-- requirements.txt
|-- packages.txt
|-- .python-version
|-- .gitignore
|-- setup_venv_windows.ps1
|-- setup_venv_windows.bat
|-- streamlit_app.py
|-- pose_pattern_utils.py
|-- video_extractor.py
|-- input_adapter.py
|-- rf_calibration.py
|-- 00_augment_positions_dataset.py
|
|-- .streamlit/
|   `-- config.toml
|
|-- artifacts_uiprmd_fk/
|   |-- kinect_domain_randomized_features.csv
|   |-- kinect_domain_randomized_features_summary.csv
|   `-- models/
|       |-- model_index.json
|       |-- training_metrics.json
|       |-- rf_m01_standard.joblib
|       |-- rf_m01_one_side.joblib
|       |-- rf_m04_standard.joblib
|       |-- rf_m04_one_side.joblib
|       |-- rf_m05_standard.joblib
|       |-- rf_m05_one_side.joblib
|       |-- rf_m07_standard.joblib
|       |-- rf_m07_one_side.joblib
|       |-- calibrator_m01_standard.joblib
|       |-- calibrator_m01_one_side.joblib
|       |-- calibrator_m04_standard.joblib
|       |-- calibrator_m04_one_side.joblib
|       |-- calibrator_m05_standard.joblib
|       |-- calibrator_m05_one_side.joblib
|       |-- calibrator_m07_standard.joblib
|       `-- calibrator_m07_one_side.joblib
|
`-- rf_kinect_domain_randomized/
    |-- README.md
    |-- 01_build_randomized_features.py
    |-- domain_randomization.py
    |-- 02_train_rf_kinect_only.py
    |-- 03_test_videos.py
    |-- app.py
    |-- templates/
    |   `-- index.html
    `-- static/
        |-- style.css
        `-- generated/
            `-- .gitkeep
```

## Yang tidak diunggah

- `dataset/` Kinect dan Vicon mentah.
- `dataset_augmented_rf_xgb_fixed/` dan seluruh hasil augmentasi.
- `Video/` dan video upload pengguna.
- `.venv/`, cache Python, hasil keyframe, JSON pengujian, ZIP, dan backup.
- Model atau data untuk `m02` Hurdle Step.

Dataset fitur `artifacts_uiprmd_fk/kinect_domain_randomized_features.csv` tetap
disertakan karena dibutuhkan input adapter saat inferensi. File ini bukan dataset
Kinect mentah dan ukurannya sekitar 28 MB.

## Catatan training

Script preprocessing dan training disimpan untuk dokumentasi/reproduksi kode.
Untuk menjalankan training ulang, pengguna harus menyediakan sendiri dataset
Kinect UI-PRMD dan membuat `augmented_index.csv`; keduanya sengaja tidak berada
di repository deployment ini. Aplikasi dapat langsung berjalan menggunakan
artefak model yang sudah tersedia tanpa training ulang.
