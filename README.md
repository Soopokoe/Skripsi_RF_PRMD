# MotionLens RF — Kinect UI-PRMD

Project ini memakai **satu aplikasi saja**, yaitu `app.py`. File yang sama dapat
dijalankan pada komputer lokal dan dipakai sebagai main file di Streamlit
Community Cloud. Keduanya dapat aktif bersamaan karena berjalan sebagai instance
terpisah dengan model yang sama.

Model aktif adalah Random Forest yang dilatih dari skeleton Kinect UI-PRMD.
MediaPipe mengubah video upload menjadi pola pose untuk inferensi. Video pengguna
tidak dipakai untuk training atau untuk mengubah threshold.

## Gerakan aktif

| Kode | Gerakan |
|---|---|
| `m01` | Deep Squat |
| `m04` | Side Lunge |
| `m05` | Sit to Stand |
| `m07` | Standing Shoulder Abduction |

M02 Hurdle Step tidak terdapat pada aplikasi, dataset yang diunggah, dataset
fitur, indeks model, model RF, maupun calibrator.

## Menjalankan setelah download

Gunakan Python 3.12. Dari PowerShell pada root project, pasang dependensi satu
kali:

```powershell
.\setup_venv_windows.ps1
```

Setelah selesai, jalankan langsung satu aplikasi:

```powershell
.\.venv\Scripts\python.exe app.py
```

`app.py` otomatis menjalankan Streamlit dan dapat dibuka melalui
`http://127.0.0.1:8501`. Tidak ada aplikasi Flask atau file run batch terpisah.

## Deployment Streamlit Community Cloud

1. Pilih repository ini dan branch `main`.
2. Isi **Main file path** dengan `app.py`.
3. Gunakan Python 3.12 jika pilihan versi Python tersedia.
4. Deploy.

Python 3.12 wajib digunakan karena versi MediaPipe pada project ini menyediakan
wheel Linux untuk Python 3.12. Dependensi JAX, SciPy, dan NumPy dikunci pada
kombinasi yang sama dengan environment pengujian agar instalasi cloud konsisten.
`packages.txt` memakai `libgl1` dan `libglib2.0-0t64`, yaitu paket GLib untuk
Debian Trixie yang menyediakan library `libgthread-2.0.so.0` bagi OpenCV.

Instance lokal tetap dapat dijalankan dengan perintah di atas meskipun instance
Streamlit Cloud sedang aktif.

## Training ulang

Dataset Kinect Positions dan Angles untuk empat gerakan aktif sudah disertakan,
termasuk kelas Correct dan Incorrect. Indeks preprocessing juga tersedia.

Membangun ulang dataset fitur:

```powershell
.\.venv\Scripts\python.exe rf_kinect_domain_randomized\01_build_randomized_features.py
```

Melatih ulang Random Forest, calibrator, dan threshold:

```powershell
.\.venv\Scripts\python.exe rf_kinect_domain_randomized\02_train_rf_kinect_only.py
```

Perintah training akan menimpa artefak aktif. Commit repository sudah membawa
model siap pakai, sehingga training ulang tidak diperlukan untuk menjalankan app.

## Struktur repository

```text
Skripsi_RF_PRMD/
|-- app.py
|-- README.md
|-- requirements.txt
|-- packages.txt
|-- .python-version
|-- .gitignore
|-- .gitattributes
|-- setup_venv_windows.ps1
|-- setup_venv_windows.bat
|-- pose_pattern_utils.py
|-- video_extractor.py
|-- input_adapter.py
|-- rf_calibration.py
|-- 00_augment_positions_dataset.py
|
|-- .streamlit/
|   `-- config.toml
|
|-- dataset/
|   |-- Correct/Segmented Movements/Kinect/
|   |   |-- Positions/    # m01, m04, m05, m07
|   |   `-- Angles/       # m01, m04, m05, m07
|   `-- Incorrect/Incorrect Segmented Movements/Kinect/
|       |-- Positions/    # m01, m04, m05, m07
|       `-- Angles/       # m01, m04, m05, m07
|
|-- dataset_augmented_rf_xgb_fixed/
|   |-- augmented_index.csv
|   `-- augmented_index_summary.csv
|
|-- artifacts_uiprmd_fk/
|   |-- kinect_domain_randomized_features.csv
|   |-- kinect_domain_randomized_features_summary.csv
|   `-- models/
|       |-- model_index.json
|       |-- training_metrics.json
|       |-- rf_m{01,04,05,07}_{standard,one_side}.joblib
|       `-- calibrator_m{01,04,05,07}_{standard,one_side}.joblib
|
`-- rf_kinect_domain_randomized/
    |-- README.md
    |-- 01_build_randomized_features.py
    |-- domain_randomization.py
    |-- 02_train_rf_kinect_only.py
    `-- 03_test_videos.py
```

## Yang tidak disertakan

- Dataset Vicon.
- Gerakan selain `m01`, `m04`, `m05`, dan `m07`, terutama M02.
- Folder video pengujian dan video upload pengguna.
- `.venv`, cache Python, keyframe sementara, ZIP, dan backup.

`artifacts_uiprmd_fk/kinect_domain_randomized_features.csv` adalah hasil
preprocessing yang tetap dibutuhkan oleh input adapter ketika app melakukan
inferensi. Dataset Kinect mentah digunakan hanya saat membangun ulang fitur.
