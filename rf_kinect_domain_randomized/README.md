# RF Kinect-only dengan Domain Randomization

Seluruh training, probability calibration, dan threshold berasal dari Kinect.
Video MediaPipe hanya dipakai untuk pengujian terakhir.

Preprocessing memakai urutan resmi 22 joint UI-PRMD dan memasangkan setiap file
`Positions` dengan file `Angles`. Posisi global direkonstruksi melalui forward
kinematics hierarkis menggunakan orientasi Euler UI-PRMD; penjumlahan translasi
tanpa rotasi parent tidak digunakan lagi.

```powershell
python rf_kinect_domain_randomized/01_build_randomized_features.py
python rf_kinect_domain_randomized/02_train_rf_kinect_only.py
python rf_kinect_domain_randomized/03_test_videos.py
```

Domain randomization dilakukan sebelum ekstraksi fitur dan meliputi proyeksi kamera
depan/serong/samping, perspective distortion, ketidakpastian depth, noise landmark,
occlusion satu sisi, pemilihan sendi terlihat, temporal warp, dan penahanan fase.

Threshold dipilih dari probabilitas out-of-fold `StratifiedGroupKFold` berdasarkan
subjek. Logistic/Platt calibration juga dilatih dari probabilitas out-of-fold Kinect.
Tidak ada video yang digunakan untuk menurunkan threshold.

## Adapter input MediaPipe

Sebelum `predict_proba`, input video diselaraskan ke domain fitur Kinect. Adapter
tidak melatih ulang Random Forest, tidak mengubah calibrator, dan tidak mengubah
angka threshold.

- Offset perspektif, amplitudo fleksi, tinggi tubuh, dan durasi fase dinormalisasi
  pada pola video tubuh bawah.
- Metadata rotasi video ponsel diterapkan otomatis sebelum MediaPipe dan sebelum
  pembuatan keyframe, sehingga fitur vertikal tidak dihitung pada frame menyamping.
- Deep Squat diproyeksikan 50% menuju median 15 tetangga Kinect terdekat yang
  dipilih dari bentuk kurva tanpa membaca label Correct/Incorrect.
- Pada Side Lunge, fitur jarak kaki dan torso yang tidak sebanding antara protokol
  video kaki-terbuka dan UI-PRMD diganti median seluruh data Kinect tanpa label,
  lalu ringkasannya diproyeksikan 25% ke tetangga Kinect tanpa label.
- Sit to Stand yang hanya merekam duduk-ke-berdiri dilengkapi menjadi konteks fase
  simetris, lalu diproyeksikan 25% berdasarkan bentuk temporal tanpa label.
- Shoulder Abduction mempertahankan profil lengan dan trimming jeda awal, lalu
  ringkasannya diproyeksikan 25% ke tetangga Kinect tanpa label.

Adapter memperbaiki kompatibilitas sensor, tetapi belum menggantikan evaluasi
dengan kumpulan video implementasi Correct dan Incorrect yang lebih besar.

Generator default mengambil lima repetisi untuk setiap kombinasi gerakan, kelas,
dan subjek: total 400 rekaman asli dengan semua 10 subjek tetap terwakili. Gunakan
`--repetitions-per-subject-label 10` bila ingin memproses seluruh 800 rekaman.

Model aktif dan metrik tersimpan di folder `artifacts_uiprmd_fk` pada root
repository dan digunakan oleh aplikasi Streamlit `app.py`.

## Menjalankan aplikasi

Dari root project:

```powershell
.\.venv\Scripts\python.exe app.py
```

Kemudian buka `http://127.0.0.1:8501`. Model dan calibrator sudah tersedia di
`artifacts_uiprmd_fk/models` pada root repository.

Setelah analisis, aplikasi menampilkan tiga keyframe setiap repetisi (awal, fase
utama, akhir). Overlay disesuaikan menurut gerakan: Shoulder Abduction menampilkan
seluruh tubuh dengan fokus lengan, sedangkan gerakan kaki lainnya menekankan
pinggul sampai kaki.
Badge keyframe adalah pemeriksaan geometri yang dapat dijelaskan, bukan prediksi
Random Forest per frame; keputusan RF tetap menggunakan pola satu repetisi penuh.
Bagian `Detail teknis` menyebutkan semua kelompok fitur yang digunakan.
