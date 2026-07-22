# Dashboard — Airline Passenger Satisfaction (Versi Deploy)

Versi ini **tidak perlu upload file** — dataset (`train.csv.gz`, `test.csv.gz`)
sudah tertanam di folder `data/`. Cocok untuk dikumpulkan sebagai link
dashboard yang bisa langsung dibuka dosen tanpa instalasi apa pun.

## Struktur folder

```
deploy/
├── dashboard.py         <- aplikasi utama
├── requirements.txt     <- dependency
├── README.md            <- file ini
└── data/
    ├── train.csv.gz
    └── test.csv.gz
```

## Cara Deploy ke Streamlit Community Cloud (gratis & permanen)

### 1. Buat repo GitHub
1. Buka https://github.com/new
2. Beri nama repo, misalnya `airline-satisfaction-dashboard`
3. Set **Public** (Streamlit Cloud gratis mensyaratkan repo publik)
4. Klik "Create repository"

### 2. Upload isi folder `deploy/` ke repo tersebut
Paling mudah lewat browser:
1. Di halaman repo, klik "Add file" → "Upload files"
2. Seret semua isi folder `deploy/` (termasuk folder `data/` beserta isinya)
3. Commit

Atau lewat command line:
```bash
cd deploy
git init
git add .
git commit -m "Initial dashboard"
git branch -M main
git remote add origin https://github.com/<username>/airline-satisfaction-dashboard.git
git push -u origin main
```

### 3. Deploy di Streamlit Community Cloud
1. Buka https://share.streamlit.io
2. Login dengan akun GitHub
3. Klik "New app"
4. Pilih repo `airline-satisfaction-dashboard`, branch `main`, file `dashboard.py`
5. Klik "Deploy"
6. Tunggu 2-5 menit (build pertama kali sedikit lama karena instal xgboost & training model)

Setelah selesai, kamu akan mendapat URL publik seperti:
```
https://airline-satisfaction-dashboard-xxxxx.streamlit.app
```

Link inilah yang dicantumkan di laporan Word/dikumpulkan ke dosen — tinggal
diklik, dashboard langsung terbuka dengan semua tab (EDA, Preprocessing,
Performa Model, Feature Importance, Insight Bisnis) tanpa perlu upload data.

## Cara jalan lokal (opsional, untuk cek dulu sebelum deploy)

```bash
cd deploy
pip install -r requirements.txt
streamlit run dashboard.py
```

## Catatan

- Training pertama kali (setelah deploy atau setelah app "sleep") butuh
  waktu karena Random Forest & XGBoost dilatih pada ±130.000 baris.
  Setelahnya di-cache oleh Streamlit selama app tidak di-restart.
- Streamlit Community Cloud app gratis akan "tidur" jika tidak diakses
  beberapa hari — dosen yang membuka link pertama kali setelah app tidur
  perlu menunggu ~30 detik untuk app bangun kembali (normal, bukan error).
"# dashboard-uasBD" 
