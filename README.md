# 🧠 Machine Learning Projects by Sandtara-Trip

![GitHub repo size](https://img.shields.io/github/repo-size/Sandtara-Trip/machine-learning)
![GitHub last commit](https://img.shields.io/github/last-commit/Sandtara-Trip/machine-learning)
![GitHub](https://img.shields.io/github/license/Sandtara-Trip/machine-learning)

Repositori ini berisi kumpulan proyek machine learning yang dikembangkan untuk mendukung layanan prediktif dan sistem rekomendasi berbasis data. Terdapat tiga proyek utama dalam repositori ini:

- Prediksi Cuaca

Model prediksi cuaca harian dan per jam berdasarkan data historis.

- Rekomendasi Hotel

Sistem rekomendasi hotel berdasarkan lokasi, harga, dan rating pengguna.

- Rekomendasi Tempat Wisata

Rekomendasi destinasi wisata berdasarkan deskripsi, kategori, dan beberapa ulasan pengguna.

---

## 📁 Struktur Direktori
```bash
machine-learning/
│
├── cuaca/
│ ├── scripts/ # Skrip prediksi cuaca
│ │ ├── forecast_daily.py # Prediksi cuaca harian
│ │ └── forecast_jam.py # Prediksi cuaca per jam
│ └── requirements.txt # Daftar dependensi Python
│
├── rekomendasi-hotel/
│ ├── app.py # Sistem rekomendasi hotel
│ └── requirements.txt # Daftar dependensi Python
│
└── rekomendasi-wisata/
├── app.py # Sistem rekomendasi wisata
└── requirements.txt # Daftar dependensi Python
```
---

## 🛠️ Cara Menjalankan

Setiap folder adalah proyek terpisah dan memiliki dependensinya sendiri. Ikuti langkah-langkah berikut untuk menjalankan salah satu proyek:

### Prediksi Cuaca
```bash
cd cuaca
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\\Scripts\\activate        # Windows
pip install -r requirements.txt
```

#### Jalankan prediksi harian:
```bash
python scripts/forecast_daily.py
```
#### Atau prediksi per jam:
```bash
python scripts/forecast_jam.py
```

### Rekomendasi Hotel
```bash
cd rekomendasi-hotel
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\\Scripts\\activate        # Windows
pip install -r requirements.txt
python app.py
```

### Rekomendasi Wisata
```bash
cd rekomendasi-wisata
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\\Scripts\\activate        # Windows
pip install -r requirements.txt
python app.py
```
