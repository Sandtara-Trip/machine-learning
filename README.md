# 🧠 Machine Learning Projects by Sandtara-Trip

Repositori ini berisi kumpulan proyek machine learning yang dikembangkan untuk mendukung layanan prediktif dan sistem rekomendasi berbasis data. Terdapat tiga proyek utama dalam repositori ini:

- Prediksi Cuaca
- Rekomendasi Hotel
- Rekomendasi Tempat Wisata

---

## 📁 Struktur Direktori

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
