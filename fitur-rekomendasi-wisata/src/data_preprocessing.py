#src/data_preprocessing.py

import os
import pandas as pd
from src.utils import preprocess_text

def fix_latitude(lat):
    if isinstance(lat, str):
        parts = lat.split('.')
        if len(parts) == 3:
            return float(f"{parts[0]}.{parts[1]}{parts[2]}")
    return lat

def fix_longitude(lon):
    if isinstance(lon, str):
        parts = lon.split('.')
        if len(parts) == 3:
            return float(f"{parts[0]}.{parts[1]}{parts[2]}")
    return lon

def load_and_clean_data(file_path, slang_dict, kategori_mapping):
    df = pd.read_csv(file_path)
    required_cols = ['Deskripsi_Singkat', 'Ulasan_1', 'Ulasan_2', 'Ulasan_3', 'Ulasan_4', 'Kategori', 'Rating', 'Jumlah_Ulasan', 'Latitude', 'Longitude']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset.")
        
    # 🔍 Tangani nilai null (bila ada)
    df['Rating'] = df['Rating'].fillna("0")
    df['Jumlah_Ulasan'] = df['Jumlah_Ulasan'].fillna("0")
    df[['Deskripsi_Singkat', 'Ulasan_1', 'Ulasan_2', 'Ulasan_3', 'Ulasan_4']] = df[['Deskripsi_Singkat', 'Ulasan_1', 'Ulasan_2', 'Ulasan_3', 'Ulasan_4']].fillna("")

    # Fix format value pada kolom Latitude, Longitude, Rating, Jumlah_Ulasan
    df['Latitude'] = df['Latitude'].apply(fix_latitude)
    df['Longitude'] = df['Longitude'].apply(fix_longitude)
    df['Rating'] = df['Rating'].str.replace(',', '.').astype(float)
    df['Kategori'] = df['Kategori'].map(kategori_mapping)
    df['Jumlah_Ulasan'] = df['Jumlah_Ulasan'].astype(str).str.replace('.', '', regex=False).astype(int)

    # Membuat kolom baru untuk teks gabungan
    df['Teks_Gabungan'] = df[['Deskripsi_Singkat', 'Ulasan_1', 'Ulasan_2', 'Ulasan_3', 'Ulasan_4']].fillna('').agg(' '.join, axis=1)
    df['text_clean'] = df['Teks_Gabungan'].apply(preprocess_text, args=(slang_dict,))
    return df

def save_processed_data(df, path='data/processed/data_clean.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)