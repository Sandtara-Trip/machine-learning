# app.py

from flask import Flask, request, jsonify
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import build_vectorizer
from src.recommender import recommend_by_query
from src.constants import kategori_mapping
from src.utils import slang_dict
import os
import re

RAW_PATH = 'data/raw/data_destinasi_wisata_raw.csv'

# Preload data dan model saat server dinyalakan
df = load_and_clean_data(RAW_PATH, slang_dict, kategori_mapping)
vectorizer, tfidf_matrix = build_vectorizer(df['text_clean'])

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({
        "message": "API Rekomendasi Wisata Aktif",
        "usage": "/rekomendasi-wisata?query=...&preferensi=...&top_n=..."
    })

def remove_directional_chars(text):
    return re.sub(r'[\u2066\u2067\u2068\u2069]', '', text)

@app.route("/rekomendasi-wisata", methods=["GET"])
def rekomendasi():
    query = request.args.get("query", "").strip().lower()
    preferensi = request.args.get("preferensi", "").strip().lower()
    top_n = int(request.args.get("top_n", 5))

    if not query:
        return jsonify({"error": "Parameter 'query' wajib diisi."}), 400

    hasil = recommend_by_query(
        query=query,
        df=df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_n=top_n,
        preferensi=preferensi if preferensi else None,
        min_rating=4.0
    )

    rekomendasi_list = []
    for _, row in hasil.iterrows():
        rekomendasi_list.append({
            "Deskripsi_Singkat": remove_directional_chars(row['Deskripsi_Singkat']),
            "Kategori": row['Kategori'],
            "Rating": row['Rating'],
            "Jumlah_Ulasan": row['Jumlah_Ulasan'],
            "Lokasi": row['Lokasi'],
            "Ulasan_1": remove_directional_chars(row['Ulasan_1']),
            "Ulasan_2": remove_directional_chars(row['Ulasan_2']),
            "Ulasan_3": remove_directional_chars(row['Ulasan_3']),
            "Ulasan_4": remove_directional_chars(row['Ulasan_4']),
        })

    return jsonify({
        "query": query,
        "preferensi": preferensi,
        "jumlah_rekomendasi": len(rekomendasi_list),
        "hasil": rekomendasi_list
    })

if __name__ == '__main__':
    app.run(debug=True)