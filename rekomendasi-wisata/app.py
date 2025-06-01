# app.py

from flask import Flask, request, jsonify
from src.recommender import get_recommendations, recommend_by_query_from_similarity
import pandas as pd
import pickle
import re
import os

app = Flask(__name__)

# ==== Load hasil preprocessing ====
df = pd.read_csv('data/data_clean.csv')

# Load similarity matrix
with open('models/recommender.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

# ==== Utility untuk membersihkan karakter aneh ====
def remove_directional_chars(text):
    return re.sub(r'[\u2066\u2067\u2068\u2069]', '', text)


@app.route("/")
def index():
    return jsonify({
        "message": "API Rekomendasi Wisata Aktif 🚀",
        "usage": "/rekomendasi-wisata?query=...&preferensi=...&top_n=..."
    })


@app.route("/rekomendasi-wisata", methods=["GET"])
def rekomendasi():
    query = request.args.get("query", "").strip().lower()
    preferensi = request.args.get("preferensi", "").strip().lower()
    top_n = int(request.args.get("top_n", 5))

    if not query:
        return jsonify({"error": "Parameter 'query' wajib diisi."}), 400

    hasil = recommend_by_query_from_similarity(
        query=query,
        df=df,
        similarity_matrix=similarity_matrix,
        top_n=top_n,
        preferensi=preferensi if preferensi else None,
        min_rating=4.0
    )

    # rekomendasi_list = []
    # for _, row in hasil.iterrows():
    #     rekomendasi_list.append({
    #         "Deskripsi_Singkat": remove_directional_chars(row.get('Deskripsi_Singkat', '')),
    #         "Kategori": row.get('Kategori'),
    #         "Rating": row.get('Rating'),
    #         "Jumlah_Ulasan": row.get('Jumlah_Ulasan'),
    #         "Lokasi": row.get('Lokasi', ''),
    #         "Ulasan_1": remove_directional_chars(row.get('Ulasan_1', '')),
    #         "Ulasan_2": remove_directional_chars(row.get('Ulasan_2', '')),
    #         "Ulasan_3": remove_directional_chars(row.get('Ulasan_3', '')),
    #         "Ulasan_4": remove_directional_chars(row.get('Ulasan_4', '')),
    #     })

    # return jsonify({
    #     "query": query,
    #     "preferensi": preferensi,
    #     "jumlah_rekomendasi": len(rekomendasi_list),
    #     "hasil": rekomendasi_list
    # })
    
    return jsonify({
        "query": query,
        "preferensi": preferensi,
        "jumlah_rekomendasi": len(hasil.to_dict(orient="records")),
        "hasil": hasil.to_dict(orient="records")
    })


if __name__ == '__main__':
    import os
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)
