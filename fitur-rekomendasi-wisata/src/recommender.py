#src/recommender.py

from sklearn.metrics.pairwise import cosine_similarity
from src.utils import slang_dict, preprocess_text
import numpy as np
import pickle

# Mapping manual preferensi pengguna ke daftar kata/frasa terkait
preferensi_map = {
    "keluarga": ["keluarga", "ramah anak", "anak-anak", "anak kecil", "family"],
    "pasangan": ["pasangan", "romantis", "honeymoon", "bulan madu"],
    "pelajar": ["pelajar", "edukatif", "belajar", "sekolah", "mahasiswa"],
    "turis": ["turis", "wisatawan", "asing", "traveler"],
    "belanja": ["belanja", "mall", "pusat oleh-oleh", "shopping", "toko"],
    "nongkrong": ["nongkrong", "cafe", "ngopi", "warung", "kopi", "hangout"],
    "olahraga": ["olahraga", "jogging", "lari", "senam", "sepeda", "outdoor"],
    "piknik": ["piknik", "berkumpul", "hamparan rumput", "tamasya"],
    "tenang": ["tenang", "damai", "sunyi", "sepi", "menenangkan"],
    "ramai": ["ramai", "hidup", "keramaian", "meriah", "ramai pengunjung"],
}

def build_similarity_matrix(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)

def get_recommendations(index, similarity_matrix, df, top_n=10,
                        kategori_filter=True,
                        min_rating=4.0,
                        preferensi=None):
    
    """
    index            : indeks baris dari destinasi acuan
    similarity_matrix: matriks kesamaan berbasis konten (TF-IDF cosine)
    df               : DataFrame destinasi
    top_n            : jumlah rekomendasi akhir yang diambil
    kategori_filter  : filter berdasarkan kategori sama (True/False)
    min_rating       : ambang batas rating minimal destinasi
    preferensi       : string preferensi user (misal: 'keluarga', 'olahraga', 'belanja')
    """
    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} di luar rentang dataset (0 - {len(df)-1})")
    
    sim_scores = list(enumerate(similarity_matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Buang dirinya sendiri
    sim_scores = sim_scores[1:]

    rekomendasi = []
    for i, score in sim_scores:
        item = df.iloc[i]

        # ✅ Filter kategori
        if kategori_filter and item['Kategori'] != df.iloc[index]['Kategori']:
            continue

        # ✅ Filter rating
        if item['Rating'] < min_rating:
            continue

        # ✅ Filter preferensi (jika disediakan)
        if preferensi:
            preferensi = preferensi.lower()
            keywords = preferensi_map.get(preferensi, [preferensi])  # fallback ke keyword tunggal

            # Cek apakah salah satu keyword muncul dalam text_clean
            if not any(keyword in item['text_clean'] for keyword in keywords):
                continue

        rekomendasi.append((i, score))

        # Stop jika sudah cukup
        if len(rekomendasi) >= top_n:
            break

    # Ambil baris data dari indeks hasil rekomendasi
    recommended_indices = [i for i, _ in rekomendasi]
    return df.iloc[recommended_indices]

def recommend_by_query(query, df, vectorizer, tfidf_matrix, top_n=5, preferensi=None, min_rating=4.5):
    query_clean = preprocess_text(query, slang_dict)
    query_vec = vectorizer.transform([query_clean])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    sim_indices = sim_scores.argsort()[::-1]

    hasil = []
    for i in sim_indices:
        row = df.iloc[i]
        if row['Rating'] < min_rating:
            continue

        if preferensi:
            keywords = preferensi_map.get(preferensi, [preferensi])
            if not any(k in row['text_clean'] for k in keywords):
                continue

        hasil.append((i, sim_scores[i]))
        if len(hasil) >= top_n:
            break

    return df.iloc[[i for i, _ in hasil]]


def save_similarity_matrix(similarity_matrix, path='recommender.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(similarity_matrix, f)

def load_similarity_matrix(path='recommender.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)