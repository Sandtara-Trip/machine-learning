# src/recommender.py

from sklearn.metrics.pairwise import cosine_similarity
from src.utils import preprocess_text, slang_dict
import numpy as np
import pickle

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

def recommend_by_query(query, df, vectorizer, tfidf_matrix, top_n=5, preferensi=None, min_rating=4.0):
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