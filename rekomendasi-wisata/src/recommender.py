# src/recommender.py

import difflib
from src.utils import preprocess_text, slang_dict
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

def get_recommendations(index, similarity_matrix, df, top_n=10,
                        kategori_filter=True,
                        min_rating=4.0,
                        preferensi=None):
    
    sim_scores = list(enumerate(similarity_matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:]

    rekomendasi = []
    for i, score in sim_scores:
        item = df.iloc[i]

        if kategori_filter and item['Kategori'] != df.iloc[index]['Kategori']:
            continue

        if item['Rating'] < min_rating:
            continue

        if preferensi:
            preferensi = preferensi.lower()
            keywords = preferensi_map.get(preferensi, [preferensi])
            if not any(keyword in item['text_clean'] for keyword in keywords):
                continue

        rekomendasi.append((i, score))
        if len(rekomendasi) >= top_n:
            break

    return df.iloc[[i for i, _ in rekomendasi]]

def recommend_by_query_from_similarity(query, df, similarity_matrix, top_n=5, preferensi=None, min_rating=4.0):
    query_clean = preprocess_text(query, slang_dict)

    best_index = df['text_clean'].apply(
        lambda x: difflib.SequenceMatcher(None, x, query_clean).ratio()
    ).idxmax()

    return get_recommendations(
        index=best_index,
        similarity_matrix=similarity_matrix,
        df=df,
        top_n=top_n,
        kategori_filter=False,
        min_rating=min_rating,
        preferensi=preferensi
    )

def save_similarity_matrix(similarity_matrix, path='recommender.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(similarity_matrix, f)

def load_similarity_matrix(path='recommender.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)