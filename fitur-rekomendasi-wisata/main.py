# main.py

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import (
    build_vectorizer, build_additional_features, combine_features,
    save_vectorizer, save_pickle, save_tfidf_matrix,
    load_vectorizer, load_pickle, load_tfidf_matrix
)
from src.recommender import build_similarity_matrix, recommend_by_query, save_similarity_matrix
from src.utils import slang_dict
from src.constants import kategori_mapping
import pandas as pd
import nltk
import os

RAW_PATH = 'data/raw/data_destinasi_wisata_raw.csv'
PROCESSED_PATH = 'data/processed/data_clean.csv'
TFIDF_PATH = 'data/interim/tfidf_matrix.npz'
SIM_MATRIX_PATH = 'models/recommender.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'
ENCODER_PATH = 'models/category_encoder.pkl'
SCALER_PATH = 'models/numeric_scaler.pkl'


def main():
    print("\n[1] Memuat dan membersihkan data...")
    df = load_and_clean_data(RAW_PATH, slang_dict, kategori_mapping)
    df.to_csv(PROCESSED_PATH, index=False)

    print("[2] Membuat vektor TF-IDF dan fitur tambahan...")
    vectorizer, tfidf_matrix = build_vectorizer(df['text_clean'])
    category_encoded, numeric_features, encoder, scaler = build_additional_features(df)
    final_matrix = combine_features(tfidf_matrix, category_encoded, numeric_features)

    print("[3] Menyimpan model dan fitur...")
    save_vectorizer(vectorizer, VECTORIZER_PATH)
    save_pickle(encoder, ENCODER_PATH)
    save_pickle(scaler, SCALER_PATH)
    save_tfidf_matrix(final_matrix, TFIDF_PATH)

    print("[4] Membangun matriks kesamaan...")
    similarity_matrix = build_similarity_matrix(final_matrix)
    save_similarity_matrix(similarity_matrix, SIM_MATRIX_PATH)

    # ======== INPUT QUERY DARI USER ==========
    print("\n[5] Rekomendasi berdasarkan input pengguna")
    print("--------------------------------------")
    query = input("Tulis kebutuhanmu (misal: 'tempat belanja untuk keluarga'): ").strip().lower()
    preferensi = input("Preferensi spesifik (opsional, misal: keluarga, belanja, olahraga, dll): ").strip().lower()
    top_n = int(input("Jumlah rekomendasi yang diinginkan: "))

    hasil = recommend_by_query(
        query=query,
        df=df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_n=top_n,
        preferensi=preferensi if preferensi else None,
        min_rating=4.0
    )

    print(f"\n✅ Rekomendasi berdasarkan query: '{query}' dan preferensi '{preferensi}':\n")
    print(hasil[['Deskripsi_Singkat', 'Kategori', 'Rating']])


if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/interim', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    main()