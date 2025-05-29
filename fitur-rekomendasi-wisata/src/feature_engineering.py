#src/feature_engineering.py

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack
import pickle
import os

def build_vectorizer(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def save_vectorizer(vectorizer, path='models/vectorizer.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_vectorizer(path='models/vectorizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_tfidf_matrix(matrix, path='data/interim/tfidf_matrix.npz'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_npz(path, matrix)

def load_tfidf_matrix(path='data/interim/tfidf_matrix.npz'):
    return load_npz(path)


def build_additional_features(df):
    # One-hot encoding kategori
    encoder = OneHotEncoder(sparse_output=True)
    category_encoded = encoder.fit_transform(df[['Kategori']])

    # Scaling fitur numerik
    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(df[['Rating', 'Jumlah_Ulasan']])

    return category_encoded, numeric_features, encoder, scaler

def combine_features(tfidf_matrix, category_encoded, numeric_features):
    return hstack([tfidf_matrix, category_encoded, numeric_features])

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)