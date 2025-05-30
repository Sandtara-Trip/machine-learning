from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load model dan data
tfidf = joblib.load('tfidf_model.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
df_grouped = joblib.load('df_grouped.pkl')

# Fungsi rekomendasi (semua hotel, diurutkan)
def rekomendasi_dari_review(teks_review_user):
    user_vec = tfidf.transform([teks_review_user])
    sim_scores = linear_kernel(user_vec, tfidf_matrix).flatten()
    
    # Urutkan semua indeks dari skor kemiripan
    sorted_indices = sim_scores.argsort()[::-1]
    
    # Ambil semua hotel berdasarkan urutan skor kemiripan
    hasil = df_grouped.iloc[sorted_indices][['Nama_hotel', 'User_Rating']]
    return hasil.to_dict(orient='records')

# Endpoint API
@app.route('/rekomendasi', methods=['POST'])
def rekomendasi_api():
    data = request.get_json()
    if 'review' not in data:
        return jsonify({'error': 'Field "review" harus disertakan'}), 400
    
    review_user = data['review']
    hasil = rekomendasi_dari_review(review_user)
    return jsonify({'rekomendasi': hasil})

# Tes endpoint
@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'API Rekomendasi Hotel. Kirim POST ke /rekomendasi dengan field "review".'})

if __name__ == '__main__':
    app.run(debug=True)
