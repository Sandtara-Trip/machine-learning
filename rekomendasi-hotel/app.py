from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import os

app = Flask(__name__)

# Load model dan data
tfidf = joblib.load('tfidf_model.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
df_grouped = joblib.load('df_grouped.pkl')

# Fungsi rekomendasi
def rekomendasi_dari_review(teks_review_user):
    user_vec = tfidf.transform([teks_review_user])
    sim_scores = linear_kernel(user_vec, tfidf_matrix).flatten()
    sorted_indices = sim_scores.argsort()[::-1]
    hasil = df_grouped.iloc[sorted_indices][['Nama_hotel', 'User_Rating']]
    return hasil.to_dict(orient='records')

# Endpoint GET dengan query parameter
@app.route('/rekomendasi', methods=['GET'])
def rekomendasi_api():
    review_user = request.args.get('review')
    if not review_user:
        return jsonify({'error': 'Query "review" harus disertakan'}), 400
    
    hasil = rekomendasi_dari_review(review_user)
    return jsonify({'rekomendasi': hasil})

# Tes endpoint
@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Gunakan endpoint /rekomendasi?review=... untuk rekomendasi hotel.'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)