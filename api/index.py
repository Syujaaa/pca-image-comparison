import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace

app = Flask(__name__, template_folder='../templates')

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Jalur API tidak ditemukan. Cek vercel.json Anda."}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Terjadi kesalahan internal di server backend Vercel."}), 500

# KALIBRASI THRESHOLD: Untuk wajah beda usia (anak vs dewasa) dengan PCA, 
# nilai 0.58 - 0.60 adalah titik potong optimal agar tidak terjadi False Rejection.
SIMILARITY_THRESHOLD = 0.50

def enhance_image_for_old_photos(img_array):
    """
    Fungsi untuk memperbaiki kontras foto lama (masa kecil) menggunakan CLAHE.
    """
    # Konversi ke LAB color space
    lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Terapkan CLAHE hanya pada L-channel (Lightness) agar warna tidak rusak
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    # Gabungkan kembali dan konversi ke BGR
    merged = cv2.merge((cl, a_channel, b_channel))
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_img

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model_wajah.pkl')

pca_model = None
is_model_ready = False

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            data_model = pickle.load(f)
        pca_model = data_model["pca_model"]
        is_model_ready = True
    except Exception as e:
        print(f"Error memuat model: {str(e)}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/compare', methods=['POST'])
@app.route('/compare', methods=['POST'])
def compare_faces(any_path=None):
    if not is_model_ready:
        return jsonify({"error": "Model PCA belum siap."}), 500

    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Dua foto dibutuhkan."}), 400
        
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    def get_deepface_embedding(file_obj):
        if file_obj.filename == '':
            return None, "Berkas kosong."
            
        img_bytes = file_obj.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Gagal membaca format gambar."
            
        try:
            # UPGRADE 1: Tambahkan align=True untuk meluruskan rotasi wajah
            results = DeepFace.represent(
                img, 
                model_name="ArcFace", 
                enforce_detection=True, 
                align=True 
            )
            return results[0]["embedding"], None
        except ValueError:
            return None, "Wajah tidak terdeteksi di salah satu foto."

    emb1, err1 = get_deepface_embedding(file1)
    if err1: return jsonify({"error": err1}), 400
        
    emb2, err2 = get_deepface_embedding(file2)
    if err2: return jsonify({"error": err2}), 400
        
    try:
        # 1. Hitung Kemiripan dari Fitur ArcFace Murni (Sangat kuat untuk beda usia)
        sim_arcface = float(cosine_similarity([emb1], [emb2])[0][0])

        # 2. Transformasi vektor ArcFace lewat "kacamata" PCA (Sesuai syarat modelmu)
        vec1 = pca_model.transform([emb1])[0]
        vec2 = pca_model.transform([emb2])[0]
        
        # Hitung Kemiripan dari ruang PCA
        sim_pca = float(cosine_similarity([vec1], [vec2])[0][0])
        eucl_dist = float(np.linalg.norm(vec1 - vec2))
        
        # UPGRADE 2: HYBRID SCORING (Ensemble)
        # Kita ambil 70% keputusan dari ArcFace yang kebal usia, dan 30% dari PCA
        hybrid_score = (sim_arcface * 0.70) + (sim_pca * 0.30)
        
        # Normalisasi agar tidak minus
        display_similarity = max(0.0, hybrid_score)
        
        is_match = bool(display_similarity >= SIMILARITY_THRESHOLD)
        
        # Penyesuaian label keyakinan yang lebih realistis untuk hybrid score
        if display_similarity >= 0.70: confidence = "Sangat Tinggi"
        elif display_similarity >= 0.60: confidence = "Tinggi"
        elif display_similarity >= SIMILARITY_THRESHOLD: confidence = "Sedang"
        else: confidence = "Rendah"
            
        return jsonify({
            "success": True,
            "is_match": is_match,
            "similarity": round(display_similarity, 4),
            "distance": round(eucl_dist, 3),
            "threshold": SIMILARITY_THRESHOLD,
            "confidence": confidence
        })
        
    except Exception as e:
        return jsonify({"error": f"Kesalahan komputasi: {str(e)}"}), 500