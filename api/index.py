import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

app = Flask(__name__, template_folder='../templates')

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Jalur API tidak ditemukan. Cek vercel.json Anda."}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Terjadi kesalahan internal di server backend Vercel."}), 500

SIMILARITY_THRESHOLD = 0.50
base_dir = os.path.dirname(os.path.abspath(__file__))

print("Memuat model ONNX Face Recognition secara Lokal...")
# OPTIMASI KUNCI: Mengarahkan root ke folder 'api' agar membaca model lokal di 'api/models/buffalo_sc'
face_app = FaceAnalysis(name='buffalo_sc', root=base_dir, providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(320, 320)) # Dikecilkan ke 320x320 agar kalkulasi Vercel super cepat

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
        print(f"Error memuat model PCA: {str(e)}")

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
    
    def get_onnx_embedding(file_obj):
        if file_obj.filename == '':
            return None, "Berkas kosong."
            
        img_bytes = file_obj.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Gagal membaca format gambar."
            
        try:
            faces = face_app.get(img)
            if len(faces) == 0:
                return None, "Wajah tidak terdeteksi di salah satu foto."
            return faces[0].normed_embedding, None
        except Exception:
            return None, "Gagal memproses ekstraksi wajah."

    emb1, err1 = get_onnx_embedding(file1)
    if err1: return jsonify({"error": err1}), 400
        
    emb2, err2 = get_onnx_embedding(file2)
    if err2: return jsonify({"error": err2}), 400
        
    try:
        sim_onnx = float(cosine_similarity([emb1], [emb2])[0][0])

        vec1 = pca_model.transform([emb1])[0]
        vec2 = pca_model.transform([emb2])[0]
        
        sim_pca = float(cosine_similarity([vec1], [vec2])[0][0])
        eucl_dist = float(np.linalg.norm(vec1 - vec2))
        
        # Hybrid scoring yang seimbang
        hybrid_score = (sim_onnx * 0.70) + (sim_pca * 0.30)
        display_similarity = max(0.0, hybrid_score)
        is_match = bool(display_similarity >= SIMILARITY_THRESHOLD)
        
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