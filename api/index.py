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

# THRESHOLD DITURUNKAN SEDIKIT: Karena kita sangat memperketat fokus wajah, 
# nilai 0.45 - 0.48 adalah titik potong aman untuk kasus beda usia yang ekstrem.
SIMILARITY_THRESHOLD = 0.45

def enhance_image_for_old_photos(img_array):
    """
    Prapemrosesan Lanjutan: Denoise -> CLAHE -> Sharpening.
    Dirancang khusus untuk memunculkan struktur T-Zone pada foto lawas/anak-anak.
    """
    # 1. Denoising (Menghilangkan bintik noise dari kamera/scan lama)
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 5, 5, 7, 15)
    
    # 2. CLAHE pada L-channel (Meningkatkan kontras tanpa merusak warna)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    contrast_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # 3. Sharpening (Mempertegas tepi mata, hidung, mulut)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened_img = cv2.filter2D(contrast_img, -1, kernel)
    
    return sharpened_img

def l2_normalize(vector):
    """Normalisasi L2 agar perhitungan cosine similarity lebih stabil."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

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
@app.route("/health")
def health():
    return {"status": "ok"}
@app.route('/api/compare', methods=['POST'])
@app.route('/compare', methods=['POST'])
def compare_faces(any_path=None):
    if not is_model_ready:
        return jsonify({"error": "Model PCA belum siap."}), 500

    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Dua foto dibutuhkan."}), 400
        
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    def get_deepface_embedding(file_obj, is_old_photo=False):
        if file_obj.filename == '':
            return None, "Berkas kosong."
            
        img_bytes = file_obj.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, "Gagal membaca format gambar."
            
        if is_old_photo:
            img = enhance_image_for_old_photos(img)
            
        try:
            # UPGRADE LOGIKA PENGENALAN
            results = DeepFace.represent(
                img_path=img, 
                model_name="ArcFace", 
                detector_backend="retinaface",
                enforce_detection=True, 
                align=True,
                # expand_percentage=0 memaksa model hanya melihat T-Zone inti 
                # (mengabaikan pipi tembam anak/rahang yang belum tumbuh)
                expand_percentage=0 
            )
            # Terapkan L2 Normalization pada output murni
            raw_embedding = results[0]["embedding"]
            norm_embedding = l2_normalize(raw_embedding)
            return norm_embedding, None
        except ValueError:
            return None, "Wajah tidak terdeteksi. Pastikan pencahayaan cukup dan wajah tidak tertutup."

    emb1, err1 = get_deepface_embedding(file1, is_old_photo=True)
    if err1: return jsonify({"error": err1}), 400
        
    emb2, err2 = get_deepface_embedding(file2, is_old_photo=False)
    if err2: return jsonify({"error": err2}), 400
        
    try:
        sim_arcface = float(cosine_similarity([emb1], [emb2])[0][0])

        vec1 = pca_model.transform([emb1])[0]
        vec2 = pca_model.transform([emb2])[0]
        
        sim_pca = float(cosine_similarity([vec1], [vec2])[0][0])
        eucl_dist = float(np.linalg.norm(vec1 - vec2))
        
        # PENGURANGAN BOBOT PCA: Model PCA standar sangat buruk untuk lintas-usia.
        # Kita ambil 90% keputusan dari ArcFace yang sudah dimaksimalkan, dan hanya 10% dari PCA.
        hybrid_score = (sim_arcface * 0.90) + (sim_pca * 0.10)
        
        display_similarity = max(0.0, hybrid_score)
        is_match = bool(display_similarity >= SIMILARITY_THRESHOLD)
        
        if display_similarity >= 0.60: confidence = "Sangat Tinggi"
        elif display_similarity >= 0.50: confidence = "Tinggi"
        elif display_similarity >= SIMILARITY_THRESHOLD: confidence = "Sedang"
        else: confidence = "Rendah"
            
        return jsonify({
            "success": True,
            "is_match": is_match,
            "similarity": round(display_similarity, 4),
            "distance": round(eucl_dist, 3),
            "threshold": SIMILARITY_THRESHOLD,
            "confidence": confidence,
            "details": {
                "arcface_score": round(sim_arcface, 4),
                "pca_score": round(sim_pca, 4)
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Kesalahan komputasi: {str(e)}"}), 500