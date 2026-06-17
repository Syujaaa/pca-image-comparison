import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.decomposition import PCA

def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, 'FGNET', 'images')
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Folder '{dataset_dir}' tidak ditemukan!")
        return

    embeddings_data = []
    labels = [] 
    print("Memulai ekstraksi fitur wajah menggunakan DeepFace (ArcFace)...")
    
    valid_images = 0
    
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dataset_dir, filename)
            person_name = filename[:3] 
            
            try:
                # UPGRADE: Menggunakan model 'ArcFace' yang sangat tangguh terhadap perubahan usia
                results = DeepFace.represent(img_path, model_name="ArcFace", enforce_detection=True)
                embedding = results[0]["embedding"]
                
                embeddings_data.append(embedding)
                labels.append(person_name)
                valid_images += 1
                print(f"Berhasil memproses: [Subjek {person_name}] -> {filename}")
                
            except ValueError:
                print(f"Gagal deteksi wajah: [Subjek {person_name}] -> {filename}")

    if valid_images < 2:
        print("Dataset terlalu sedikit untuk melatih PCA.")
        return

    X = np.array(embeddings_data)
    unique_people = len(set(labels))
    
    print(f"\nTotal dataset siap latih: {valid_images} gambar dari {unique_people} subjek unik.")
    print("Melatih model PCA untuk mereduksi dimensi ArcFace...")
    
    # Mempertahankan 95% informasi penting dari 512 dimensi ArcFace
    pca = PCA(n_components=0.95) 
    pca.fit(X)
    
    print(f"Model PCA berhasil dilatih! Komponen utama yang disimpan: {pca.n_components_}")

    model_data = {
        "pca_model": pca
    }
    
    api_dir = os.path.join(current_dir, 'api')
    if not os.path.exists(api_dir):
        os.makedirs(api_dir)
        
    output_filename = os.path.join(api_dir, 'model_wajah.pkl')
    with open(output_filename, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\nSelesai! Model hybrid ArcFace+PCA disimpan di '{output_filename}'.")

if __name__ == "__main__":
    train_model()