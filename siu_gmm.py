import numpy as np
import cv2
import os
import time
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

# **GMM ile anomaly map çıkarma fonksiyonu**
def generate_anomaly_map(image, num_components=3):
    pixels = image.reshape(-1, 1)  # Piksel değerlerini vektörize et
    
    gmm = GaussianMixture(n_components=num_components, covariance_type="full", random_state=42)
    gmm.fit(pixels)

    scores = gmm.score_samples(pixels)
    scores = scores.reshape(image.shape)

    threshold = np.percentile(scores, 5)  # En düşük 5%'lik pikseller anomali
    anomaly_map = (scores < threshold).astype(np.uint8)

    return anomaly_map

# **Paralel çalışacak fonksiyon**
def process_image(image_path, output_dir):
    image = cv2.imread(image_path, 0)  # Grayscale olarak yükle
    if image is None:
        print(f"Hata: {image_path} okunamadı!")
        return None
    
    anomaly_map = generate_anomaly_map(image)

    # **Kaydet**
    filename = os.path.basename(image_path).split('.')[0] + "_anomaly.npy"
    np.save(os.path.join(output_dir, filename), anomaly_map)

    return filename  # İşlenen dosya ismini döndür

# **Ana çalışma fonksiyonu**
def process_all_images(input_dir, output_dir, num_workers=4):
    start_time = time.time()
    
    # **Tüm dosyaları al (PNG ve JPG desteği)**
    image_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) 
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"{len(image_files)} görüntü işleniyor...")

    # **Joblib ile paralel işleme**
    results = Parallel(n_jobs=num_workers)(
        delayed(process_image)(image_path, output_dir) for image_path in image_files
    )

    end_time = time.time()
    print(f"Tamamlandı! Toplam süre: {end_time - start_time:.2f} saniye")

# **Kullanım**
input_directory = "./tryy/"  # 📂 BT görüntülerinin olduğu klasör
output_directory = "./try/"  # 📂 Çıktıların kaydedileceği klasör
os.makedirs(output_directory, exist_ok=True)

process_all_images(input_directory, output_directory, num_workers=8)  # İşlemci çekirdek sayısına göre ayarla
