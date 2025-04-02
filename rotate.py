import os
import cv2
import numpy as np

# 🔹 Giriş ve çıkış klasör yollarını tanımla
input_folder = "../../Yolo Model/Yolo v10/H"   # Orijinal resimlerin olduğu klasör
output_folder = "../../Yolo Model/Yolo v10/H_R"  # Döndürülmüş resimlerin kaydedileceği klasör

# 🔹 Kaydedilecek rotasyon açılarının listesi
angles = [90, 180, 270]  # İstediğin açılar eklenebilir

# 🔹 Çıkış klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 🔹 Klasördeki tüm resimleri işle
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ Hata: {filename} yüklenemedi.")
            continue

        print(f"✅ {filename} işleniyor...")

        for angle in angles:
            # 🔹 Resmi döndür
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))

            # 🔹 Yeni dosya adını oluştur ve kaydet
            new_filename = f"{os.path.splitext(filename)[0]}_rot{angle}.jpg"
            output_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(output_path, rotated_img)
            print(f"   ➡ {new_filename} kaydedildi.")

print("🎉 İşlem tamamlandı! Tüm resimler döndürüldü ve kaydedildi.")
