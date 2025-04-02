import os
import cv2
import numpy as np

# Scale (Ölçeklendirme) fonksiyonu
def apply_scale(image, scale_factor=1.2):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    # Görseli ölçeklendir
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Görseli 256x256 boyutuna getir
    final_image = cv2.resize(resized_image, (256, 256), interpolation=cv2.INTER_LINEAR)
    return final_image

# Görselleri yükleme fonksiyonu
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

# Scale uygula ve kaydet
def apply_scale_and_save(image_folder, output_folder, scale_factor=1.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Çıktı klasörü oluştur

    # Görselleri yükle
    images = load_images_from_folder(image_folder)

    # Her bir görsele scale uygula ve kaydet
    for i, image in enumerate(images):
        # Scale uygula
        scaled_image = apply_scale(image, scale_factor=scale_factor)

        # Kaydet
        output_path = os.path.join(output_folder, f'scale_{i}.jpg')
        cv2.imwrite(output_path, scaled_image)
        print(f"Scaled image saved: {output_path}")

# Ana kod
if __name__ == "__main__":
    image_folder = '../output_images'  # Görsellerin bulunduğu klasör
    output_folder = '../scaled_images'  # Scale uygulanmış görsellerin kaydedileceği klasör

    # Scale uygula ve kaydet
    apply_scale_and_save(image_folder, output_folder, scale_factor=1.2)