import os
import cv2
import numpy as np

# Gaussian Blur fonksiyonu
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Görselleri yükleme fonksiyonu
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (256, 256))  # Görselleri 256x256 boyutuna getir
                images.append(img)
    return images

# Blur uygula ve kaydet
def apply_blur_and_save(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Çıktı klasörü oluştur

    # Görselleri yükle
    images = load_images_from_folder(image_folder)

    # Her bir görsele blur uygula ve kaydet
    for i, image in enumerate(images):
        # Gaussian Blur uygula
        blurred_image = apply_gaussian_blur(image)

        # Kaydet
        output_path = os.path.join(output_folder, f'blur_{i}.jpg')
        cv2.imwrite(output_path, blurred_image)
        print(f"Blurred image saved: {output_path}")

# Ana kod
if __name__ == "__main__":
    image_folder = '../output_images'  # Görsellerin bulunduğu klasör
    output_folder = '../blur_images'  # Blur uygulanmış görsellerin kaydedileceği klasör

    # Blur uygula ve kaydet
    apply_blur_and_save(image_folder, output_folder)