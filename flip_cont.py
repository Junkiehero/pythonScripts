import cv2
import numpy as np
import os

# Flip fonksiyonu (Yatay ve Dikey)
def apply_flip(image):
    flip_type = np.random.choice([0, 1, -1])  # Yatay (0), Dikey (1), Her iki yön (-1)
    flipped_image = cv2.flip(image, flip_type)
    return flipped_image

# Contrast Adjustment fonksiyonu
def apply_contrast_adjustment(image, alpha=1.5, beta=0):  
    # alpha kontrast artırma katsayısı (1.0 = orijinal, >1 = daha fazla kontrast)
    # beta parlaklık artırma katsayısı (0 = orijinal, >0 = daha parlak)
    contrast_adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrast_adjusted_image

# Görselleri yükleme fonksiyonu
def load_images_from_folder(folder_path):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (256, 256))  # Görselleri 256x256 boyutuna getir
                images.append(img)

                # Etiketleme (örnek: 'H' -> 0, 'hasta' -> 1 gibi)
                if 'H' in filename:  # Örneğin 'H' dosya adını içeriyorsa sağlıklı
                    labels.append(0)  # Sağlıklı
                else:
                    labels.append(1)  # Hasta
                filenames.append(filename)
    return images, labels, filenames

# Görselleri işleme ve kaydetme fonksiyonu
def process_and_save_images(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Çıktı klasörünü oluştur

    # Görselleri yükle
    images, labels, filenames = load_images_from_folder(image_folder)

    # Her bir görsel için işlemler
    for i, (image, label) in enumerate(zip(images, labels)):
        # Flip uygula
        flipped_image = apply_flip(image)

        # Kontrast ayarlaması uygula
        contrast_adjusted_image = apply_contrast_adjustment(image)

        # Flip + Contrast adjustment uygula
        flipped_contrast_adjusted_image = apply_contrast_adjustment(flipped_image)

        # Kaydetme
        filename = filenames[i] if i < len(filenames) else f"synthetic_{i}"

        # Orijinal
        cv2.imwrite(os.path.join(output_folder, f'{filename}_original.jpg'), image)
        # Flip
        cv2.imwrite(os.path.join(output_folder, f'{filename}_flip.jpg'), flipped_image)
        # Contrast Adjustment
        cv2.imwrite(os.path.join(output_folder, f'{filename}_contrast_adjusted.jpg'), contrast_adjusted_image)
        # Flip + Contrast Adjustment
        cv2.imwrite(os.path.join(output_folder, f'{filename}_flip_contrast_adjusted.jpg'), flipped_contrast_adjusted_image)

        print(f"Processed and saved: {filename}")

# Ana kod
if __name__ == "__main__":
    image_folder = '../../Yolo Model/Yolo v10/H_Mix'  # İşlenecek görsellerin olduğu klasör
    output_folder = '../../Yolo Model/Yolo v10/H_Last'  # Tüm çıktılar burada saklanacak

    # Görselleri işleyip kaydet
    process_and_save_images(image_folder, output_folder)
