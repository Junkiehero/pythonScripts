import cv2
import numpy as np
import os

# Maskeleri kaydedeceğimiz klasör
masks_dir = 'G:/My Drive/GAN/masks_vit/'

# Eğer klasör yoksa oluştur
if not os.path.exists(masks_dir):
    os.makedirs(masks_dir)

# Görsellerin bulunduğu klasör (örneğin './data' gibi)
image_dir = 'G:/My Drive/GAN/images/'

# Görsellerin listesi
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

# Görsel boyutları
img_height, img_width = 256, 256

# Görseldeki kırmızı tonlarını bulmak için maskeyi çıkarma işlemi
def extract_mask(image_path):
    # Görseli yükle
    image = cv2.imread(image_path)

    # Görseli HSV formatına dönüştür
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Kırmızı tonlarını bulmak için HSV değerlerini alacağız.
    lower_red1 = np.array([0, 120, 120])  # Alt sınır (düşük tonlar)
    upper_red1 = np.array([10, 255, 255])  # Üst sınır (düşük tonlar)

    lower_red2 = np.array([170, 120, 120])  # Alt sınır (yüksek tonlar)
    upper_red2 = np.array([180, 255, 255])  # Üst sınır (yüksek tonlar)

    # Maskeleri uygula
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Maskeleri birleştir
    mask = cv2.bitwise_or(mask1, mask2)

    return mask

# Maskeleri çıkar ve kaydet
for image_file in image_files:
    # Görseli tam yoluyla belirt
    image_path = os.path.join(image_dir, image_file)

    # Maskeyi çıkar
    mask = extract_mask(image_path)

    # Maskeyi kaydet
    mask_filename = os.path.join(masks_dir, image_file)
    cv2.imwrite(mask_filename, mask)

    print(f"Mask saved for {image_file}")
