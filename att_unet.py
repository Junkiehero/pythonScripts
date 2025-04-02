import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from multiprocessing import freeze_support

# Model hiperparametreleri
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri yollarını değişkenlere atıyoruz
IMAGE_DIR = "./dataset"  # Görüntülerin bulunduğu dizin
MASK_DIR = "./dataset"  # Maskelerin bulunduğu dizin

# Attention U-Net Modeli
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

def upsample_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )

class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()
        self.encoder1 = conv_block(3, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = upsample_block(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        
        self.upconv3 = upsample_block(512, 256)
        self.decoder3 = conv_block(512, 256)
        
        self.upconv2 = upsample_block(256, 128)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = upsample_block(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = nn.MaxPool2d(kernel_size=2)(e1)
        
        e2 = self.encoder2(p1)
        p2 = nn.MaxPool2d(kernel_size=2)(e2)
        
        e3 = self.encoder3(p2)
        p3 = nn.MaxPool2d(kernel_size=2)(e3)
        
        e4 = self.encoder4(p3)
        p4 = nn.MaxPool2d(kernel_size=2)(e4)
        
        b = self.bottleneck(p4)
        
        u4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat([e4, u4], dim=1))
        
        u3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat([e3, u3], dim=1))
        
        u2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat([e2, u2], dim=1))
        
        u1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat([e1, u1], dim=1))
        
        out = self.sigmoid(self.final_conv(d1))
        return out

# Modeli oluştur ve GPU'ya taşı
model = AttentionUNet().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# Veri seti sınıfı
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Dönüşümler ve veri yükleyiciler
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

train_dataset = CustomDataset(IMAGE_DIR, MASK_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Eğitim döngüsü
def train(model, dataloader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

# Ana fonksiyon
def main():
    train(model, train_loader, criterion, optimizer)

if __name__ == '__main__':
    freeze_support()  # Windows için gerekli
    main()  # Ana fonksiyonu başlat

# Modeli kaydet
torch.save(model.state_dict(), "final_model.pth")
