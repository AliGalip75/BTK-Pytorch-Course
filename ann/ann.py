'''
MNIST veriseti ile rakam tanıma için yapay sinir ağı modeli.
'''

import numpy as np
from torch import nn, optim # -> Sinir ağı ve optimizasyon için
import torch.nn.functional as F
import torch # -> Tensor işlemleri için
import torchvision # -> Görüntü işleme ve ön eğitimli modeller için
import torchvision.transforms as transforms # -> Görüntü dönüşümleri için
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri seti yükleme ve ön işleme
def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(), # Görüntüleri tensöre dönüştürür ve piksel değerlerini [0,1] aralığına ölçeklendirir
        transforms.Normalize((0.1307,), (0.3081,)) # Ortalama ve standart sapma ile normalleştirir ve veri dağılımını iyileştirir
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    

    # veri yükleyiciler oluşturma
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

# train_loader, test_loader = get_data_loaders()

# veri görselleştirme
def visualize_data(loader, n):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    print(images[0].shape)
    print(images[0].squeeze().shape)
    
    rows = (n + 5) // 6
    cols = 6 if n>=6 else n
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2.5)) # n farklı görüntü için alt grafik oluşturma
    
    # Axes dizisini düzleştirme (2D'den 1D'ye)
    axes = axes.flatten()
    
    for i in range(n):
        # images[i] -> (1, 28, 28), squeeze() ile -> (28, 28)
        axes[i].imshow(images[i].squeeze(), cmap="gray") # gri tonlamalı görüntüleri gösterme
        axes[i].set_title(f"Label: {labels[i].item()}") # etiketleri başlık olarak ekleme
        axes[i].axis("off")
    plt.show()
        
# visualize_data(train_loader, 5)

# model tanımlama

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # 28x28 görüntüleri 784 boyutlu vektörlere dönüştürür
        self.fc1 = nn.Linear(28 * 28, 512) # İlk tam bağlantılı katman, (input_size=784, output_size=512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU() # ReLU aktivasyon fonksiyonu
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # initial x shape: (batch_size, 1, 28, 28)
        x = self.flatten(x) # x shape: (batch_size, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# model = NeuralNetwork().to(device)

# Kayıp fonksiyonu ve optimizatör tanımlama
# criterion = nn.CrossEntropyLoss() # Çok sınıflı sınıflandırma için çapraz entropi kaybı
# optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizatörü

# model eğitimi
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train() # Eğitim moduna geçiş
    train_losses = [] # Her epoch sonu kayıplarını saklamak için liste
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad() # Önceki adımın gradyanlarını sıfırla
            
            predictions = model(images) # İleri yayılım
            loss = criterion(predictions, labels) # Kayıp hesaplama
            loss.backward() # Geri yayılım
            optimizer.step() # Ağırlıkları güncelle
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    # loss grafiği
    plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-' )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()


# train_model(model, train_loader, criterion, optimizer, epochs=5)

# model değerlendirme
def evaluate_model(model, test_loader):
    model.eval() # Değerlendirme moduna geçiş
    correct = 0
    total = 0
    with torch.no_grad(): # Geri yayılım yapılmayacak
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # İleri yayılım
            
            # Doğruluk oranını hesaplama
            _, predicted = torch.max(outputs.data, 1) # (skor, sınıf indeksi)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
# evaluate_model(model, test_loader)

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    visualize_data(train_loader, 5)
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, epochs=5)
    evaluate_model(model, test_loader)