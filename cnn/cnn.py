"""
Problem: CIFAR-10 veri seti kullanarak basit bir Konvolüsyonel Sinir Ağı (CNN) modeli oluşturun, eğitin ve test edin.
"""
# %% Import libraries
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load dataset
def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # rgb kanalları için normalizasyon
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


# %% Visualize dataset
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def get_sample_images(train_loader, n=4):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    return images, labels

def visualize(n):
    train_loader, test_loader = get_data_loaders()
    images, labels = get_sample_images(train_loader)
    plt.figure(figsize=(10,4))
    for i in range(n):
        plt.subplot(1, n, i+1) # (satır, sütun, indeks)
        imshow(images[i])
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.show()


# %% Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # (kanal sayısı, filtre sayısı, kernel boyutu, padding)
        # Padding ile görselin dışına sıfır ekleyerek boyut kaybını önlüyoruz
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # (kernel boyutu, stride)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10) # CIFAR-10 için 10 sınıf
        
        # image 3x32x32 -> conv1 32x32x32 -> relu -> pool 32x16x16
        # -> conv2 64x16x16 -> relu -> pool 64x8x8 -> flatten 64*8*8

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # flatten, fully connected katmanlara geçiş
        # 64 kanal, her biri 8x8 boyutunda
        # -1 batch size için otomatik hesaplama
        # "Elimde kaç tane resim varsa o kadar satır olsun, ama her satırın uzunluğu kesinlikle 4096 olsun."
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
model = CNN().to(device)

# %% Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# %% Train the model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    print('Starting Training...')
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}], Loss: {avg_loss:.4f}')
    print('Finished Training')
    # Kayıp grafiğini çiz
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend() # sağ üstteki label göstergesi
    plt.show()

# %% Evaluate the model

def evaluate_model(model, test_loader, dataset_type='Test'):
    model.eval()
    correct = 0 # Doğru tahmin sayısı
    total = 0 # Toplam veri sayısı
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # prediction
            _, predicted = torch.max(outputs.data, 1) # en yüksek skorlu sınıfı al
            total += labels.size(0) # Toplam veri sayısını güncelle
            correct += (predicted == labels).sum().item() # Doğru tahminleri say
        print(f'{dataset_type} Accuracy of the model on the test images: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    # visualize some training images
    get_sample_images(train_loader)
    visualize(4)
    train_model(model, train_loader, criterion, optimizer, epochs=5)
    evaluate_model(model, test_loader, "Test")