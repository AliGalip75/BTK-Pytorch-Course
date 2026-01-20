'''
rnn: Tekrar Eden Sinir Ağları (Recurrent Neural Networks - RNN)
'''

# %% Veriyi içe aktar
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def generate_data(seq_length=50, num_samples=1000):
    # seq_length ->	Modele verilecek girdi dizisinin uzunluğu
    # num_samples -> Sinüs dalgasında üretilecek toplam nokta sayısı
    X = np.linspace(0, 100, num_samples)
    y = np.sin(X)  # Sinüs dalgası
    sequences = [] # Girdi dizileri
    targets = [] # Tahmin edilecek değer
    for i in range(len(X) - seq_length):
        sequences.append(y[i:i+seq_length]) # seq_length=5 için [ y0, y1, y2, y3, y4 ]
        targets.append(y[i+seq_length]) # input dizisinin sonraki değeri, y5
    
    plt.figure(figsize=(10, 4))
    plt.plot(X, y, label="Sinüs Dalgası")
    plt.xlabel("Zaman")
    plt.ylabel("Genlik")
    plt.title("Sinüs Dalgası")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return np.array(sequences), np.array(targets)



sequences, targets = generate_data()



# %% rnn modeli tanımla

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        super(RNN, self).__init__()
        # RNN’in hafıza kapasitesini belirlemek.
        # Kaç boyutlu geçmiş bilgi taşıyacağını söyler.
        self.hidden_size = hidden_size 
        
        # üst üste kaç RNN katmanı olduğunu söyler
        self.num_layers = num_layers
        
        # Zaman bağımlı ilişkiyi öğrenen çekirdek yapı
        # Her zaman adımında:
        #   önceki gizli durumu
        #   mevcut girdiyi
        # birleştirip yeni bir gizli durum üretir.
        # batch_first=True
        # Veriyi (batch, time, feature) düzeninde alabilmek için
        # (sadece tensor şekli meselesi)
        
        # input_size: Her zaman adımında RNN’e giren özellik sayısıdır.
        # Sadece sıcaklık → input_size = 1
        # Sıcaklık + nem + basınç → input_size = 3
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
       
       # RNN’in öğrendiği zamansal temsili
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, h_n = self.rnn(x) 
        # out: (batch_size, seq_len, hidden_size), RNN’in zaman boyunca ürettiği ara temsiller
        # h_n: (num_layers, batch_size, hidden_size), Her katmanın en son zaman adımındaki hafızası
        out = self.fc(out[:, -1, :])  # Sadece son zaman adımının çıktısını kullan
        return out

# %% rnn train et
seq_length = 50 # input dizisi uzunluğu
input_size = 1 # Her zaman adımında giren özellik sayısı
hidden_size = 16 # RNN’in hafıza kapasitesi
output_size = 1 # Tahmin edilecek değer sayısı
num_layers = 1 # RNN katmanı sayısı
epochs = 20 # Eğitim döngüsü sayısı
batch_size = 32 # Her eğitim adımında işlenecek örnek sayısı
learning_rate = 0.001 # Öğrenme hızı

X, y = generate_data(seq_length=seq_length, num_samples=2000)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # (num_samples, seq_length, 1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) # (num_samples, 1)

dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss() # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs) # Model tahminleri
        loss = criterion(outputs, targets) # Kayıp hesaplama
        
        optimizer.zero_grad() # Gradients sıfırlama
        loss.backward() # Backpropagation
        optimizer.step() # Ağırlık güncelleme
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# %% rnn test et

# Test verisi oluştur
x_test = np.linspace(100, 110, seq_length).reshape(1, -1)  # (1, seq_length), ilk test dizisi
y_test = np.sin(x_test)  # (1, seq_length), gerçek değerler

x_test2 = np.linspace(120, 150, seq_length).reshape(1, -1)  # (1, seq_length), ikinci test dizisi
y_test2 = np.sin(x_test2)  # (1, seq_length), gerçek değerler

# From numpy to torch tensor
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)  # (1, seq_length, 1)
x_test2_tensor = torch.tensor(x_test2, dtype=torch.float32).unsqueeze(-1)  # (1, seq_length, 1)

# Modeli değerlendirme moduna al
model.eval()

# 1️⃣ Test dizisi oluştur (sinüs DEĞERLERİ)
t_test = np.linspace(100, 110, seq_length)
x_test = np.sin(t_test).reshape(1, -1)  # (1, seq_length)

# Gerçek bir sonraki değer
true_next = np.sin(110)

# Torch tensor
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)

# 2️⃣ Tahmin
with torch.no_grad():
    predicted = model(x_test_tensor).item()

print(f"Gerçek değer : {true_next:.4f}")
print(f"Tahmin       : {predicted:.4f}")

# 3️⃣ Görselleştirme
plt.figure(figsize=(10, 4))

# Girdi sinüs dizisi
plt.plot(t_test, x_test.flatten(), marker='o', label='Input Sinüs')

# Gerçek ve tahmin edilen noktalar
plt.scatter(110, true_next, color='green', s=80, label='True Next')
plt.scatter(110, predicted, color='red', s=80, label='Predicted Next')

plt.xlabel("Zaman")
plt.ylabel("Genlik")
plt.title("RNN Sonraki Sinüs Değeri Tahmini")
plt.legend()
plt.grid(True)
plt.show()