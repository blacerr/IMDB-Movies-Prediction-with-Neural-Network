import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


# CUDA'nın kullanılabilirliğini kontrol et
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Kullanilan Cihaz: {device}')

# CSV dosyasını oku
data = pd.read_csv('imdb_movies.csv')

# Dönüştürülecek sütunlar
columns_to_convert = ['names', 'country']

# LabelEncoder nesnelerini saklamak için bir sözlük
label_encoders = {}

# Belirtilen sütunları seçme ve dönüştürme
for column in columns_to_convert:
    if (column in data.columns):  # Sütun veri setinde varsa
        label_encoder = LabelEncoder()  # LabelEncoder nesnesi oluştur
        data[column] = label_encoder.fit_transform(data[column])  # Sütunu sayısal değerlere dönüştür
        label_encoders[column] = label_encoder  # Encoder'ı sakla

# NaN (eksik) değerleri atma
data = data.dropna()

# Gereksiz sütunları çıkarma
data =data.drop(['date_x', 'genre', 'overview', 'crew', 'orig_title', 'status', 'orig_lang'], axis=1)

# Özellikleri ve hedef değişkeni seçme
features = data[['budget_x', 'revenue', 'country']]
target = data['score']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Verileri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PyTorch tensorlerine dönüştürme
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# PyTorch veri kümeleri ve veri yükleyicileri oluşturma
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Modeli tanımlama
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(3, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

model = NeuralNetwork().to(device)

# Kayıp fonksiyonu ve optimizasyon yöntemi belirleme
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Modeli eğitme
epochs = 5000
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Veriyi GPU'ya taşı
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Modeli değerlendirme ve tahminleri kaydetme
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Veriyi GPU'ya taşı
        outputs = model(inputs)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

    # Doğruluk metrikleri
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f'Ortalama Mutlak Hata: {mae}')
    print(f'Ortalama Kare Hata: {mse}')



# Tüm veri kümesi için tahminler yapma
all_data_scaled = scaler.transform(features)
all_data_tensor = torch.tensor(all_data_scaled, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    all_predictions = model(all_data_tensor).cpu().numpy()

# Tahminleri orijinal veri çerçevesine ekleme
data['Tahminler'] = pd.Series(all_predictions.flatten(), index=features.index)

# LabelEncoder nesneleri ile sayısal değerleri geri dönüştürme
for column, encoder in label_encoders.items():
    data[column] = encoder.inverse_transform(data[column].astype(int))
# Sonuçları kaydetme
data.to_csv('Sonuc.csv', index=False)
