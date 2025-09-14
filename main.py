"""
problem tanimi : veri sikistitmasi -> autoencoders
veri : FashionMINST
"""

# import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np


# %% veri seti yukleme ve on isleme

transform = transforms.Compose([transforms.ToTensor()])  # goruntuyu tensore cevir

# egitim ve test veri setini indir ve yukle
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# batch size
batch_size = 64

# egitim ve test veri yukleyicileri olusturma
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)


# %% auto encoders gelistirme

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),               # 28x28 -> 784 vektor
            nn.Linear(28*28, 256),      # tam bagli katman : 784 -> 256
            nn.ReLU(),                  # aktivasyon fonksiyonu
            nn.Linear(256, 64),         # tam bagli katman : 256 -> 64 (latent)
            nn.ReLU()
        )

        # decoder 
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),         # tam bagli katman : 64 -> 256
            nn.ReLU(),                  # aktivasyon fonksiyonu
            nn.Linear(256, 28*28),      # tam bagli katman : 256 -> 784
            nn.Sigmoid(),               # 0-1 araliginda tutmak icin (girdi ToTensor ile 0-1)
            nn.Unflatten(1, (1, 28, 28))# tek boyutlu ciktiyi tekrar 28x28 yapar
        )

    def forward(self, x):
        encoded = self.encoder(x)        # giris verisini kodlar
        decoded = self.decoder(encoded)  # kodlanmis veriyi tekrar goruntuye donusturur
        return decoded


# %% callback: early stopping 

class EarlyStopping:   # erken durdurma (callback)
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience        # kac epoch boyunca gelisme olmazsa durdur
        self.min_delta = min_delta      # kayiptaki minimum iyilesme miktari
        self.best_loss = None           # en iyi (en dusuk) kayip degeri
        self.counter = 0                # gelisme olmayan ardil epoch sayaci

    def __call__(self, loss):
        # gelisme var mi kontrol et
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0            # gelisme varsa sayaci sifirla
        else:
            self.counter += 1           # gelisme yoksa sayaci artir

        # patience asildiysa durdurma sinyali ver
        if self.counter >= self.patience:
            return True
        return False


# %% model training 

# hyperparameters
epochs = 50
learning_rate = 1e-4   # learning_rate (10 uzeri -4)

# model, loss ve optimizer tanimlayalim
model = AutoEncoder()                      # model tanimlama
criterion = nn.MSELoss()                   # kayip fonksiyonu -> MSE: ortalama kare hata
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  

early_stopping = EarlyStopping(patience=5, min_delta=0.001)   # erken durdurma objesi


# egitim fonksiyonu
def training(model, train_loader, optimizer, criterion, early_stopping, epochs):
    model.train()  # modeli egitim moduna al
    for epoch in range(epochs):  # epoch dongusu
        total_loss = 0.0         # epoch basina toplam kayip
        for inputs, _ in train_loader:
            optimizer.zero_grad()               # gradyanlari sifirla
            outputs = model(inputs)             # ileri yayilim
            loss = criterion(outputs, inputs)   # rekonstruksiyon kaybi
            loss.backward()                     # gradyanlari hesapla
            optimizer.step()                    # agirliklari guncelle
            total_loss += loss.item()           # toplam kayba ekle

        # epoch sonu ortalama kayip ve early stopping
        avg_loss = total_loss / len(train_loader)                 # epoch ortalama kayip
        print(f'Epoch {epoch+1}/{epochs}, loss: {avg_loss:.3f}')  # ekrana yazdir

        if early_stopping(avg_loss):                               # erken durdurma kontrolu
            print(f'Early stopping at epoch {epoch+1}')            # bilgi mesaji
            break


# egitimi baslat
training(model, train_loader, optimizer, criterion, early_stopping, epochs)


# %% model testing 

from scipy.ndimage import gaussian_filter  

def compute_ssim(img1, img2, sigma=1.5):
    """
    iki goruntu arasindaki benzerligi hesaplar
    """
    C1 = (0.001*255)**2    # ssim sabitlerinden bir tanesi
    C2 = (0.03*255)**2     # diger bir sabit 
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # goruntulerin ortalamalari
    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)
    
    # ortalama, varyans ve kovaryans terimleri
    mu1_sq = mu1**2
    mu2_sq = mu2**2 
    mu1_mu2 = mu1 * mu2
    
    # varyans hesabi
    sigma1_sq = gaussian_filter(img1**2, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2**2, sigma) - mu2_sq

    # kovaryans hesabi
    sigma12 = gaussian_filter(img1*img2, sigma) - mu1_mu2
    
    # ssim haritasi hesaplama
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def evaluate(model, test_loader, n_images=10):
    model.eval()      # modelimizi degerlendirme moduna al
    
    with torch.no_grad():   
        for batch in test_loader:
            inputs, _ = batch
            outputs = model(inputs)    # modelin ciktilarini uret

    # NumPy'a cevirirken grafigi kopar ve CPU'ya tasi (GPU olsa da calissin)
    inputs  = inputs.detach().cpu().numpy()     # <-- DÜZELTME
    outputs = outputs.detach().cpu().numpy()    # <-- DÜZELTME

    # subplot olusturma
    fig, axes = plt.subplots(2, n_images, figsize=(n_images, 3))  # <-- DÜZELTME: subplots
    ssim_scores = []     

    for i in range(n_images):
        img1 = np.squeeze(inputs[i])     # orijinal goruntu
        img2 = np.squeeze(outputs[i])    # yeniden olusturulmus goruntu
        
        ssim_score = compute_ssim(img1, img2)
        ssim_scores.append(ssim_score)   # ssim skorunu listeye ekle
        
        axes[0, i].imshow(img1, cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(img2, cmap='gray')
        axes[1, i].axis('off')
        
    axes[0, 0].set_title('Original')     
    axes[1, 0].set_title('Decoded image')
    plt.show()
    
    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM: {avg_ssim:.4f}")

evaluate(model, test_loader, n_images=10)  
