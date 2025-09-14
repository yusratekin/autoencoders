Amaç: FashionMNIST (28×28 gri tonlu kıyafet görselleri) üzerinde autoencoder ile veri sıkıştırma (latent temsil) ve yeniden üretim (reconstruction) yapıyoruz.

Nasıl: Bir encoder görüntüyü 784 boyuttan 64 boyuta sıkıştırıyor; decoder bu 64’ten tekrar 784’e çıkarıp 28×28 resme dönüştürüyor.

Eğitim: Çıkış ile giriş arasındaki farkı MSELoss ile ölçüp Adam ile ağırlıkları güncelliyoruz. EarlyStopping ile iyileşme durunca erken durduruyoruz.

Değerlendirme: Testten bir batch alıp orijinal ve yeniden oluşturulan görüntüleri çiziyoruz, SSIM hesaplayıp görsel benzerliği sayısallaştırıyoruz.
