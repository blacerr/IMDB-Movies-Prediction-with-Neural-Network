# IMDB Movies Prediction

Bu proje, IMDB filmleri veri kümesi kullanarak film puanlarını tahmin etmek için bir sinir ağı modeli oluşturur. Model, PyTorch kütüphanesi kullanılarak eğitilmiştir.
## İçindekiler
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Açıklaması](#proje-açıklaması)
- [Veri Seti](#veri-seti)
- [Model Yapısı](#model-yapısı)
- [Eğitim ve Değerlendirme](#eğitim-ve-değerlendirme)
- [Sonuçlar](#sonuçlar)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

## Kurulum

Gerekli paketleri yüklemek için aşağıdaki adımları izleyin:

1. Gerekli Python paketlerini yükleyin:
    
bash
    pip install pandas scikit-learn torch
    
- `pandas` - Veri işleme ve analiz için
- `torch` - Derin öğrenme ve sinir ağı oluşturma için
- `sklearn` - Veriyi ön işleme ve model değerlendirme için

2. Proje dosyasını klonlayın veya indirin.

## Kullanım

Proje dosyasını çalıştırmak için:

1. main.py dosyasını çalıştırın:
    
bash
    python main.py

## Proje Açıklaması

1. **Veri Yükleme ve Ön İşleme**
   - Veriler CSV dosyasından yüklenir.
   - Belirtilen sütunlar sayısal değerlere dönüştürülür.
   - Eksik değerler ve gereksiz sütunlar temizlenir.
   - Özellikler ve hedef değişken ayrılır.

2. **Veri Bölme ve Ölçeklendirme**
   - Veri eğitim ve test setlerine ayrılır.
   - Özellikler standartlaştırılır.

3. **PyTorch Veri Kümesi ve Veri Yükleyicileri**
   - Veriler PyTorch tensorlerine dönüştürülür.
   - Eğitim ve test veri kümesi ve veri yükleyicileri oluşturulur.

4. **Model Tanımlama ve Eğitim**
   - Bir sinir ağı modeli tanımlanır ve eğitilir.
   - Modelin performansı epoch başına kayıp fonksiyonu ile izlenir.

5. **Model Değerlendirme ve Tahmin**
   - Model test verileri üzerinde değerlendirilir ve doğruluk metrikleri hesaplanır.
   - Tüm veri kümesi üzerinde tahminler yapılır ve sonuçlar CSV dosyasına kaydedilir.

## Veri Seti

- `imdb_movies.csv`: IMDB filmleri hakkında çeşitli bilgiler içeren CSV dosyası.

## Model Yapısı

- **Giriş Katmanı:** 3 özellik (budget_x, revenue, country)
- **Gizli Katman 1:** 64 nöron
- **Gizli Katman 2:** 32 nöron
- **Çıkış Katmanı:** 1 nöron (puan tahmini)

## Eğitim ve Değerlendirme

- **Kayıp Fonksiyonu:** Ortalama Kare Hata (MSE)
- **Optimizasyon Yöntemi:** Adam
- **Epoch Sayısı:** 5000

## Sonuçlar

- Modelin tahminleri `Sonuc.csv` dosyasına kaydedilir.
- Sayısal değerler, orijinal kategorik değerlerle geri dönüştürülür.

## İletişim

Proje ile ilgili sorularınız için [uzayk204@gmail.com](mailto:uzayk204@gmail.com) adresinden iletişime geçebilirsiniz.

## Katkıda Bulunma

Katkılarınızı memnuniyetle kabul ediyoruz! Lütfen katkıda bulunmadan önce [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını okuyun.

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.


