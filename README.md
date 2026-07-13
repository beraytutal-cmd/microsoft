# Akıllı Gıda Tedarik Zinciri — Talep Tahmini & Risk Analizi

> Microsoft AI Skills Challenge 2026 | Staj Projesi
> Geliştirici: Beray Tutal

---

## Proje Özeti

Bu proje, perakende gıda sektöründe **Facebook Prophet** tabanlı zaman serisi tahminlemesi kullanarak stoksuz kalma ve israf/aşırı stok risklerini önceden tespit etmeyi amaçlamaktadır.

Gıda ürünlerinin raf ömrü kısa olduğundan, doğru stok yönetimi hem müşteri memnuniyeti hem de maliyet optimizasyonu açısından kritik öneme sahiptir. Bu proje, veri odaklı bir yaklaşımla bu dengeyi sağlamayı hedefler.

---

## Problem Tanımı

Perakende gıda zincirlerinde iki kritik risk bulunmaktadır:

| Risk | Tanım | Sonuç |
|------|-------|-------|
| **Stoksuz Kalma** | Tahmini talep > Mevcut stok | Müşteri kaybı, satış kaybı |
| **Aşırı Stok / İsraf** | Stok > 2x günlük satış | Bozulma, israf, gereksiz maliyet |

Bu projede her iki risk de otomatik olarak hesaplanmakta ve görselleştirilmektedir.

---

## Veri Seti

- **Kaynak:** [Retail Store Inventory Forecasting Dataset (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Boyut:** 73.100 kayıt, 15 sütun
- **Zaman Aralığı:** 2022-01-01 — 2023-12-31 (2 yıl)
- **Mağazalar:** S001, S002, ... (birden fazla)
- **Ürünler:** P0001 — P0020 (20 farklı ürün)
- **Kategoriler:** Groceries, Toys, Electronics, Furniture, Clothing
- **Eksik Veri:** Yok (tüm sütunlarda 0 null)

### Sütunlar

| Sütun | Tip | Açıklama |
|-------|-----|----------|
| Date | tarih | Kayıt tarihi |
| Store ID | string | Mağaza kimliği |
| Product ID | string | Ürün kimliği |
| Category | string | Ürün kategorisi |
| Region | string | Coğrafi bölge (North/South/East/West) |
| Inventory Level | int | Günlük stok miktarı |
| Units Sold | int | Günlük satış miktarı |
| Units Ordered | int | Günlük sipariş miktarı |
| Demand Forecast | float | Sistem tahmini talep |
| Price | float | Birim fiyatı ($) |
| Discount | int | İndirim yüzdesi |
| Weather Condition | string | Hava durumu |
| Holiday/Promotion | int | Tatil/promosyon (0/1) |
| Competitor Pricing | float | Rakip fiyatı ($) |
| Seasonality | string | Mevsim |

---

## Analiz Adımları

### 1. Veri Yükleme ve Hazırlık
- CSV dosyası yükleniyor
- `Date` sütunu datetime'a dönüştürülüyor
- Sadece **Groceries** kategorisi filtreleniyor (14.611 kayıt)

### 2. Keşifsel Veri Analizi (EDA)
- Günlük toplam satışlar hesaplanıyor
- Ham satış verisi ve 7 günlük hareketli ortalama görselleştiriliyor

### 3. Risk Tespiti
- **Stoksuz Kalma:** `Demand Forecast > Inventory Level`
- **Aşırı Stok/İsraf:** `Inventory Level > Units Sold * 2`

### 4. Genel Talep Tahmini (Tüm Ürünler)
- Facebook Prophet ile 30 gün ileriye tahmin
- Train: İlk 671 gün / Test: Son 30 gün
- **MAE:** 465.9 birim

### 5. Ürün Bazında Analiz (P0016)
- En çok satan ürün: **P0016** (110.560 birim)
- Günlük ortalama satış: 213.8 birim
- Tahmini günlük talep: 236 birim
- 30 günde 9 gün israf riski

### 6. Model Karşılaştırması

| Model | MAE (birim) | Durum |
|-------|-------------|-------|
| Baseline (sabit ortalama) | 109.2 | **En İyi** |
| Prophet (varsayılan) | 117.3 | İkinci |
| Prophet (geliştirilmiş) | 130.5 | En Kötü |

> Bu bulgu, model seçiminin **ürün bazında** yapılması gerektiğini gösteriyor.

### 7. Risk Dashboard
4 panelden oluşan görsel özet:
1. Tahmin vs Gerçek Satış
2. Model Karşılaştırması (MAE)
3. Risk Dağılımı (Pasta Grafik)
4. Proje Özet Metrikleri

### 8. Sezonluk Analiz

| Metrik | Değer |
|--------|-------|
| En yüksek satış ayı | Kasım (162 birim) |
| En düşük satış ayı | Aralık (129 birim) |
| En yüksek satış günü | Pazar (151 birim) |
| En düşük satış günü | Çarşamba (136 birim) |

---

## Temel Bulgular

1. Grocery kategorisinde kayıtların **%50.4'ünde** israf/aşırı stok riski tespit edildi
2. **%3.7'sinde** stoksuz kalma riski var
3. En çok satan ürün P0016'da **Kasım** en yüksek, **Aralık** en düşük satış ayı
4. **Pazar** günleri satış zirvesi, **Çarşamba** en düşük gün
5. Model karşılaştırmasında P0016 için baseline model daha iyi performans gösterdi

---

## Risk Skorlama Mantığı

```
Eğer Tahmini Talep > Stok ise
    → "Stoksuz Kalma Riski"

Eğer Stok > 2 x Günlük Satış ise
    → "İsraf / Aşırı Stok Riski"

Aksi halde
    → "Normal"
```

---

## Kullanılan Teknolojiler

| Kütüphane | Kullanım Amacı |
|-----------|----------------|
| Python 3.14 | Programlama dili |
| pandas | Veri manipülasyonu ve analizi |
| matplotlib | Grafik ve görselleştirme |
| Facebook Prophet | Zaman serisi tahmini |
| scikit-learn | MAE metrik hesaplama |
| Jupyter Notebook | Analiz ortamı |

---

## Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.14+
- pip3

### Kurulum

```bash
pip3 install pandas matplotlib prophet scikit-learn
```

### Çalıştırma

1. `Untitled0.ipynb` dosyasını VS Code veya Jupyter'da açın
2. "Run All" ile tüm hücreleri çalıştırın

---

## Dosya Yapısı

```
microsoft/
├── AGENTS.md                    # AI agent talimatları ve detaylı proje dokümantasyonu
├── README.md                    # Bu dosya
├── Untitled0.ipynb              # Ana analiz notebook'u
├── retail_store_inventory.csv   # Ham veri seti
├── risk_dashboard.png           # Risk dashboard görseli
└── sezonluk_analiz.png          # Sezonluk analiz grafiği
```

---

## Gelecek İçin Öneriler

1. **Çoklu ürün modeli:** Her ürün için ayrı Prophet modeli eğitilmeli
2. **Ek değişkenler:** Hava durumu, tatil, indirim gibi değişkenler modele dahil edilmeli
3. **Mağaza bazlı analiz:** Farklı mağazaların farklı talep örüntüleri olabilir
4. **Gerçek zamanlı güncelleme:** Model periyodik olarak yeniden eğitilmeli
5. **Maliyet analizi:** İsraf ve stoksuz kalmanın finansal etkisi hesaplanmalı
6. **XGBoost / LSTM:** Daha gelişmiş modellerle karşılaştırma yapılmalı

---

## Bilinen Sorunlar

- Emoji warning'ler: DejaVu Sans fontu emoji desteklemiyor
- Veri seti sentetik (Kaggle), gerçek üretim verisiyle doğrulama yapılmalı
- Risk eşik değerleri gerçek işletme verilerine göre ayarlanmalı

---

**Beray Tutal** — Microsoft AI Skills Challenge 2026
