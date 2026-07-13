# AGENTS.md — Akıllı Gıda Tedarik Zinciri Projesi

## Proje Hakkında

Bu proje, perakende gıda sektöründe **talep tahmini** yaparak **stoksuz kalma** ve **israf/aşırı stok** risklerini önceden tespit etmeyi amaçlar. Facebook Prophet zaman serisi modellemesi kullanılarak gelecek satışlar tahmin edilir ve bu tahminlere göre risk skorları üretilir.

**Proje Sahibi:** Beray Tutal
**Bağlam:** Microsoft AI Skills Challenge 2026 — Staj Projesi
**Veri Kaynağı:** [Retail Store Inventory Forecasting Dataset (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
**Dil:** Python 3.14
**Platform:** Jupyter Notebook / VS Code

---

## Dosya Yapısı

```
microsoft/
├── AGENTS.md                    # Bu dosya — AI agent talimatları
├── README.md                    # Proje tanımı ve çalıştırma kılavuzu
├── Untitled0.ipynb              # Ana analiz notebook'u (tüm adımlar burada)
├── retail_store_inventory.csv   # Ham veri seti (73.100 kayıt, 15 sütun)
├── risk_dashboard.png           # Risk dashboard görseli (4 panel)
└── sezonluk_analiz.png          # Sezonluk satış analiz grafiği
```

---

## Veri Seti Detayları

**Dosya:** `retail_store_inventory.csv`
**Boyut:** 73.100 satır x 15 sütun
**Zaman Aralığı:** 2022-01-01 — 2023-12-31 (2 yıl)
**Mağaza Sayısı:** Birden fazla (S001, S002, ...)
**Ürün Sayısı:** P0001 — P0020 (20 farklı ürün)
**Kategoriler:** Groceries, Toys, Electronics, Furniture, Clothing

### Sütun Tanımları

| Sütun | Tip | Açıklama |
|-------|-----|----------|
| Date | tarih | Kayıt tarihi (YYYY-MM-DD) |
| Store ID | string | Mağaza kimliği (S001, S002, ...) |
| Product ID | string | Ürün kimliği (P0001 — P0020) |
| Category | string | Ürün kategorisi (Groceries, Toys, vb.) |
| Region | string | Coğrafi bölge (North, South, East, West) |
| Inventory Level | int | O günkü stok miktarı (birim) |
| Units Sold | int | O günkü satış miktarı (birim) |
| Units Ordered | int | O günkü sipariş miktarı (birim) |
| Demand Forecast | float | Sistemin tahmini talep değeri (birim) |
| Price | float | Birim fiyatı ($) |
| Discount | int | İndirim yüzdesi (0, 5, 10, 20) |
| Weather Condition | string | Hava durumu (Sunny, Rainy, Cloudy, Snowy) |
| Holiday/Promotion | int | Tatil/promosyon durumu (0 veya 1) |
| Competitor Pricing | float | Rakip fiyatı ($) |
| Seasonality | string | Mevsim (Spring, Summer, Autumn, Winter) |

---

## Notebook Akışı (Adım Adım)

### 1. Veri Yükleme ve Hazırlık (Hücre 1-3)

**Amaç:** CSV dosyasını yüklemek, temizlemek ve Groceries kategorisini filtrelemek.

```python
import pandas as pd
df = pd.read_csv("retail_store_inventory.csv")

# Eksik veri kontrolü — tüm sütunlarda 0 eksik
print(df.isnull().sum())  # Sonuç: 0 (tüm sütunlarda)

# Sadece gıda ürünleri
groceries = df[df["Category"] == "Groceries"]
# Sonuç: 14.611 kayıt (toplamın ~%20'si)
```

**Kritik Notlar:**
- Veri setinde hiç eksik değer yok (null = 0 tüm sütunlarda)
- `Date` sütunu string olarak yükleniyor, `pd.to_datetime()` dönüşümü gerekli
- Groceries kategorisi en fazla kayda sahip kategori

---

### 2. Keşifsel Veri Analizi — EDA (Hücre 4-5)

**Amaç:** Günlük gıda satışlarının zaman içindeki örüntüsünü görselleştirmek.

```python
daily_sales = groceries.groupby("Date")["Units Sold"].sum()

# Ham günlük satış grafiği
plt.plot(daily_sales.index, daily_sales.values)

# 7 günlük hareketli ortalama ile yumuşatılmış grafik
daily_sales_smooth = daily_sales.rolling(7).mean()
```

**Bulgular:**
- Günlük satışlarda belirgin dalgalanmalar var
- 7 günlük hareketli ortalama ile trend daha net görülüyor
- Belirli dönemlerde zirve ve dip noktaları mevcut

---

### 3. Stoksuz Kalma ve İsraf Riski Analizi (Hücre 6)

**Amaç:** Verideki riskli durumları tespit etmek.

```python
# Stoksuz kalma: Talep > Mevcut Stok
stockout = groceries[groceries["Demand Forecast"] > groceries["Inventory Level"]]

# Aşırı stok / israf: Stok > 2x Satış
overstock = groceries[groceries["Inventory Level"] > groceries["Units Sold"] * 2]
```

**Sonuçlar:**

| Risk Türü | Kayıt Sayısı | Oran |
|-----------|-------------|------|
| Toplam Grocery Kaydı | 14.611 | — |
| Stoksuz Kalma Riski | 535 | %3.7 |
| İsraf / Aşırı Stok Riski | 7.364 | %50.4 |

**Yorum:** Grocery kategorisinde kayıtların yarısından fazlasında aşırı stok/israf riski bulunuyor. Bu, gıda sektöründe ciddi bir kaynak israfına işaret ediyor.

---

### 4. Genel Talep Tahmini — Tüm Ürünler (Hücre 7-10)

**Amaç:** Tüm gıda ürünlerinin günlük toplam satışı için Facebook Prophet ile zaman serisi tahmini yapmak.

```python
from prophet import Prophet

# Prophet için gerekli format: ds (tarih) ve y (değer)
df_prophet = daily_sales.reset_index()
df_prophet.columns = ["ds", "y"]

# Model eğitimi
model = Prophet()
model.fit(df_prophet)

# 30 gün ileriye tahmin
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Görselleştirme
fig = model.plot(forecast)
```

**Model Performansı (Tüm Ürünler):**

| Metric | Değer |
|--------|-------|
| MAE (Ortalama Mutlak Hata) | 465.9 birim |
| Train set | İlk 671 gün |
| Test set | Son 30 gün |

**Risk Analizi (Son 30 Gün — Tüm Ürünler):**
- Ortalama günlük stok: 5.512 birim
- Tahmini ortalama günlük talep: ~2.600 birim
- Sonuç: 30 günün tamamında **israf/aşırı stok riski** tespit edildi

---

### 5. Ürün Bazında Derin Analiz — P0016 (Hücre 11-14)

**Amaç:** En çok satan ürüne (P0016) özel detaylı analiz yapmak.

```python
# En çok satan ürünleri listele
urun_listesi = groceries.groupby("Product ID")["Units Sold"].sum().sort_values(ascending=False)
# P0016: 110.560 birim ile 1. sırada

# P0016 için günlük satışı hesapla
urun = groceries[groceries["Product ID"] == "P0016"].copy()
urun_gunluk = urun.groupby("Date")["Units Sold"].sum().reset_index()
urun_gunluk.columns = ["ds", "y"]
# Toplam gün sayısı: 517, Günlük ortalama satış: 213.8 birim
```

**P0016 Tahmin Modeli:**

```python
urun_model = Prophet()
urun_model.fit(urun_gunluk)

urun_future = urun_model.make_future_dataframe(periods=30)
urun_forecast = urun_model.predict(urun_future)
```

**P0016 Risk Analizi (Son 30 Gün):**

| Metrik | Değer |
|--------|-------|
| Tahmini Ortalama Günlük Talep | 236 birim |
| Ortalama Günlük Stok | 425 birim |
| Normal Gün | 45 |
| İsraf/Aşırı Stok Riski | 9 |

---

### 6. Model Karşılaştırması (Hücre 15-16)

**Amaç:** Farklı modelleri karşılaştırarak en iyi tahmin yöntemini belirlemek.

**Test Metodolojisi:**
- Train: İlk 487 gün
- Test: Son 30 gün
- Metrik: MAE (Ortalama Mutlak Hata)

```python
from sklearn.metrics import mean_absolute_error

# Baseline: Son 30 günün ortalaması (sabit değer tahmini)
baseline_mae = 109.2 birim

# Prophet (varsayılan parametreler)
prophet_mae = 117.3 birim

# Prophet (geliştirilmiş: yearly + weekly seasonality, multiplicative)
prophet_mae3 = 130.5 birim
```

**Sonuç Tablosu (P0016):**

| Model | MAE (birim) | Durum |
|-------|-------------|-------|
| Baseline (sabit ortalama) | 109.2 | **En İyi** |
| Prophet (varsayılan) | 117.3 | İkinci |
| Prophet (geliştirilmiş) | 130.5 | En Kötü |

**Kritik Bulgular:**
- P0016 ürünü için **baseline model** en iyi performansı gösterdi
- Prophet'in gelişmiş ayarları bu üründe daha kötü sonuç verdi
- **Bu bulgu önemli:** Model seçimi ürün bazında yapılmalı, tek bir model tüm ürünlerde en iyi sonucu vermez
- Yüksek satış varyasyonuna sahip ürünlerde basit modeller daha başarılı olabilir

---

### 7. Risk Dashboard (Hücre 17)

**Amaç:** Tüm analiz sonuçlarını tek bir görsel panelde toplamak.

Dashboard 4 panelden oluşur:

1. **Tahmin vs Gerçek Satış (Sol Üst):** Son 30 gün için gerçek satış, Prophet tahmini ve baseline karşılaştırması
2. **Model Karşılaştırması (Sağ Üst):** 3 modelin MAE değerlerinin çubuk grafiği
3. **Risk Dağılımı (Sol Alt):** P0016 için 30 günlük risk dağılımı (pasta grafik)
4. **Proje Özet Metrikleri (Sağ Alt):** Tüm önemli istatistiklerin tablosu

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
plt.savefig("risk_dashboard.png", dpi=150, bbox_inches="tight")
```

---

### 8. Sezonluk Analiz (Hücre 18)

**Amaç:** Hangi ay ve haftanın gününde satışların zirve yaptığını belirlemek.

```python
urun["Ay"] = urun["Date"].dt.month
urun["HaftaninGunu"] = urun["Date"].dt.day_name()

aylik_satis = urun.groupby("Ay")["Units Sold"].mean()
gun_satis = urun.groupby("HaftaninGunu")["Units Sold"].mean()
```

**Bulgular (P0016 Ürünü):**

| Metrik | Değer |
|--------|-------|
| En yüksek satış ayı | Kasım (162 birim) |
| En düşük satış ayı | Aralık (129 birim) |
| En yüksek satış günü | Pazar (151 birim) |
| En düşük satış günü | Çarşamba (136 birim) |

**Yorum:**
- Kasım ayında yıl sonu yoğunluğu ve tatil öncesi alışveriş etkisi var
- Aralık'ta düşüş, muhtemelen stok yetersizliği veya erken dönem alımlarının oluşu
- Pazar günleri hafta sonu alışveriş yoğunluğu nedeniyle zirve yapıyor

---

## Kullanılan Kütüphaneler

| Kütüphane | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| pandas | — | Veri manipülasyonu ve analizi |
| matplotlib | 3.11.0 | Grafik ve görselleştirme |
| prophet | — | Zaman serisi tahmini (Facebook) |
| scikit-learn | — | MAE metrik hesaplama |
| cmdstanpy | — | Prophet backend (Stan) |
| numpy | — | Sayısal işlemler |

**Kurulum:**
```bash
pip3 install pandas matplotlib prophet scikit-learn
```

---

## Risk Skorlama Mantığı

Proje içindeki risk değerlendirmesi şu kurallara dayanır:

```
Eğer Tahmini Talep > Stok ise
    → "Stoksuz Kalma Riski" (müşteri kaybı, satış kaybı)

Eğer Stok > 2 x Günlük Satış ise
    → "İsraf / Aşırı Stok Riski" (bozulma, gereksiz maliyet)

Aksi halde
    → "Normal"
```

---

## Gelecek İçin Öneriler

1. **Çoklu ürün modeli:** Her ürün için ayrı Prophet modeli eğitilmeli
2. **Ek değişkenler:** Hava durumu, tatil, indirim gibi değişkenler modele dahil edilmeli (Prophet `add_regressor` ile)
3. **Mağaza bazlı analiz:** Farklı mağazaların farklı talep örüntüleri olabilir
4. **Gerçek zamanlı güncelleme:** Model periyodik olarak yeniden eğitilmeli
5. **Maliyet analizi:** İsraf ve stoksuz kalmanın finansal etkisi hesaplanmalı
6. **XGBoost / LSTM:** Daha gelişmiş modellerle karşılaştırma yapılmalı
7. **Automated pipeline:** Günlük otomatik veri çekme ve tahmin üretme sistemi

---

## Bilinen Sorunlar ve Kısıtlamalar

- `Untitled0.ipynb` dosya adı anlamsız, `analiz.ipynb` olarak yeniden adlandırılmalı
- Prophet'in `plotly` bağımlılığı var, interaktif grafikler devre dışı
- Bazı hücrelerde `pip install` çağrısı var, notebook ortamında çalıştırılmalı
- Emoji warning'ler: DejaVu Sans fontu emoji desteklemiyor, dashboard görsellerinde bazı simgeler eksik
- Veri seti sentetik (Kaggle), gerçek üretim verisiyle doğrulama yapılmalı
- Risk eşik değerleri (%50 israf eşiği = 2x satış) rastgele seçilmiş, gerçek işletme verilerine göre ayarlanmalı

---

## AI Agent Talimatları

Bu projede çalışan bir AI agent şunları yapabilir:

1. **Yeni ürün analizi:** `Product ID` parametresini değiştirerek herhangi bir ürün için aynı analizleri tekrarlayabilirsin
2. **Farklı zaman aralıkları:** `periods=30` değerini değiştirerek farklı vadelerde tahmin üretebilirsin
3. **Hiperparametre optimizasyonu:** Prophet'in `changepoint_prior_scale`, `seasonality_mode` gibi parametrelerini ayarlayarak modeli iyileştirebilirsin
4. **Yeni risk eşikleri:** `groceries["Inventory Level"] > groceries["Units Sold"] * 2` ifadesindeki `2` çarpanını değiştirerek farklı risk seviyeleri tanımlayabilirsin
5. **Ek görselleştirme:** Mevcut matplotlib şablonlarını genişleterek yeni grafikler ekleyebilirsin

### Çalışma Kuralları
- `pip3` kullan (macOS — `pip` değil)
- Prophet modeli her çalıştırmada yeniden eğitiliyor, uzun sürebilir (~2-3 saniye)
- Grafikleri kaydederken `bbox_inches="tight"` kullan, aksi halde kesilir
- Türkçe karakter sorunu olabilir, matplotlib'te font ayarı gerekebilir
