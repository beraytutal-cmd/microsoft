#  Akıllı Gıda Tedarik Zinciri — Talep Tahmini & Risk Analizi

##  Proje Özeti
Bu proje, perakende gıda sektöründe talep tahmini yaparak stoksuz kalma ve israf risklerini önceden tespit etmeyi amaçlamaktadır. Yanlış stok yönetimi nedeniyle oluşan kayıpları veri odaklı bir yaklaşımla minimize etmek için geliştirilmiştir.

##  Problem
Perakende gıda zincirlerinde iki kritik risk:
- **Stoksuz kalma:** Talep stoktan fazla → müşteri kaybı, satış kaybı
- **Aşırı stok:** Gereğinden fazla sipariş → bozulma, israf, gereksiz maliyet

##  Veri Seti
- **Kaynak:** Retail Store Inventory Forecasting Dataset (Kaggle)
- **Boyut:** 73.100 kayıt, 15 sütun
- **Kapsam:** 2022-2024 arası günlük satış, stok, talep ve kampanya verileri
- **Kategoriler:** Groceries, Electronics, Toys, Furniture, Clothing

##  Temel Bulgular
- Grocery kategorisinde kayıtların **%50.4'ünde** israf/aşırı stok riski tespit edildi
- **%3.7'sinde** stoksuz kalma riski var
- En çok satan ürün **P0016** için Kasım en yüksek, Aralık en düşük satış ayı
- **Pazar** günleri satış zirvesi, **Çarşamba** en düşük gün

##  Kullanılan Yöntemler
| Yöntem | Açıklama |
|--------|----------|
| Keşifsel Veri Analizi (EDA) | Satış örüntüleri, mevsimsellik analizi |
| Facebook Prophet | Zaman serisi tabanlı talep tahmini |
| Baseline Model | Karşılaştırma için sabit ortalama tahmini |
| Risk Skorlama | Stok/talep oranına göre otomatik uyarı sistemi |

##  Model Performansı (P0016 Ürünü)
| Model | MAE (birim) |
|-------|-------------|
| Baseline | 109.2 |
| Prophet (varsayılan) | 117.3 |
| Prophet (geliştirilmiş) | 130.5 |

> Bu ürün için yüksek satış varyasyonu nedeniyle baseline model daha iyi performans gösterdi. Bu bulgu, model seçiminin ürün bazında yapılması gerektiğini ortaya koymaktadır.

##  Kullanılan Teknolojiler
- Python 3.14
- pandas, matplotlib
- Facebook Prophet
- scikit-learn
- Jupyter Notebook / VS Code

##  Dosya Yapısı
microsoft/
├── Untitled0.ipynb          # Ana analiz notebook'u
├── retail_store_inventory.csv  # Veri seti
├── risk_dashboard.png       # Risk dashboard görseli
├── sezonluk_analiz.png      # Sezonluk analiz görseli
└── README.md                # Bu dosya
##  Nasıl Çalıştırılır?
1. Gerekli kütüphaneleri kur:pip3 install pandas matplotlib prophet scikit-learn
2. `Untitled0.ipynb` dosyasını VS Code veya Jupyter'da aç
3. "Run All" ile tüm hücreleri çalıştır

##  Geliştirici
**Beray Tutal**
Microsoft AI Skills Challenge 2026 Staj Projesi
