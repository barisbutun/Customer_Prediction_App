# Customer Churn Prediction App / Müşteri Çıkış Tahmini Uygulaması

## Overview / Genel Bakış
A Streamlit application for predicting customer churn using Artificial Neural Networks (ANN) with advanced RFM analysis.  
Gelişmiş RFM analizi ile Yapay Sinir Ağları (YSA) kullanarak müşteri çıkışını tahmin eden bir Streamlit uygulaması.

## Key Features / Temel Özellikler
- **RFM Analysis** / **RFM Analizi**:
  - Recency, Frequency, Monetary metrics / Yenilik, Sıklık, Parasal Değer metrikleri
  - Automated feature calculation / Otomatik özellik hesaplama

- **Data Processing** / **Veri İşleme**:
  - Automatic date parsing / Otomatik tarih ayrıştırma
  - Payment method encoding / Ödeme yöntemi kodlaması
  - Age-based segmentation / Yaşa dayalı segmentasyon

- **Machine Learning** / **Makine Öğrenmesi**:
  - ANN with customizable layers / Katmanları özelleştirilebilir YSA
  - SMOTE for balanced training / Dengeli eğitim için SMOTE
  - Hyperparameter tuning / Hiperparametre ayarlama

## How to Use / Nasıl Kullanılır

### 1. Data Upload / Veri Yükleme
- Upload your CSV file / CSV dosyanızı yükleyin
- Required columns / Gerekli sütunlar:
  - `Purchase Date`, `Product Price`, `Customer Age`
  - `Payment Method`, `Gender`, `Churn` (target)

### 2. Model Training / Model Eğitimi
- Select validation method / Doğrulama yöntemi seçin:
  - 5-Fold Cross-Validation / 5-Kat Çapraz Doğrulama
  - 10-Fold Cross-Validation / 10-Kat Çapraz Doğrulama
  - 66-34 Train-Test Split / 66-34 Eğitim-Test Ayrımı

### 3. View Results / Sonuçları Görüntüle
- Accuracy and ROC AUC scores / Doğruluk ve ROC AUC skorları
- Confusion matrices / Karışıklık matrisleri
- Feature importance / Özellik önem dereceleri

## Technical Details / Teknik Detaylar

### Requirements / Gereksinimler
```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
