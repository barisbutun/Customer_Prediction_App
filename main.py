import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Streamlit Başlığı
st.title("Müşteri Çıkışı (Churn) Tahmini Uygulaması")
def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve", color='blue')
    plt.plot([0, 1], [0, 1], '--', color='gray', label="Random Guessing")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)

# Veri Yükleme
uploaded_file = st.file_uploader("Veri setinizi yükleyin (CSV formatında):", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Yüklenen Veri:")
    st.dataframe(df.head())

    # Veri İşleme
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
    df.dropna(subset=['Purchase Date'], inplace=True)

    average_price = df['Product Price'].mean()
    average_age=df['Customer Age'].mean()

    df['Recency'] = (pd.Timestamp.now() - df['Purchase Date']).dt.days
    df['Frequency'] = df.groupby('Customer ID')['Purchase Date'].transform('count')
    df['Monetary'] = df.groupby('Customer ID')['Total Purchase Amount'].transform('sum')
    df['Cash'] = (df['Payment Method'] == 'Cash').astype(int)
    df['Crypto'] = (df['Payment Method'] == 'Crypto').astype(int)
    df['PayPal'] = (df['Payment Method'] == 'PayPal').astype(int)
    df['Credit_Card'] = (df['Payment Method'] == 'Credit Card').astype(int)
    df['Cheap_Product_Taking_Customer'] = df['Product Price'] < average_price
    df['Expensive_Product_Taking_Customer'] = df['Product Price'] > average_price
    df['Age_average_or_older'] = (df['Customer Age'] >= average_age).astype(int)
    df['age_average_under'] = (df['Customer Age'] < average_age).astype(int)
    df['Male'] = (df['Gender'] == 'Male').astype(int)
    df['Female'] = (df['Gender'] == 'Female').astype(int)

    features = ['Recency', 'Frequency', 'Monetary', 'Age_average_or_older', 'age_average_under', 'Male', 'Female', 'Cash', 'Crypto', 'PayPal', 'Credit_Card', 'Expensive_Product_Taking_Customer', 'Cheap_Product_Taking_Customer', 'Churn']
    df = df[features].drop_duplicates().dropna()

    st.subheader("İşlenen Veri Seti:")
    st.dataframe(df.head())


    st.subheader("Ürünlerin Ortalaması:")
    st.write(f"{average_price:.2f}")

    # Özellik ve Hedef Ayrımı
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # SMOTE ile Dengeleme
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Verileri Ölçekleme
    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(X_resampled)



    def calculate_and_plot_confusion_matrix(kf, X_resampled, y_resampled, hidden_neurons, learning_rate, epochs, title):
        y_true_all = []
        y_pred_all = []        

        for train_index, test_index in kf.split(X_resampled, y_resampled):
            X_train, X_test = X_resampled[train_index], X_resampled[test_index]
            y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        _, _, y_pred, _ = train_ann(X_train, y_train, X_test, y_test, hidden_neurons, learning_rate, epochs)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # Konfüzyon Matrisi Hesaplama
        conf_matrix = confusion_matrix(y_true_all, y_pred_all)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        st.pyplot(fig)

    # Model Tanımlama
    def train_ann(X_train, y_train, X_test, y_test, hidden_neurons=10, learning_rate=0.01, epochs=1000):
        model = MLPClassifier(hidden_layer_sizes=(hidden_neurons), learning_rate_init=learning_rate, max_iter=epochs, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        return accuracy, roc_auc, y_pred, model

    
        


    # Çapraz Doğrulama (5-Fold ve 10-Fold)
    
    kf_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies_5 = []
    roc_aucs_5 = []

    st.subheader("5-Fold Çapraz Doğrulama Sonuçları:")
    for train_index, test_index in kf_5.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        acc, roc_auc, _, _ = train_ann(X_train, y_train, X_test, y_test, hidden_neurons=10, learning_rate=0.01, epochs=500)
        accuracies_5.append(acc)
        roc_aucs_5.append(roc_auc)
    
    st.write(f"5-Fold Ortalama Doğruluk: {np.mean(accuracies_5):.4f}")
    st.write(f"5-Fold Ortalama ROC AUC: {np.mean(roc_aucs_5):.4f}")    

    st.subheader("5-Fold Çapraz Doğrulama Konfüzyon Matrisi:")
    kf_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    calculate_and_plot_confusion_matrix(kf_5, X_resampled, y_resampled, hidden_neurons=10, learning_rate=0.01, epochs=500, title="5-Fold Konfüzyon Matrisi")


    

    st.subheader("10-Fold Çapraz Doğrulama Sonuçları:")
    kf_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracies_10 = []
    roc_aucs_10 = []

    for train_index, test_index in kf_10.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        acc, roc_auc, _, _ = train_ann(X_train, y_train, X_test, y_test, hidden_neurons=10, learning_rate=0.01, epochs=500)
        accuracies_10.append(acc)
        roc_aucs_10.append(roc_auc)

    st.write(f"10-Fold Ortalama Doğruluk: {np.mean(accuracies_10):.4f}")
    st.write(f"10-Fold Ortalama ROC AUC: {np.mean(roc_aucs_10):.4f}")

    st.subheader("10-Fold Çapraz Doğrulama Konfüzyon Matrisi:")
    kf_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    calculate_and_plot_confusion_matrix(kf_10, X_resampled, y_resampled, hidden_neurons=10, learning_rate=0.01, epochs=500, title="10-Fold Konfüzyon Matrisi")
    

    # %66-%34 Eğitim/Test Ayırma
    st.subheader("%66-%34 Eğitim/Test Ayırma Sonuçları:")
    accuracies_split = []
    roc_aucs_split = []

    for i in range(5):  # 5 farklı rassal ayırma
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.34, random_state=i)

        acc, roc_auc, _, _ = train_ann(X_train, y_train, X_test, y_test, hidden_neurons=10, learning_rate=0.01, epochs=500)
        accuracies_split.append(acc)
        roc_aucs_split.append(roc_auc)

        if i == 4:
            acc, roc_auc, _, model = train_ann(X_train, y_train, X_test, y_test, hidden_neurons=10, learning_rate=0.01, epochs=500)
            y_prob = model.predict_proba(X_test)[:, 1]
            plot_roc_curve(y_test, y_prob)

    st.write(f"%66-%34 Ortalama Doğruluk: {np.mean(accuracies_split):.4f}")
    st.write(f"%66-%34 Ortalama ROC AUC: {np.mean(roc_aucs_split):.4f}")

    # Parametre Optimizasyonu
    st.subheader("Parametre Optimizasyonu:")
    param_grid = {
        'hidden_layer_sizes': [(10,), (10,10)],
        'learning_rate_init': [0.01,0.02],
        'max_iter': [100]
    }
    grid_search = GridSearchCV(MLPClassifier(random_state=42), param_grid, scoring='accuracy', cv=5,n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    st.write("En İyi Parametreler:")
    st.write(grid_search.best_params_)

    # En İyi Model ile Performans
    best_model = grid_search.best_estimator_
    y_pred_final = best_model.predict(X_resampled)
    st.subheader("Son Modelin Konfüzyon Matrisi:")
    conf_matrix = confusion_matrix(y_resampled, y_pred_final)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title("Confusion Matrix")
    st.pyplot(fig)
