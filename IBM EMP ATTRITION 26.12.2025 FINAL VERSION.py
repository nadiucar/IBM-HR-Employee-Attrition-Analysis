# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 17:12:38 2025

@author: nadiucar
"""

# İŞ VE VERİ ANLAYIŞI
#%% Pandas,Os,seaborn,matplotlib.pyplot,numpy kütüphanelerinin çalışma ortamına alınması işlemi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import XGBoost as xg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
#%%
#csv uzantılı veri setimi çalışma ortamıma çektim.Ardından benzersizliklerine ilk 20 sütuna ve  tümm sütunlarına baktım.
## Veri setini doğrudan aynı klasörden okuyoruz
he = pd.read_csv("HREMP.csv", sep=",", header=0)
he.columns
he.nunique()
he.head(20)
#%%
# HeatMap yardımı ile korelasyon bakma
# (Tüm çalışanlar için ortak olan sütunların çıkarılması bu analizin ön hazırlığıdır)
cols_to_drop = ["EmployeeCount","Over18","StandardHours","EmployeeNumber","DailyRate","MonthlyRate","HourlyRate","JobLevel"]
he_clean = he.drop(columns=cols_to_drop)
numeric_he = he_clean.select_dtypes(include=[np.number])
#%%
#Sayısal değişkenler arasındakikorelasyon katsayılarını hesaplayıp grafikleştirdim.
corr_cal = numeric_he.corr()
mask = np.triu(np.ones_like(corr_cal, dtype=bool))
plt.figure(figsize=(20, 12))
sns.set(font_scale=0.8)
sns.heatmap(corr_cal,mask=mask,annot=True,fmt=".2f",cmap='coolwarm',linewidths=0.5)
plt.title("IBM HR Attrition - Sayısal Değişkenler Korelasyon Matrisi",fontsize=22,fontweight='bold',pad=20)
plt.show()
#%%
# Outlier Değer Varmı? Onu inceledim.
outlier_cols = ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion']
custom_colors = ['#e2bc9b', '#e18e00', '#d64527', '#a52a2a'] 

plt.figure(figsize=(15, 14))
for i, (col, color) in enumerate(zip(outlier_cols, custom_colors), 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=he_clean[col], color=color, width=0.4)
    plt.title(f'{col} Dağılımı', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel("Değer", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
#%%
# VERİ HAZIRLAMA
#%% 0-1 olacak şekilde ikili(Binary) işlemi yaptım
he_clean['Attrition'] = he_clean['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
he_clean["OverTime"] = he_clean["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)

# Stringer olan sütunlar için One-Hot Encoding yaptım. 
# Ayrıca Dummy Variable tuzağına düşmemek için aynı mantığa gelen sütunları drop_first=True ile temizledim.
he_clean_encode = pd.get_dummies(he_clean, columns=["BusinessTravel","Department","EducationField","JobRole","MaritalStatus","Gender"], drop_first= True)
print(f"Yeni veri seti boyutu: {he_clean_encode.shape}")
#%% Aykırı değerlerin modeli şaşırtmaması için modeli sağlıklı hale getirdik
X = he_clean_encode.drop("Attrition", axis=1)
y = he_clean_encode["Attrition"]
#%% Veriyi 0.8 Train - 0.2 Test olacak şekilde bölmeyi tercih ettim(Standart 80/20 değer!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Ölçeklendirme yapacağım sayısal değişken listesini numeric_cols değişkenine atadım
numeric_cols = ['Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
                'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 
                'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
                'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
                'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Ölçeklendirme yapıyorum(RobustScaler)
preprocessor = ColumnTransformer(
    transformers=[('num', RobustScaler(), numeric_cols)],
    remainder='passthrough'
)
X_train_scaled = preprocessor.fit_transform(X_train)

#Veri sızıntısını (Data Leakage) önlemek için test verisine sadece eğitimden öğrendiğin ölçeklendirmeyi uygulamasını söylüyorum.
X_test_scaled = preprocessor.transform(X_test)

print("İşlem Tamam: Sayısal veriler 'Robust' şekilde ölçeklendi, Binary veriler korundu!")
#%%
# MODELLEME VE DEĞERLENDİRME

# 1. Modeli Tanımladım(Lojistik Regresyon)
log_model = LogisticRegression(max_iter=2000)

# 2. Modeli eğittim
log_model.fit(X_train_scaled, y_train)

# 3. tahmin yaptım
y_pred = log_model.predict(X_test_scaled)

# 4. Performansı Raporladım
print("--- Lojistik Regresyon Sonuçları ---")
print(classification_report(y_test, y_pred))

# Modeli 'dengeli' ağırlıklarla tekrar kurup Attrition Yes/No dengesini kurmak istedim.
log_model_balanced = LogisticRegression(max_iter=2000, class_weight='balanced')
log_model_balanced.fit(X_train_scaled, y_train)
y_pred_balanced = log_model_balanced.predict(X_test_scaled)
print("--- Dengeli Lojistik Regresyon Sonuçları ---")
print(classification_report(y_test, y_pred_balanced))

# Random Forest Modeli
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
print("--- Random Forest Sonuçları ---")
print(classification_report(y_test, y_pred_rf))

# XGBoost Modeli
ratio = 247 / 47
xgb_model = XGBClassifier(scale_pos_weight=ratio, n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
print("--- XGBoost Sonuçları ---")
print(classification_report(y_test, y_pred_xgb))
#%%
# UYGULAMA VE GÖRSELLEŞTİRME

# Lojistik Regresyon katsayılarını görselleştirdim
importance = log_model_balanced.coef_[0]
feature_names = X_train.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='magma')
plt.title('İşten Ayrılmayı Tetikleyen En Güçlü 10 Faktör')
plt.show()

# Confusion Matrix Görselleştirdim
cm = confusion_matrix(y_test, y_pred_balanced)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Kalıyor (Negatif)", "Ayrılıyor (Pozitif)"])
disp.plot(cmap='Reds', values_format='d')
plt.title('Modelin "Risk" Yakalama Başarısı (Lojistik Regresyon)', fontsize=14)
plt.grid(False)
plt.show()

# Sınıf Dağılımı Pie Chart bakarak dengesiz Yes/No Kanıtladım.
labels = ['Kalanlar (No)', 'Ayrılanlar (Yes)']
sizes = [247, 47] 
colors = ['#2e7d32', '#c62828']
explode = (0, 0.1) 
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
        autopct='%1.1f%%', shadow=True, startangle=140,
        textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Veri Setindeki Sınıf Dağılımı (Attrition Imbalance)', fontsize=15)
plt.show()