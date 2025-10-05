# --- EPL Match Prediction Model ---
# Decoder Project | GPT-5
# Dataset: epl-training.csv

# ===============================================================
# 1. Import Library
# ===============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ===============================================================
# 2. Load Dataset
# ===============================================================
print("📂 Loading dataset...")
df = pd.read_csv("epl-training.csv")

print("\n--- Sample Data ---")
print(df.head())

# ===============================================================
# 3. Cek Informasi Dataset
# ===============================================================
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ===============================================================
# 4. Preprocessing
# ===============================================================
# Cek kolom target
target_col = 'result'  # ganti sesuai nama kolom hasil (misal: 'result' / 'outcome' / 'winner')
if target_col not in df.columns:
    raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataset.")

# Pisahkan fitur dan target
X = df.drop(columns=[target_col])
y = df[target_col]

# Label Encoding untuk kolom non-numerik
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"\n🔠 Encoding kolom kategorikal: {list(categorical_cols)}")
    encoder = LabelEncoder()
    for col in categorical_cols:
        X[col] = encoder.fit_transform(X[col])

# Normalisasi (optional)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================================================
# 5. Split Data (Train & Test)
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Jumlah Data Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ===============================================================
# 6. Train Model
# ===============================================================
print("\n🚀 Training RandomForest Classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight='balanced_subsample'
)
model.fit(X_train, y_train)

# ===============================================================
# 7. Evaluasi Model
# ===============================================================
y_pred = model.predict(X_test)

print("\n✅ Model Evaluation")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================================================
# 8. Confusion Matrix Visualization
# ===============================================================
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - EPL Match Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================================================
# 9. Simpan Model
# ===============================================================
joblib.dump(model, "epl_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n💾 Model, encoder, dan scaler berhasil disimpan!")
print("📁 File: epl_model.pkl, encoder.pkl, scaler.pkl")

# ===============================================================
# 10. Contoh Prediksi
# ===============================================================
print("\n🔮 Contoh Prediksi:")
sample = X.iloc[0:1]  # ambil satu baris contoh
sample_scaled = scaler.transform(sample)
pred_result = model.predict(sample_scaled)[0]
print(f"Hasil prediksi: {pred_result}")
