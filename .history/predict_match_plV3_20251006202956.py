import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

print("📊 Memuat data...")

# =============================
# 1️⃣ BACA DATA
# =============================
df = pd.read_csv("historical_matches.csv")  # Ganti nama file sesuai data kamu

print(f"✅ Data dimuat: {len(df)} baris, dari {df['Date'].min()} sampai {df['Date'].max()}")

# =============================
# 2️⃣ NORMALISASI NAMA KOLOM
# =============================
df.columns = df.columns.str.strip().str.lower()
possible_home_goals = ['fthg', 'home_goals', 'homegoal', 'home_score', 'hg']
possible_away_goals = ['ftag', 'away_goals', 'awaygoal', 'away_score', 'ag']

home_col = next((c for c in df.columns if c in possible_home_goals), None)
away_col = next((c for c in df.columns if c in possible_away_goals), None)

if not home_col or not away_col:
    raise KeyError("❌ Kolom skor Home/Away tidak ditemukan. Cek nama kolom di CSV kamu!")

# =============================
# 3️⃣ BUAT LABEL HASIL PERTANDINGAN
# =============================
def get_result(row):
    if row[home_col] > row[away_col]:
        return 2  # Home menang
    elif row[home_col] < row[away_col]:
        return 0  # Away menang
    else:
        return 1  # Seri

df["Result"] = df.apply(get_result, axis=1)
print("🔧 Membuat label hasil (Result)...")

# =============================
# 4️⃣ PERSIAPAN DATA DASAR
# =============================
df = df.dropna(subset=["home", "away"])
teams = pd.concat([df["home"], df["away"]]).unique()

# Pastikan urutan kronologis
df["Date"] = pd.to_datetime(df["date"])
df = df.sort_values("Date")

# =============================
# 5️⃣ HITUNG FITUR ROLLING (LEAKAGE-SAFE)
# =============================
WINDOW = 5
rolling_feats = []

team_records = []
for team in teams:
    team_df = df[(df["home"] == team) | (df["away"] == team)].copy()
    team_df["is_home"] = np.where(team_df["home"] == team, 1, 0)
    team_df["GoalsFor"] = np.where(team_df["is_home"], team_df[home_col], team_df[away_col])
    team_df["GoalsAgainst"] = np.where(team_df["is_home"], team_df[away_col], team_df[home_col])
    team_df["Win"] = (team_df["GoalsFor"] > team_df["GoalsAgainst"]).astype(int)
    team_df["Draw"] = (team_df["GoalsFor"] == team_df["GoalsAgainst"]).astype(int)
    team_df["Loss"] = (team_df["GoalsFor"] < team_df["GoalsAgainst"]).astype(int)
    team_df["Points"] = team_df["Win"] * 3 + team_df["Draw"]
    team_df = team_df.sort_values("Date")

    for feat in ["GoalsFor", "GoalsAgainst", "Points"]:
        team_df[f"Roll_{feat}_L{WINDOW}"] = (
            team_df[feat].shift().rolling(WINDOW, min_periods=1).mean()
        )
        rolling_feats.append(f"Roll_{feat}_L{WINDOW}")

    team_df["Team"] = team
    team_records.append(team_df[["Date", "Team"] + [f"Roll_{f}_L{WINDOW}" for f in ["GoalsFor", "GoalsAgainst", "Points"]]])

team_df = pd.concat(team_records)

# =============================
# 6️⃣ MERGE FITUR HOME DAN AWAY
# =============================
df = df.merge(team_df.add_prefix("Home_"), left_on=["Date", "home"], right_on=["Home_Date", "Home_Team"], how="left")
df = df.merge(team_df.add_prefix("Away_"), left_on=["Date", "away"], right_on=["Away_Date", "Away_Team"], how="left")

# =============================
# 7️⃣ BERSIHKAN & PILIH FITUR
# =============================
features = [
    "Home_Roll_GoalsFor_L5", "Home_Roll_GoalsAgainst_L5", "Home_Roll_Points_L5",
    "Away_Roll_GoalsFor_L5", "Away_Roll_GoalsAgainst_L5", "Away_Roll_Points_L5"
]

for col in features:
    if col not in df.columns:
        print(f"⚠️ Kolom {col} tidak ditemukan, membuat kolom kosong (NaN)...")
        df[col] = np.nan

df_model = df.dropna(subset=features + ["Result"]).copy()

print(f"🧹 Dataset siap: {df_model.shape[0]} baris x {df_model.shape[1]} kolom")

# =============================
# 8️⃣ TRAIN TEST SPLIT (80/20)
# =============================
X = df_model[features]
y = df_model["Result"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

# =============================
# 9️⃣ MODEL ENSEMBLE
# =============================
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=3)
lr = LogisticRegression(max_iter=500, multi_class="multinomial")

models = [("RF", rf), ("GB", gb), ("LR", lr)]

preds = []
for name, model in models:
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)
    preds.append(p)
    print(f"✅ {name} selesai training")

# Ensemble rata-rata probabilitas
p_final = np.mean(preds, axis=0)
y_pred = np.argmax(p_final, axis=1)

# =============================
# 🔟 EVALUASI
# =============================
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
ll = log_loss(y_test, p_final)
print("\n📊 Evaluasi Akhir (Hold-out temporal):")
print(f"  - Accuracy        : {acc*100:.2f}%")
print(f"  - Balanced Acc.   : {bal_acc*100:.2f}%")
print(f"  - Log Loss        : {ll:.4f}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Away", "Draw", "Home"]))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =============================
# 💾 SIMPAN MODEL
# =============================
joblib.dump((models, scaler, features), "model_epl_final_v3.joblib")
print("\n💾 Model tersimpan sebagai 'model_epl_final_v3.joblib'")

print("\n✅ Selesai. Model siap digunakan untuk prediksi pertandingan berikutnya.")
