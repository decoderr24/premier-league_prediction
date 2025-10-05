import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

# =========================================
# 1️⃣ LOAD DATA
# =========================================
print("📊 Memuat data historis Premier League...")
df = pd.read_csv("epl-training.csv")
df["Date"] = pd.to_datetime(df["Date"])
print(f"✅ Data: {len(df)} pertandingan dari {df['Date'].min().date()} sampai {df['Date'].max().date()}")

# =========================================
# 2️⃣ FEATURE ENGINEERING
# =========================================
print("⚙️ Menghitung form 5 pertandingan terakhir per tim...")

def get_team_form(df, team, n=5):
    team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    team_matches = team_matches.sort_values("Date")
    results = []

    for i, row in team_matches.iterrows():
        prev_matches = team_matches[team_matches["Date"] < row["Date"]].tail(n)
        if len(prev_matches) < n:
            results.append(None)
            continue

        # hitung statistik sederhana
        if row["HomeTeam"] == team:
            gf = prev_matches.apply(lambda r: r["FTHG"] if r["HomeTeam"] == team else r["FTAG"], axis=1).mean()
            ga = prev_matches.apply(lambda r: r["FTAG"] if r["HomeTeam"] == team else r["FTHG"], axis=1).mean()
            win_rate = prev_matches.apply(lambda r: 1 if (r["FTR"] == "H" and r["HomeTeam"] == team) or (r["FTR"] == "A" and r["AwayTeam"] == team) else 0, axis=1).mean()
        else:
            gf = prev_matches.apply(lambda r: r["FTAG"] if r["AwayTeam"] == team else r["FTHG"], axis=1).mean()
            ga = prev_matches.apply(lambda r: r["FTHG"] if r["AwayTeam"] == team else r["FTAG"], axis=1).mean()
            win_rate = prev_matches.apply(lambda r: 1 if (r["FTR"] == "A" and r["AwayTeam"] == team) or (r["FTR"] == "H" and r["HomeTeam"] == team) else 0, axis=1).mean()

        results.append((gf, ga, win_rate))
    return results

home_features = get_team_form(df, df["HomeTeam"].iloc[0])
away_features = get_team_form(df, df["AwayTeam"].iloc[0])

df["home_avg_gf"] = df.apply(lambda x: np.nan, axis=1)
df["home_avg_ga"] = df.apply(lambda x: np.nan, axis=1)
df["home_win_rate"] = df.apply(lambda x: np.nan, axis=1)
df["away_avg_gf"] = df.apply(lambda x: np.nan, axis=1)
df["away_avg_ga"] = df.apply(lambda x: np.nan, axis=1)
df["away_win_rate"] = df.apply(lambda x: np.nan, axis=1)

teams = df["HomeTeam"].unique()
for team in teams:
    home_stats = get_team_form(df, team)
    for i, val in enumerate(home_stats):
        if val:
            gf, ga, win_rate = val
            if df.iloc[i]["HomeTeam"] == team:
                df.at[i, "home_avg_gf"] = gf
                df.at[i, "home_avg_ga"] = ga
                df.at[i, "home_win_rate"] = win_rate
            elif df.iloc[i]["AwayTeam"] == team:
                df.at[i, "away_avg_gf"] = gf
                df.at[i, "away_avg_ga"] = ga
                df.at[i, "away_win_rate"] = win_rate

df = df.dropna()

# =========================================
# 3️⃣ LABEL ENCODING
# =========================================
def map_result(x):
    if x == "H":
        return "Home"
    elif x == "A":
        return "Away"
    else:
        return "Draw"

df["Result"] = df["FTR"].map(map_result)

features = [
    "home_avg_gf", "home_avg_ga", "home_win_rate",
    "away_avg_gf", "away_avg_ga", "away_win_rate"
]

X = df[features]
y = df["Result"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_map = dict(zip(le.transform(le.classes_), le.classes_))
print("🔁 Label encoding:", label_map)

# =========================================
# 4️⃣ SPLIT DATA
# =========================================
split_date = df["Date"].quantile(0.9)
X_train = X[df["Date"] < split_date]
X_test = X[df["Date"] >= split_date]
y_train = y_encoded[df["Date"] < split_date]
y_test = y_encoded[df["Date"] >= split_date]

print(f"📅 Train: {len(X_train)} | Test: {len(X_test)} pertandingan")

# =========================================
# 5️⃣ TRAINING MODEL
# =========================================
print("🧠 Melatih model XGBoost (regularized, no leak)...")
model = XGBClassifier(
    use_label_encoder=False,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
)

model.fit(X_train, y_train)

# =========================================
# 6️⃣ EVALUASI
# =========================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_prob)
print("\n📊 HASIL EVALUASI:")
print(f"   - Accuracy        : {acc*100:.2f}%")
print(f"   - Log Loss        : {ll:.4f}\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix (rows true, cols pred):")
print(confusion_matrix(y_test, y_pred))

# =========================================
# 7️⃣ FEATURE IMPORTANCE
# =========================================
plt.figure(figsize=(10,5))
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.title("Feature importance (gain) - top features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# =========================================
# 8️⃣ SIMPAN MODEL
# =========================================
joblib.dump(model, "model_epl_v1.joblib")
joblib.dump(le, "labelencoder_v1.joblib")

print("\n✅ Model & encoder berhasil disimpan!")
