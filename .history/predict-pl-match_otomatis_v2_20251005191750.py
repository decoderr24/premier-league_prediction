# predict-pl-match_otomatis_v2_fixed.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ======================
# 1️⃣ LOAD DATA
# ======================
print("📊 Memuat data historis Premier League...")
df = pd.read_csv("epl-training.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"✅ Data: {df.shape[0]} pertandingan dari {df['Date'].min().date()} sampai {df['Date'].max().date()}")

# ======================
# 2️⃣ PERSIAPAN FITUR DASAR
# ======================
df = df.rename(columns={
    'HomeTeam': 'Home', 'AwayTeam': 'Away',
    'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'
})
# result as string for clarity (will encode later)
df['Result'] = np.select(
    [df['HomeGoals'] > df['AwayGoals'], df['HomeGoals'] < df['AwayGoals']],
    ['Home', 'Away'], default='Draw'
)

# ======================
# 3️⃣ HITUNG FORM (LAST 5 MATCHES)
# ======================
print("⚙️ Menghitung form 5 pertandingan terakhir per tim...")

def compute_team_form(team, current_date, df_local):
    past = df_local[((df_local['Home'] == team) | (df_local['Away'] == team)) & (df_local['Date'] < current_date)]
    last5 = past.tail(5)
    if last5.empty:
        return pd.Series([np.nan]*4, index=['avg_gf','avg_ga','win_rate','goal_diff_avg'])
    gf, ga = [], []
    for _, row in last5.iterrows():
        if row['Home'] == team:
            gf.append(row['HomeGoals'])
            ga.append(row['AwayGoals'])
        else:
            gf.append(row['AwayGoals'])
            ga.append(row['HomeGoals'])
    wins = sum([1 if g1 > g2 else 0 for g1,g2 in zip(gf,ga)])
    return pd.Series([
        np.mean(gf), np.mean(ga), wins/len(last5), np.mean(np.array(gf)-np.array(ga))
    ], index=['avg_gf','avg_ga','win_rate','goal_diff_avg'])

# Buat kolom untuk fitur form home & away
home_features, away_features = [], []
# NOTE: pass a local copy of df without the new columns to avoid self-contamination
# but since we only added Result and Date/Home/Away/Goals exist, it's safe to use df directly.
for i, row in df.iterrows():
    home_features.append(compute_team_form(row['Home'], row['Date'], df))
    away_features.append(compute_team_form(row['Away'], row['Date'], df))

home_form = pd.DataFrame(home_features, index=df.index)
away_form = pd.DataFrame(away_features, index=df.index)

home_form = home_form.add_prefix("home_")
away_form = away_form.add_prefix("away_")

df = pd.concat([df, home_form, away_form], axis=1)
df = df.dropna().reset_index(drop=True)   # buang baris yang tidak punya cukup history
print("✅ Fitur form berhasil dibuat.")

# ======================
# 4️⃣ FITUR HEAD-TO-HEAD (H2H)
# ======================
print("⚔️ Menghitung H2H historis per pasangan tim...")

def get_h2h_stats(home, away, date, df_local):
    past = df_local[((df_local['Home'] == home) & (df_local['Away'] == away)) |
                    ((df_local['Home'] == away) & (df_local['Away'] == home))]
    past = past[past['Date'] < date]
    if past.empty:
        return pd.Series([0,0,0], index=['h2h_home_wins','h2h_away_wins','h2h_draws'])
    hw, aw, dr = 0,0,0
    for _, r in past.iterrows():
        if r['HomeGoals'] > r['AwayGoals']:
            if r['Home'] == home: hw += 1
            else: aw += 1
        elif r['AwayGoals'] > r['HomeGoals']:
            if r['Away'] == away: aw += 1
            else: hw += 1
        else:
            dr += 1
    return pd.Series([hw, aw, dr], index=['h2h_home_wins','h2h_away_wins','h2h_draws'])

h2h_stats = []
for i, row in df.iterrows():
    h2h_stats.append(get_h2h_stats(row['Home'], row['Away'], row['Date'], df))

df = pd.concat([df, pd.DataFrame(h2h_stats)], axis=1)
print("✅ Fitur H2H ditambahkan.")

# ======================
# 5️⃣ SIAPKAN DATA UNTUK MODEL
# ======================
features = [
    'home_avg_gf','home_avg_ga','home_win_rate','home_goal_diff_avg',
    'away_avg_gf','away_avg_ga','away_win_rate','away_goal_diff_avg',
    'h2h_home_wins','h2h_away_wins','h2h_draws'
]
target = 'Result'

split_date = '2023-01-01'
train = df[df['Date'] < split_date].copy()
test = df[df['Date'] >= split_date].copy()

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]
print(f"📅 Train: {train.shape[0]} | Test: {test.shape[0]} pertandingan")

# ======================
# Encode target labels -> numeric (important!)
# ======================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)   # assume test labels are subset of training labels (should be)

print("🔁 Label encoding:", dict(enumerate(le.classes_)))

# ======================
# 6️⃣ TRAIN MODEL XGBOOST (dengan fallback kompatibilitas)
# ======================
print("🧠 Melatih model XGBoost (regularized, no leak)...")

model = XGBClassifier(
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    reg_alpha=0.5,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)

# Fit with early stopping if supported; fallback to basic fit if not
try:
    # pass encoded y for eval_set
    model.fit(
        X_train, y_train_enc,
        eval_set=[(X_test, y_test_enc)],
        early_stopping_rounds=30,
        verbose=False
    )
except TypeError:
    print("⚠️ early_stopping_rounds tidak didukung oleh environment ini. Melatih tanpa early stopping...")
    model.fit(X_train, y_train_enc)
except Exception as e:
    # general fallback to avoid crash in weird envs
    print("⚠️ Terjadi error saat fit dengan early stopping:", e)
    print("Melatih tanpa early stopping...")
    model.fit(X_train, y_train_enc)

print("✅ Model selesai dilatih.")

# ======================
# 7️⃣ EVALUASI MODEL
# ======================
y_pred_enc = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# decode preds back to labels for readable report
y_pred = le.inverse_transform(y_pred_enc)

acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test_enc, y_proba)

print("\n📊 HASIL EVALUASI:")
print(f"   - Accuracy        : {acc*100:.2f}%")
print(f"   - Log Loss        : {ll:.4f}")
print("\n", classification_report(y_test, y_pred))
print("Confusion Matrix (rows true, cols pred):\n", confusion_matrix(y_test, y_pred))

# ======================
# 8️⃣ FUNGSI PREDIKSI MANUAL (menggunakan le untuk map class order)
# ======================
def predict_match(home_team, away_team):
    print("\n" + "="*40)
    print(f"🔮 PREDIKSI: {home_team} vs {away_team}")
    print("="*40)
    date = df['Date'].max() + pd.Timedelta(days=1)

    hf = compute_team_form(home_team, date, df)
    af = compute_team_form(away_team, date, df)
    h2h = get_h2h_stats(home_team, away_team, date, df)
    if hf.isna().any() or af.isna().any():
        print("⚠️ Tidak cukup data form untuk salah satu tim.")
        return
    X_pred = pd.DataFrame([[
        hf['avg_gf'], hf['avg_ga'], hf['win_rate'], hf['goal_diff_avg'],
        af['avg_gf'], af['avg_ga'], af['win_rate'], af['goal_diff_avg'],
        h2h['h2h_home_wins'], h2h['h2h_away_wins'], h2h['h2h_draws']
    ]], columns=features)
    proba = model.predict_proba(X_pred)[0]
    pred_enc = np.argmax(proba)
    pred_label = le.inverse_transform([pred_enc])[0]

    # print probabilities aligned with label encoder order
    print("\n📈 Probabilitas (label order dari encoder):")
    for idx, cls in enumerate(le.classes_):
        print(f"   - {cls:5s} : {proba[idx]*100:6.2f}%")
    print(f"\n💡 Kesimpulan: {pred_label}")

# ======================
# 9️⃣ CONTOH PREDIKSI
# ======================
predict_match("Arsenal", "Man Utd")
predict_match("Liverpool", "Man City")
predict_match("Chelsea", "Tottenham")
