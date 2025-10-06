# predict-pl-match_otomatis_v2.py
"""
Versi perbaikan pipeline prediksi hasil pertandingan:
- Fitur: form (rolling mean 5), win rate, avg goal diff, last result, home/away streak
- Validasi temporal: TimeSeriesSplit
- Model: XGBoost jika ada, fallback RandomForest
- Evaluasi: accuracy, balanced_accuracy, log_loss, classification_report, confusion matrix
- Fungsi predict_match yang menangani mapping nama dan ketiadaan data
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

# Model & util
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# coba import xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# -----------------------------
# 1. Load & prepare data
# -----------------------------
print("📊 Memuat data...")
if not os.path.exists("epl-training.csv"):
    raise FileNotFoundError("File 'epl-training.csv' tidak ditemukan di folder saat ini.")

df = pd.read_csv("epl-training.csv", encoding='ISO-8859-1', low_memory=False)
# Kolom dasar yang kita butuhkan (sesuaikan jika berbeda)
expected = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']
for c in expected:
    if c not in df.columns:
        raise KeyError(f"Kolom {c} tidak ditemukan. Pastikan dataset sesuai (kaggle epl).")

df = df[expected].copy()
df.rename(columns={
    'HomeTeam': 'Home', 'AwayTeam': 'Away',
    'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals',
    'HS': 'HomeShots', 'AS': 'AwayShots',
    'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget'
}, inplace=True)

# parse date (beberapa dataset punya format berbeda)
for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
    try:
        df['Date'] = pd.to_datetime(df['Date'], format=fmt)
        break
    except Exception:
        continue
if df['Date'].dtype == object:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)
print(f"✅ Data dimuat: {df.shape[0]} baris, dari {df['Date'].min().date()} sampai {df['Date'].max().date()}")

# -----------------------------
# 2. Basic engineered features
# -----------------------------
print("🔧 Membuat fitur dasar (GoalDiff, Result)...")
df['GoalDiff'] = df['HomeGoals'] - df['AwayGoals']
def result_label(row):
    if row['HomeGoals'] > row['AwayGoals']: return 1   # Home win
    if row['HomeGoals'] < row['AwayGoals']: return 2   # Away win
    return 0                                          # Draw
df['Result'] = df.apply(result_label, axis=1)

# -----------------------------
# 3. Per-team timeline & rolling features (window=5)
# -----------------------------
print("📈 Menghitung rolling stats per tim (window=5)...")
team_rows = []
for _, r in df.iterrows():
    team_rows.append({'Date': r['Date'], 'Team': r['Home'], 'GoalsFor': r['HomeGoals'], 'GoalsAgainst': r['AwayGoals'],
                      'ShotsFor': r['HomeShots'], 'SOT_For': r['HomeShotsOnTarget'], 'IsHome': 1,
                      'Win': 1 if r['HomeGoals']>r['AwayGoals'] else 0, 'Draw': 1 if r['HomeGoals']==r['AwayGoals'] else 0})
    team_rows.append({'Date': r['Date'], 'Team': r['Away'], 'GoalsFor': r['AwayGoals'], 'GoalsAgainst': r['HomeGoals'],
                      'ShotsFor': r['AwayShots'], 'SOT_For': r['AwayShotsOnTarget'], 'IsHome': 0,
                      'Win': 1 if r['AwayGoals']>r['HomeGoals'] else 0, 'Draw': 1 if r['AwayGoals']==r['HomeGoals'] else 0})

team_df = pd.DataFrame(team_rows).sort_values(['Team', 'Date']).reset_index(drop=True)

# rolling agregations
window = 5
agg = team_df.groupby('Team').rolling(window=window, on='Date', min_periods=1).agg({
    'GoalsFor': 'mean',
    'GoalsAgainst': 'mean',
    'ShotsFor': 'mean',
    'SOT_For': 'mean',
    'Win': 'mean',
    'Draw': 'mean'
}).reset_index().rename(columns={
    'GoalsFor': f'AvgGoalsFor_L{window}',
    'GoalsAgainst': f'AvgGoalsAgainst_L{window}',
    'ShotsFor': f'AvgShotsFor_L{window}',
    'SOT_For': f'AvgSOTFor_L{window}',
    'Win': f'WinRate_L{window}',
    'Draw': f'DrawRate_L{window}'
})

# merge rolling back to original matches (for home & away)
# buat helper: ambil rolling stats terakhir per tim per tanggal
rolling_stats = agg[['Team', 'Date', f'AvgGoalsFor_L{window}', f'AvgGoalsAgainst_L{window}',
                     f'AvgShotsFor_L{window}', f'AvgSOTFor_L{window}', f'WinRate_L{window}', f'DrawRate_L{window}']]

# merge untuk Home
df = pd.merge(df, rolling_stats, left_on=['Home','Date'], right_on=['Team','Date'], how='left').rename(columns={
    f'AvgGoalsFor_L{window}': f'AvgGoalsFor_Home_L{window}',
    f'AvgGoalsAgainst_L{window}': f'AvgGoalsAgainst_Home_L{window}',
    f'AvgShotsFor_L{window}': f'AvgShotsFor_Home_L{window}',
    f'AvgSOTFor_L{window}': f'AvgSOTFor_Home_L{window}',
    f'WinRate_L{window}': f'WinRate_Home_L{window}',
    f'DrawRate_L{window}': f'DrawRate_Home_L{window}'
}).drop(columns=['Team'])

# merge untuk Away
df = pd.merge(df, rolling_stats, left_on=['Away','Date'], right_on=['Team','Date'], how='left').rename(columns={
    f'AvgGoalsFor_L{window}': f'AvgGoalsFor_Away_L{window}',
    f'AvgGoalsAgainst_L{window}': f'AvgGoalsAgainst_Away_L{window}',
    f'AvgShotsFor_L{window}': f'AvgShotsFor_Away_L{window}',
    f'AvgSOTFor_L{window}': f'AvgSOTFor_Away_L{window}',
    f'WinRate_L{window}': f'WinRate_Away_L{window}',
    f'DrawRate_L{window}': f'DrawRate_Away_L{window}'
}).drop(columns=['Team'])

# -----------------------------
# 4. Fitur last match result & streak
# -----------------------------
print("🔁 Membuat fitur Last Result & Streak...")
# last result: 1=win, 0=draw, -1=loss dari perspektif tim
def get_last_result(team, date, df_matches):
    prev = df_matches[((df_matches['Home']==team)|(df_matches['Away']==team)) & (df_matches['Date'] < date)]
    if prev.empty: return 0
    row = prev.iloc[-1]
    if row['Home']==team:
        if row['HomeGoals']>row['AwayGoals']: return 1
        if row['HomeGoals']<row['AwayGoals']: return -1
        return 0
    else:
        if row['AwayGoals']>row['HomeGoals']: return 1
        if row['AwayGoals']<row['HomeGoals']: return -1
        return 0

def get_recent_streak(team, date, df_matches, lookback=5):
    prev = df_matches[((df_matches['Home']==team)|(df_matches['Away']==team)) & (df_matches['Date'] < date)].tail(lookback)
    if prev.empty: return 0
    # streak of wins as integer (jumlah kemenangan beruntun terakhir dari paling dekat)
    streak = 0
    for i in range(len(prev)-1, -1, -1):
        r = prev.iloc[i]
        is_win = (r['Home']==team and r['HomeGoals']>r['AwayGoals']) or (r['Away']==team and r['AwayGoals']>r['HomeGoals'])
        if is_win:
            streak += 1
        else:
            break
    return streak

# apply (may be slow on large df — masih ok untuk ~10k rows)
df['HomeLastResult'] = df.apply(lambda r: get_last_result(r['Home'], r['Date'], df), axis=1)
df['AwayLastResult'] = df.apply(lambda r: get_last_result(r['Away'], r['Date'], df), axis=1)
df['HomeWinStreak_L5'] = df.apply(lambda r: get_recent_streak(r['Home'], r['Date'], df, 5), axis=1)
df['AwayWinStreak_L5'] = df.apply(lambda r: get_recent_streak(r['Away'], r['Date'], df, 5), axis=1)

# -----------------------------
# 5. Head-to-head counts (simple)
# -----------------------------
print("⚔️ Menghitung H2H sederhana (jumlah pertemuan sebelumnya)...")
def calc_h2h_counts(home, away, date, df_matches):
    prev = df_matches[((df_matches['Home']==home)&(df_matches['Away']==away))|((df_matches['Home']==away)&(df_matches['Away']==home))]
    prev = prev[prev['Date'] < date]
    if prev.empty: return 0,0,0
    home_wins = ((prev['Home']==home) & (prev['HomeGoals']>prev['AwayGoals'])) | ((prev['Away']==home) & (prev['AwayGoals']>prev['HomeGoals']))
    away_wins = ((prev['Home']==away) & (prev['HomeGoals']>prev['AwayGoals'])) | ((prev['Away']==away) & (prev['AwayGoals']>prev['HomeGoals']))
    draws = prev['HomeGoals']==prev['AwayGoals']
    return int(home_wins.sum()), int(away_wins.sum()), int(draws.sum())

h2h_home_wins, h2h_away_wins, h2h_draws = [], [], []
for _, r in df.iterrows():
    a,b,c = calc_h2h_counts(r['Home'], r['Away'], r['Date'], df)
    h2h_home_wins.append(a); h2h_away_wins.append(b); h2h_draws.append(c)
df['h2h_home_wins'] = h2h_home_wins
df['h2h_away_wins'] = h2h_away_wins
df['h2h_draws'] = h2h_draws

# -----------------------------
# 6. Final feature set & cleaning
# -----------------------------
print("🧹 Membersihkan & menyiapkan fitur akhir...")
features = [
    # Home form
    f'AvgGoalsFor_Home_L{window}', f'AvgGoalsAgainst_Home_L{window}', f'AvgShotsFor_Home_L{window}', f'AvgSOTFor_Home_L{window}',
    f'WinRate_Home_L{window}', f'DrawRate_Home_L{window}', 'HomeLastResult', 'HomeWinStreak_L5',
    # Away form
    f'AvgGoalsFor_Away_L{window}', f'AvgGoalsAgainst_Away_L{window}', f'AvgShotsFor_Away_L{window}', f'AvgSOTFor_Away_L{window}',
    f'WinRate_Away_L{window}', f'DrawRate_Away_L{window}', 'AwayLastResult', 'AwayWinStreak_L5',
    # H2H
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
    # dasar
    'GoalDiff'
]

# Hapus baris yg memiliki NA di fitur penting
df_model = df.dropna(subset=features + ['Result']).copy()
X = df_model[features]
y = df_model['Result']

print(f"Dataset untuk modeling: {X.shape[0]} baris x {X.shape[1]} fitur")

# -----------------------------
# 7. Train / validation (TimeSeriesSplit)
# -----------------------------
print("🧪 Membuat TimeSeriesSplit untuk cross-validation temporal...")
tscv = TimeSeriesSplit(n_splits=5)

# Pilih model: XGBoost jika tersedia, else RandomForest
if HAS_XGB:
    print("⚡ XGBoost tersedia, menggunakan XGBClassifier.")
    base_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, verbosity=0)
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
else:
    print("ℹ️ XGBoost tidak tersedia — fallback ke RandomForestClassifier.")
    base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [6, 10]
    }

# pipeline: scaling (XGBoost biasanya tidak butuh, tapi tetap aman)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', base_model)
])

# GridSearchCV dengan TimeSeriesSplit
print("🔎 Melakukan GridSearchCV (CV temporal, cepat)...")
gsearch = GridSearchCV(
    pipeline,
    param_grid={'clf__' + k: v for k, v in param_grid.items()},
    cv=tscv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)
gsearch.fit(X, y)
best = gsearch.best_estimator_
print(f"✅ Best params: {gsearch.best_params_}")
print(f"✅ Best CV score (balanced_accuracy): {gsearch.best_score_:.4f}")

# -----------------------------
# 8. Final evaluation on last fold hold-out
# -----------------------------
# split temporally: gunakan 80% untuk train, 20% terakhir sebagai hold-out test
split_idx = int(len(df_model) * 0.8)
X_train_final = X.iloc[:split_idx]
y_train_final = y.iloc[:split_idx]
X_test_final = X.iloc[split_idx:]
y_test_final = y.iloc[split_idx:]

print("🧩 Melatih model final pada 80% data awal dan evaluasi pada 20% terakhir...")
best.fit(X_train_final, y_train_final)
y_pred = best.predict(X_test_final)
y_proba = best.predict_proba(X_test_final)

acc = accuracy_score(y_test_final, y_pred)
bal_acc = balanced_accuracy_score(y_test_final, y_pred)
ll = log_loss(y_test_final, y_proba)

print("\n📊 Evaluasi Akhir (Hold-out temporal):")
print(f"  - Accuracy        : {acc*100:.2f}%")
print(f"  - Balanced Acc.   : {bal_acc*100:.2f}%")
print(f"  - Log Loss        : {ll:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test_final, y_pred, target_names=['Draw','Home','Away']))

# Confusion matrix (printed)
cm = confusion_matrix(y_test_final, y_pred)
print("Confusion Matrix (rows true, cols pred):")
print(cm)

# -----------------------------
# 9. Predict function interactive
# -----------------------------
print("\n🔮 Menyediakan fungsi predict_match (pakai rolling_stats & df_model)...")
# reuse rolling_stats DataFrame defined earlier (agg). But rolling_stats punya kolom 'Team', 'Date', ...
# Untuk fungsi prediksi kita cari latest rolling stats untuk tiap tim
team_latest = rolling_stats.sort_values('Date').groupby('Team').tail(1).set_index('Team')

def predict_match(home_team_input, away_team_input, model=best, team_roll_df=team_latest):
    name_map = {"Man Utd": "Man United", "Spurs": "Tottenham", "Chealsea": "Chelsea"}
    home = name_map.get(home_team_input, home_team_input)
    away = name_map.get(away_team_input, away_team_input)
    print("="*40)
    print(f"PREDIKSI: {home_team_input} vs {away_team_input}")
    print("="*40)
    # ambil rolling terakhir
    if home not in team_roll_df.index or away not in team_roll_df.index:
        print("⚠️ Data form/rolling tidak ditemukan untuk salah satu tim. Pastikan nama tim konsisten.")
        return None
    h = team_roll_df.loc[home]
    a = team_roll_df.loc[away]
    # ambil fitur sesuai urutan features
    feat_vals = [
        h[f'AvgGoalsFor_L{window}'], h[f'AvgGoalsAgainst_L{window}'], h[f'AvgShotsFor_L{window}'], h[f'AvgSOTFor_L{window}'],
        h[f'WinRate_L{window}'], h[f'DrawRate_L{window}'], 0, 0,   # placeholders untuk HomeLastResult, HomeWinStreak (tidak tersedia utk "hari ini")
        a[f'AvgGoalsFor_L{window}'], a[f'AvgGoalsAgainst_L{window}'], a[f'AvgShotsFor_L{window}'], a[f'AvgSOTFor_L{window}'],
        a[f'WinRate_L{window}'], a[f'DrawRate_L{window}'], 0, 0,
        0,0,0,   # h2h placeholders (kamu bisa hitung H2H aktual utk pertandingan tgl tertentu)
        0        # GoalDiff placeholder
    ]
    X_input = np.array(feat_vals).reshape(1, -1)
    # jika pipeline ada scaler, tetap aman
    probs = model.predict_proba(X_input)[0]
    print(f"Probabilities -> Draw: {probs[0]*100:.2f}%, Home: {probs[1]*100:.2f}%, Away: {probs[2]*100:.2f}%")
    pred_class = np.argmax(probs)
    label_map = {0: "Seri", 1: home_team_input + " Menang", 2: away_team_input + " Menang"}
    print("Kesimpulan:", label_map[pred_class])
    return probs

# contoh prediksi (gunakan tim yang ada di dataset)
try:
    predict_match("Brentford", "Man Utd")
except Exception as e:
    print("Info: contoh prediksi gagal karena data input — ini normal jika tim tidak cocok nama/formatnya.", e)

# -----------------------------
# 10. Simpan model (opsional)
# -----------------------------
import joblib
joblib.dump(best, "model_epl_best.joblib")
print("\n💾 Model tersimpan sebagai 'model_epl_best.joblib'")

print("\nSelesai. Jika ingin, kamu bisa:")
print("- Menambahkan fitur klasemen (posisi) per match date (butuh data klasemen per tanggal).")
print("- Menambahkan pemain absen / lineup / injury (butuh sumber data tambahan).")
print("- Menambahkan/menyesuaikan mapping nama tim agar konsisten.")
