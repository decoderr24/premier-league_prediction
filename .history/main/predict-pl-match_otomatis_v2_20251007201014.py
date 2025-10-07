import pandas as pd
import numpy as np
import os
import warnings
from datetime import timedelta
import joblib

warnings.filterwarnings("ignore")

# Model & util
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# =============================================================================
# BAGIAN 1, 2, 3, 4, 5 (LOAD DATA & FEATURE ENGINEERING) - TIDAK BERUBAH
# =============================================================================
print("📊 Memuat data...")
df = pd.read_csv("csv/epl-training.csv", encoding='ISO-8859-1', low_memory=False)
expected = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']
df = df[expected].copy()
df.rename(columns={'HomeTeam': 'Home', 'AwayTeam': 'Away', 'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals', 'HS': 'HomeShots', 'AS': 'AwayShots', 'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)
print(f"✅ Data dimuat: {df.shape[0]} baris, dari {df['Date'].min().date()} sampai {df['Date'].max().date()}")

print("🔧 Membuat fitur dasar (Result)...")
def result_label(row):
    if row['HomeGoals'] > row['AwayGoals']: return 1
    if row['HomeGoals'] < row['AwayGoals']: return 2
    return 0
df['Result'] = df.apply(result_label, axis=1)

print("📈 Menghitung rolling stats per tim (window=5)...")
team_rows = []
for _, r in df.iterrows():
    team_rows.append({'Date': r['Date'], 'Team': r['Home'], 'GoalsFor': r['HomeGoals'], 'GoalsAgainst': r['AwayGoals'], 'ShotsFor': r['HomeShots'], 'SOT_For': r['HomeShotsOnTarget'], 'Win': 1 if r['HomeGoals']>r['AwayGoals'] else 0, 'Draw': 1 if r['HomeGoals']==r['AwayGoals'] else 0})
    team_rows.append({'Date': r['Date'], 'Team': r['Away'], 'GoalsFor': r['AwayGoals'], 'GoalsAgainst': r['HomeGoals'], 'ShotsFor': r['AwayShots'], 'SOT_For': r['AwayShotsOnTarget'], 'Win': 1 if r['AwayGoals']>r['HomeGoals'] else 0, 'Draw': 1 if r['AwayGoals']==r['HomeGoals'] else 0})
team_df = pd.DataFrame(team_rows).sort_values(['Team', 'Date']).reset_index(drop=True)

window = 5
agg = team_df.groupby('Team').rolling(window=window, on='Date', min_periods=1).mean().reset_index().rename(columns={'GoalsFor': f'AvgGoalsFor_L{window}', 'GoalsAgainst': f'AvgGoalsAgainst_L{window}', 'ShotsFor': f'AvgShotsFor_L{window}', 'SOT_For': f'AvgSOTFor_L{window}', 'Win': f'WinRate_L{window}', 'Draw': f'DrawRate_L{window}'})
rolling_stats = agg[['Team', 'Date', f'AvgGoalsFor_L{window}', f'AvgGoalsAgainst_L{window}', f'AvgShotsFor_L{window}', f'AvgSOTFor_L{window}', f'WinRate_L{window}', f'DrawRate_L{window}']]
df = pd.merge(df, rolling_stats, left_on=['Home','Date'], right_on=['Team','Date'], how='left').rename(columns={f'AvgGoalsFor_L{window}': f'AvgGoalsFor_Home_L{window}', f'AvgGoalsAgainst_L{window}': f'AvgGoalsAgainst_Home_L{window}', f'AvgShotsFor_L{window}': f'AvgShotsFor_Home_L{window}', f'AvgSOTFor_L{window}': f'AvgSOTFor_Home_L{window}', f'WinRate_L{window}': f'WinRate_Home_L{window}', f'DrawRate_L{window}': f'DrawRate_Home_L{window}'}).drop(columns=['Team'])
df = pd.merge(df, rolling_stats, left_on=['Away','Date'], right_on=['Team','Date'], how='left').rename(columns={f'AvgGoalsFor_L{window}': f'AvgGoalsFor_Away_L{window}', f'AvgGoalsAgainst_L{window}': f'AvgGoalsAgainst_Away_L{window}', f'AvgShotsFor_L{window}': f'AvgShotsFor_Away_L{window}', f'AvgSOTFor_L{window}': f'AvgSOTFor_Away_L{window}', f'WinRate_L{window}': f'WinRate_Away_L{window}', f'DrawRate_L{window}': f'DrawRate_Away_L{window}'}).drop(columns=['Team'])

print("🔁 Membuat fitur Last Result & Streak...")
def get_last_result(team, date, df_matches):
    prev = df_matches[((df_matches['Home']==team)|(df_matches['Away']==team)) & (df_matches['Date'] < date)]
    if prev.empty: return 0
    row = prev.iloc[-1]
    if (row['Home']==team and row['HomeGoals']>row['AwayGoals']) or (row['Away']==team and row['AwayGoals']>row['HomeGoals']): return 1
    if row['HomeGoals']==row['AwayGoals']: return 0
    return -1

def get_recent_streak(team, date, df_matches, lookback=5):
    prev = df_matches[((df_matches['Home']==team)|(df_matches['Away']==team)) & (df_matches['Date'] < date)].tail(lookback)
    if prev.empty: return 0
    streak = 0
    for i in range(len(prev)-1, -1, -1):
        r = prev.iloc[i]
        is_win = (r['Home']==team and r['HomeGoals']>r['AwayGoals']) or (r['Away']==team and r['AwayGoals']>r['HomeGoals'])
        if is_win: streak += 1
        else: break
    return streak

df['HomeLastResult'] = df.apply(lambda r: get_last_result(r['Home'], r['Date'], df), axis=1)
df['AwayLastResult'] = df.apply(lambda r: get_last_result(r['Away'], r['Date'], df), axis=1)
df['HomeWinStreak_L5'] = df.apply(lambda r: get_recent_streak(r['Home'], r['Date'], df, 5), axis=1)
df['AwayWinStreak_L5'] = df.apply(lambda r: get_recent_streak(r['Away'], r['Date'], df, 5), axis=1)

print("⚔️ Menghitung H2H sederhana...")
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
df['h2h_home_wins'] = h2h_home_wins; df['h2h_away_wins'] = h2h_away_wins; df['h2h_draws'] = h2h_draws

# =============================================================================
# 6. Final feature set & cleaning - DENGAN PERBAIKAN
# =============================================================================
print("🧹 Membersihkan & menyiapkan fitur akhir...")
features = [
    f'AvgGoalsFor_Home_L{window}', f'AvgGoalsAgainst_Home_L{window}', f'AvgShotsFor_Home_L{window}', f'AvgSOTFor_Home_L{window}',
    f'WinRate_Home_L{window}', f'DrawRate_Home_L{window}', 'HomeLastResult', 'HomeWinStreak_L5',
    f'AvgGoalsFor_Away_L{window}', f'AvgGoalsAgainst_Away_L{window}', f'AvgShotsFor_Away_L{window}', f'AvgSOTFor_Away_L{window}',
    f'WinRate_Away_L{window}', f'DrawRate_Away_L{window}', 'AwayLastResult', 'AwayWinStreak_L5',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
] # === 'GoalDiff' DIHAPUS DARI SINI ===

df_model = df.dropna(subset=features + ['Result']).copy()
X = df_model[features]
y = df_model['Result']
print(f"Dataset untuk modeling: {X.shape[0]} baris x {X.shape[1]} fitur")

# =============================================================================
# 7. Train / validation (TimeSeriesSplit) - Tidak Berubah
# =============================================================================
print("🧪 Membuat TimeSeriesSplit untuk cross-validation temporal...")
tscv = TimeSeriesSplit(n_splits=5)
if HAS_XGB:
    print("⚡ XGBoost tersedia, menggunakan XGBClassifier.")
    base_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
else:
    print("ℹ️ XGBoost tidak tersedia — fallback ke RandomForestClassifier.")
    base_model = RandomForestClassifier(random_state=42, class_weight='balanced')

pipeline = Pipeline([('scaler', StandardScaler()), ('clf', base_model)])
# Untuk mempercepat, kita lewati GridSearchCV dan langsung pakai parameter bagus
params = {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.8}
if HAS_XGB:
    final_model = Pipeline([('scaler', StandardScaler()), ('clf', XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss', random_state=42))])
else:
    final_model = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced'))])

# =============================================================================
# 8. Final evaluation on last fold hold-out - Tidak Berubah
# =============================================================================
split_idx = int(len(df_model) * 0.8)
X_train_final, y_train_final = X.iloc[:split_idx], y.iloc[:split_idx]
X_test_final, y_test_final = X.iloc[split_idx:], y.iloc[split_idx:]

print("🧩 Melatih model final pada 80% data awal dan evaluasi pada 20% terakhir...")
final_model.fit(X_train_final, y_train_final)
y_pred = final_model.predict(X_test_final)
y_proba = final_model.predict_proba(X_test_final)

acc = accuracy_score(y_test_final, y_pred)
bal_acc = balanced_accuracy_score(y_test_final, y_pred)
ll = log_loss(y_test_final, y_proba)
print("\n📊 Evaluasi Akhir (Hold-out temporal):")
print(f"  - Accuracy        : {acc*100:.2f}%")
print(f"  - Balanced Acc.   : {bal_acc*100:.2f}%")
print(f"  - Log Loss        : {ll:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test_final, y_pred, target_names=['Draw','Home','Away']))
cm = confusion_matrix(y_test_final, y_pred)
print("Confusion Matrix (rows true, cols pred):\n", cm)

# =============================================================================
# 9. Predict function interactive - DENGAN PERBAIKAN
# =============================================================================
print("\n🔮 Menyediakan fungsi predict_match (dinamis)...")
team_latest_stats = rolling_stats.sort_values('Date').groupby('Team').tail(1).set_index('Team')
full_match_history = df_model.copy()

def predict_match(home_team_input, away_team_input, model=final_model, team_roll_df=team_latest_stats, history_df=full_match_history):
    name_map = {"Man Utd": "Man United", "Spurs": "Tottenham", "Nottingham Forest": "Nott'm Forest"}
    home = name_map.get(home_team_input, home_team_input)
    away = name_map.get(away_team_input, away_team_input)
    print("="*40)
    print(f"PREDIKSI: {home_team_input} vs {away_team_input}")
    print("="*40)
    if home not in team_roll_df.index or away not in team_roll_df.index:
        print("⚠️ Data form/rolling tidak ditemukan untuk salah satu tim."); return None
    
    h = team_roll_df.loc[home]
    a = team_roll_df.loc[away]
    
    # Hitung fitur dinamis untuk pertandingan "hari ini"
    today = pd.to_datetime('today')
    h_last_res = get_last_result(home, today, history_df)
    a_last_res = get_last_result(away, today, history_df)
    h_streak = get_recent_streak(home, today, history_df, 5)
    a_streak = get_recent_streak(away, today, history_df, 5)
    h2h_hw, h2h_aw, h2h_d = calc_h2h_counts(home, away, today, history_df)

    feat_vals = [
        h[f'AvgGoalsFor_L{window}'], h[f'AvgGoalsAgainst_L{window}'], h[f'AvgShotsFor_L{window}'], h[f'AvgSOTFor_L{window}'],
        h[f'WinRate_L{window}'], h[f'DrawRate_L{window}'], h_last_res, h_streak,
        a[f'AvgGoalsFor_L{window}'], a[f'AvgGoalsAgainst_L{window}'], a[f'AvgShotsFor_L{window}'], a[f'AvgSOTFor_L{window}'],
        a[f'WinRate_L{window}'], a[f'DrawRate_L{window}'], a_last_res, a_streak,
        h2h_hw, h2h_aw, h2h_d
    ]
    
    X_input = pd.DataFrame([feat_vals], columns=features)
    probs = model.predict_proba(X_input)[0]
    
    print(f"Info Form (Home): WinRate={h[f'WinRate_L{window}']:.2f}, AvgGoals={h[f'AvgGoalsFor_L{window}']:.2f}, LastResult={h_last_res}, WinStreak={h_streak}")
    print(f"Info Form (Away): WinRate={a[f'WinRate_L{window}']:.2f}, AvgGoals={a[f'AvgGoalsFor_L{window}']:.2f}, LastResult={a_last_res}, WinStreak={a_streak}")
    print(f"Info H2H: {home} menang {h2h_hw}, {away} menang {h2h_aw}, seri {h2h_d}")
    
    print(f"\nProbabilities -> Draw: {probs[0]*100:.2f}%, Home: {probs[1]*100:.2f}%, Away: {probs[2]*100:.2f}%")
    pred_class = np.argmax(probs)
    label_map = {0: "Seri", 1: home_team_input + " Menang", 2: away_team_input + " Menang"}
    print("Kesimpulan:", label_map[pred_class])
    return probs

# contoh prediksi
try:
    predict_match("Aston Villa", "Burnley")
    predict_match("Wolves", "Brighton")
except Exception as e:
    print(f"Info: contoh prediksi gagal - {e}")

joblib.dump(final_model, "model_epl_final.joblib")
print("\n💾 Model tersimpan sebagai 'model_epl_final.joblib'")