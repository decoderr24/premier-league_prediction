# predict-pl-match_otomatis_v3.py
# Versi v3 — peningkatan feature engineering + Optuna tuning + ensemble
# - Fitur baru: EWM (weighted rolling), FormDiff, HomeAdvantage, RestDays
# - Hyperparameter tuning (Optuna) jika tersedia
# - Ensemble (XGB + RandomForest) dengan fallback
# - Fungsi predict_match diperbarui untuk menangani feature baru

import pandas as pd
import numpy as np
import os
import warnings
from datetime import timedelta
import joblib

warnings.filterwarnings('ignore')

# Model & util
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

# -------------------------
# 1) Load & basic preprocessing (tidak berubah banyak)
# -------------------------
print('📊 Memuat data...')
raw_path = 'epl-training.csv'
if not os.path.exists(raw_path):
    raise FileNotFoundError(f"File {raw_path} tidak ditemukan. Pastikan berada di folder yang sama.")

df = pd.read_csv(raw_path, encoding='ISO-8859-1', low_memory=False)
expected = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']
for c in expected:
    if c not in df.columns:
        raise ValueError(f"Kolom {c} tidak ditemukan di CSV.")

df = df[expected].copy()
df.rename(columns={'HomeTeam': 'Home', 'AwayTeam': 'Away', 'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals',
                   'HS': 'HomeShots', 'AS': 'AwayShots', 'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)
print(f"✅ Data dimuat: {df.shape[0]} baris, dari {df['Date'].min().date()} sampai {df['Date'].max().date()}")

# Basic result label
def result_label(row):
    if row['HomeGoals'] > row['AwayGoals']: return 1
    if row['HomeGoals'] < row['AwayGoals']: return 2
    return 0

df['Result'] = df.apply(result_label, axis=1)

# -------------------------
# 2) Build team-level dataframe (for rolling/EWM features)
# -------------------------
print('🔁 Menyusun team-level records untuk fitur rolling/EWM...')
team_rows = []
for _, r in df.iterrows():
    team_rows.append({'Date': r['Date'], 'Team': r['Home'], 'GoalsFor': r['HomeGoals'], 'GoalsAgainst': r['AwayGoals'], 'ShotsFor': r['HomeShots'], 'SOT_For': r['HomeShotsOnTarget'], 'Win': 1 if r['HomeGoals']>r['AwayGoals'] else 0, 'Draw': 1 if r['HomeGoals']==r['AwayGoals'] else 0, 'IsHome': 1})
    team_rows.append({'Date': r['Date'], 'Team': r['Away'], 'GoalsFor': r['AwayGoals'], 'GoalsAgainst': r['HomeGoals'], 'ShotsFor': r['AwayShots'], 'SOT_For': r['AwayShotsOnTarget'], 'Win': 1 if r['AwayGoals']>r['HomeGoals'] else 0, 'Draw': 1 if r['AwayGoals']==r['HomeGoals'] else 0, 'IsHome': 0})
team_df = pd.DataFrame(team_rows).sort_values(['Team','Date']).reset_index(drop=True)

# Parameters
WINDOW = 5
EWM_SPAN = 5

# Rolling means (simple) - existing style
agg = team_df.groupby('Team').rolling(window=WINDOW, on='Date', min_periods=1).mean().reset_index()
agg = agg.rename(columns={'GoalsFor': f'AvgGoalsFor_L{WINDOW}', 'GoalsAgainst': f'AvgGoalsAgainst_L{WINDOW}', 'ShotsFor': f'AvgShotsFor_L{WINDOW}', 'SOT_For': f'AvgSOTFor_L{WINDOW}', 'Win': f'WinRate_L{WINDOW}', 'Draw': f'DrawRate_L{WINDOW}'})
rolling_stats = agg[['Team','Date', f'AvgGoalsFor_L{WINDOW}', f'AvgGoalsAgainst_L{WINDOW}', f'AvgShotsFor_L{WINDOW}', f'AvgSOTFor_L{WINDOW}', f'WinRate_L{WINDOW}', f'DrawRate_L{WINDOW}']]

# Exponential weighted features (lebih sensitif ke pertandingan paling akhir)
print('📈 Menghitung EWM feature (lebih menekankan pertandingan terakhir)...')
team_df = team_df.sort_values(['Team','Date']).reset_index(drop=True)
team_df[f'EWM_GoalsFor_span{EWM_SPAN}'] = team_df.groupby('Team')['GoalsFor'].transform(lambda x: x.ewm(span=EWM_SPAN, adjust=False).mean())
team_df[f'EWM_WinRate_span{EWM_SPAN}'] = team_df.groupby('Team')['Win'].transform(lambda x: x.ewm(span=EWM_SPAN, adjust=False).mean())

ewm_stats = team_df.groupby(['Team','Date'])[[f'EWM_GoalsFor_span{EWM_SPAN}', f'EWM_WinRate_span{EWM_SPAN}']].last().reset_index()

# Merge rolling & ewm back to match-level dataframe
print('🔗 Menggabungkan statistik kembali ke level pertandingan...')
df = pd.merge(df, rolling_stats, left_on=['Home','Date'], right_on=['Team','Date'], how='left').rename(columns={f'AvgGoalsFor_L{WINDOW}': f'AvgGoalsFor_Home_L{WINDOW}', f'AvgGoalsAgainst_L{WINDOW}': f'AvgGoalsAgainst_Home_L{WINDOW}', f'AvgShotsFor_L{WINDOW}': f'AvgShotsFor_Home_L{WINDOW}', f'AvgSOTFor_L{WINDOW}': f'AvgSOTFor_Home_L{WINDOW}', f'WinRate_L{WINDOW}': f'WinRate_Home_L{WINDOW}', f'DrawRate_L{WINDOW}': f'DrawRate_Home_L{WINDOW}'}).drop(columns=['Team'])
df = pd.merge(df, rolling_stats, left_on=['Away','Date'], right_on=['Team','Date'], how='left').rename(columns={f'AvgGoalsFor_L{WINDOW}': f'AvgGoalsFor_Away_L{WINDOW}', f'AvgGoalsAgainst_L{WINDOW}': f'AvgGoalsAgainst_Away_L{WINDOW}', f'AvgShotsFor_L{WINDOW}': f'AvgShotsFor_Away_L{WINDOW}', f'AvgSOTFor_L{WINDOW}': f'AvgSOTFor_Away_L{WINDOW}', f'WinRate_L{WINDOW}': f'WinRate_Away_L{WINDOW}', f'DrawRate_L{WINDOW}': f'DrawRate_Away_L{WINDOW}'}).drop(columns=['Team'])

# Merge EWM
df = pd.merge(df, ewm_stats, left_on=['Home','Date'], right_on=['Team','Date'], how='left').rename(columns={f'EWM_GoalsFor_span{EWM_SPAN}': f'EWM_GoalsFor_Home_span{EWM_SPAN}', f'EWM_WinRate_span{EWM_SPAN}': f'EWM_WinRate_Home_span{EWM_SPAN}'}).drop(columns=['Team'])
df = pd.merge(df, ewm_stats, left_on=['Away','Date'], right_on=['Team','Date'], how='left').rename(columns={f'EWM_GoalsFor_span{EWM_SPAN}': f'EWM_GoalsFor_Away_span{EWM_SPAN}', f'EWM_WinRate_span{EWM_SPAN}': f'EWM_WinRate_Away_span{EWM_SPAN}'}).drop(columns=['Team'])

# -------------------------
# 3) Last result, streak, H2H, RestDays, HomeAdvantage, FormDiff
# -------------------------
print('🔧 Membuat fitur LastResult, Streak, H2H, RestDays, HomeAdvantage, FormDiff...')

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


def calc_h2h_counts(home, away, date, df_matches):
    prev = df_matches[((df_matches['Home']==home)&(df_matches['Away']==away))|((df_matches['Home']==away)&(df_matches['Away']==home))]
    prev = prev[prev['Date'] < date]
    if prev.empty: return 0,0,0
    home_wins = ((prev['Home']==home) & (prev['HomeGoals']>prev['AwayGoals'])) | ((prev['Away']==home) & (prev['AwayGoals']>prev['HomeGoals']))
    away_wins = ((prev['Home']==away) & (prev['HomeGoals']>prev['AwayGoals'])) | ((prev['Away']==away) & (prev['AwayGoals']>prev['HomeGoals']))
    draws = prev['HomeGoals']==prev['AwayGoals']
    return int(home_wins.sum()), int(away_wins.sum()), int(draws.sum())


def days_since_last(team, date, df_matches):
    prev = df_matches[((df_matches['Home']==team)|(df_matches['Away']==team)) & (df_matches['Date'] < date)]
    if prev.empty: return np.nan
    return (date - prev.iloc[-1]['Date']).days


def home_advantage_score(team, date, df_matches, lookback=5):
    prev_home = df_matches[(df_matches['Home']==team) & (df_matches['Date']<date)].tail(lookback)
    if prev_home.empty: return 0.0
    return (prev_home['HomeGoals'] > prev_home['AwayGoals']).sum() / len(prev_home)

# Create columns with iteration (vectorized where possible)
home_last_res, away_last_res, home_streak, away_streak = [], [], [], []
h2h_home_wins, h2h_away_wins, h2h_draws = [], [], []
home_rest, away_rest, home_adv, away_adv = [], [], [], []

for _, r in df.iterrows():
    date = r['Date']
    h = r['Home']; a = r['Away']
    home_last_res.append(get_last_result(h, date, df))
    away_last_res.append(get_last_result(a, date, df))
    home_streak.append(get_recent_streak(h, date, df, WINDOW))
    away_streak.append(get_recent_streak(a, date, df, WINDOW))
    hw, aw, dr = calc_h2h_counts(h, a, date, df)
    h2h_home_wins.append(hw); h2h_away_wins.append(aw); h2h_draws.append(dr)
    home_rest.append(days_since_last(h, date, df))
    away_rest.append(days_since_last(a, date, df))
    home_adv.append(home_advantage_score(h, date, df, lookback=WINDOW))
    away_adv.append(home_advantage_score(a, date, df, lookback=WINDOW))

# Attach
df['HomeLastResult'] = home_last_res
df['AwayLastResult'] = away_last_res
df['HomeWinStreak_L5'] = home_streak
df['AwayWinStreak_L5'] = away_streak

df['h2h_home_wins'] = h2h_home_wins
df['h2h_away_wins'] = h2h_away_wins
df['h2h_draws'] = h2h_draws

df['HomeRestDays'] = home_rest
df['AwayRestDays'] = away_rest

df['HomeAdvantageScore'] = home_adv
df['AwayAdvantageScore'] = away_adv

# Form difference features
print('➕ Menambahkan fitur FormDiff (selisih performa Home-Away)...')
df['FormDiff_WinRate'] = df[f'WinRate_Home_L{WINDOW}'] - df[f'WinRate_Away_L{WINDOW}']
df['FormDiff_Goals'] = df[f'AvgGoalsFor_Home_L{WINDOW}'] - df[f'AvgGoalsFor_Away_L{WINDOW}']
df['FormDiff_EWM_Goals'] = df[f'EWM_GoalsFor_Home_span{EWM_SPAN}'] - df[f'EWM_GoalsFor_Away_span{EWM_SPAN}']

# -------------------------
# 4) Final features & cleaning
# -------------------------
features = [
    # rolling home
    f'AvgGoalsFor_Home_L{WINDOW}', f'AvgGoalsAgainst_Home_L{WINDOW}', f'AvgShotsFor_Home_L{WINDOW}', f'AvgSOTFor_Home_L{WINDOW}', f'WinRate_Home_L{WINDOW}', f'DrawRate_Home_L{WINDOW}',
    # ewm home
    f'EWM_GoalsFor_Home_span{EWM_SPAN}', f'EWM_WinRate_Home_span{EWM_SPAN}',
    'HomeLastResult', 'HomeWinStreak_L5', 'HomeRestDays', 'HomeAdvantageScore',
    # rolling away
    f'AvgGoalsFor_Away_L{WINDOW}', f'AvgGoalsAgainst_Away_L{WINDOW}', f'AvgShotsFor_Away_L{WINDOW}', f'AvgSOTFor_Away_L{WINDOW}', f'WinRate_Away_L{WINDOW}', f'DrawRate_Away_L{WINDOW}',
    # ewm away
    f'EWM_GoalsFor_Away_span{EWM_SPAN}', f'EWM_WinRate_Away_span{EWM_SPAN}',
    'AwayLastResult', 'AwayWinStreak_L5', 'AwayRestDays', 'AwayAdvantageScore',
    # H2H & diffs
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'FormDiff_WinRate', 'FormDiff_Goals', 'FormDiff_EWM_Goals'
]

print('🧹 Membersihkan & menyiapkan fitur akhir...')
df_model = df.dropna(subset=features + ['Result']).copy()
X = df_model[features]
y = df_model['Result']
print(f"Dataset untuk modeling: {X.shape[0]} baris x {X.shape[1]} fitur")

# -------------------------
# 5) Hyperparameter tuning (Optuna) - optional
# -------------------------

def run_optuna_xgb(X_train, y_train, n_trials=40):
    if not HAS_OPTUNA or not HAS_XGB:
        print('⚠️ Optuna atau XGBoost tidak tersedia, melewati tuning Optuna.')
        return None
    print('🔎 Menjalankan Optuna tuning untuk XGBoost (TimeSeries CV)...')
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0)
        }
        scores = []
        for train_idx, test_idx in tscv.split(X_train):
            m = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, **params)
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            ypred = m.predict(X_train.iloc[test_idx])
            scores.append(accuracy_score(y_train.iloc[test_idx], ypred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print('🔔 Optuna selesai. Best params:', study.best_params)
    return study.best_params

# -------------------------
# 6) Build final model (ensemble) and evaluate on temporal hold-out
# -------------------------
print('🧪 Menyiapkan model ensemble dan training final...')
split_idx = int(len(df_model) * 0.8)
X_train_final, y_train_final = X.iloc[:split_idx], y.iloc[:split_idx]
X_test_final, y_test_final = X.iloc[split_idx:], y.iloc[split_idx:]

# Try tuning XGB on training portion
best_xgb_params = None
if HAS_OPTUNA and HAS_XGB:
    try:
        best_xgb_params = run_optuna_xgb(X_train_final, y_train_final, n_trials=30)
    except Exception as e:
        print('Info: Optuna tuning gagal atau memakan waktu — memakai default params. Error:', e)

if HAS_XGB:
    xgb_params = {'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8}
    if best_xgb_params:
        xgb_params.update(best_xgb_params)
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, **xgb_params)
else:
    xgb_clf = None

rf_clf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, class_weight='balanced')

estimators = []
if xgb_clf is not None:
    estimators.append(('xgb', xgb_clf))
estimators.append(('rf', rf_clf))

ensemble = VotingClassifier(estimators=estimators, voting='soft')

pipeline = Pipeline([('scaler', StandardScaler()), ('clf', ensemble)])

print('🔁 Melatih pipeline ensemble pada 80% data training...')
pipeline.fit(X_train_final, y_train_final)

print('📈 Evaluasi akhir pada 20% hold-out...')
y_pred = pipeline.predict(X_test_final)
try:
    y_proba = pipeline.predict_proba(X_test_final)
except Exception:
    # predict_proba mungkin tidak tersedia (mis. stacking tanpa prob). Tangani gracefully
    y_proba = None

acc = accuracy_score(y_test_final, y_pred)
bal_acc = balanced_accuracy_score(y_test_final, y_pred)
ll = log_loss(y_test_final, y_proba) if y_proba is not None else np.nan
print('\n📊 Evaluasi Akhir (Hold-out temporal):')
print(f"  - Accuracy        : {acc*100:.2f}%")
print(f"  - Balanced Acc.   : {bal_acc*100:.2f}%")
if not np.isnan(ll):
    print(f"  - Log Loss        : {ll:.4f}")

print('\nClassification Report:\n')
print(classification_report(y_test_final, y_pred, target_names=['Draw','Home','Away']))
cm = confusion_matrix(y_test_final, y_pred)
print('Confusion Matrix (rows true, cols pred):\n', cm)

# -------------------------
# 7) Predict function (dinamis) menggunakan feature terakhir (rolling & ewm tail)
# -------------------------
print('\n🔮 Menyediakan fungsi predict_match (dinamis) versi v3...')
team_latest_stats = rolling_stats.sort_values('Date').groupby('Team').tail(1).set_index('Team')
# for ewm, take the latest per team
team_ewm_latest = ewm_stats.sort_values('Date').groupby('Team').tail(1).set_index('Team')
full_match_history = df_model.copy()

name_map = {"Man Utd": "Man United", "Spurs": "Tottenham", "Nottingham Forest": "Nott'm Forest"}

def predict_match(home_team_input, away_team_input, model=pipeline, team_roll_df=team_latest_stats, team_ewm_df=team_ewm_latest, history_df=full_match_history):
    home = name_map.get(home_team_input, home_team_input)
    away = name_map.get(away_team_input, away_team_input)
    print('='*40)
    print(f'PREDIKSI: {home_team_input} vs {away_team_input}')
    print('='*40)
    if home not in team_roll_df.index or away not in team_roll_df.index:
        print('⚠️ Data form/rolling tidak ditemukan untuk salah satu tim.'); return None

    h = team_roll_df.loc[home]
    a = team_roll_df.loc[away]
    he = team_ewm_df.loc[home]
    ae = team_ewm_df.loc[away]

    today = pd.to_datetime('today')
    h_last_res = get_last_result(home, today, history_df)
    a_last_res = get_last_result(away, today, history_df)
    h_streak = get_recent_streak(home, today, history_df, WINDOW)
    a_streak = get_recent_streak(away, today, history_df, WINDOW)
    h2h_hw, h2h_aw, h2h_d = calc_h2h_counts(home, away, today, history_df)
    h_rest = days_since_last(home, today, history_df)
    a_rest = days_since_last(away, today, history_df)
    h_adv = home_advantage_score(home, today, history_df, lookback=WINDOW)
    a_adv = home_advantage_score(away, today, history_df, lookback=WINDOW)

    feat_vals = [
        h[f'AvgGoalsFor_L{WINDOW}'], h[f'AvgGoalsAgainst_L{WINDOW}'], h[f'AvgShotsFor_L{WINDOW}'], h[f'AvgSOTFor_L{WINDOW}'], h[f'WinRate_L{WINDOW}'], h[f'DrawRate_L{WINDOW}'],
        he[f'EWM_GoalsFor_span{EWM_SPAN}'] , he[f'EWM_WinRate_span{EWM_SPAN}'],
        h_last_res, h_streak, h_rest, h_adv,
        a[f'AvgGoalsFor_L{WINDOW}'], a[f'AvgGoalsAgainst_L{WINDOW}'], a[f'AvgShotsFor_L{WINDOW}'], a[f'AvgSOTFor_L{WINDOW}'], a[f'WinRate_L{WINDOW}'], a[f'DrawRate_L{WINDOW}'],
        ae[f'EWM_GoalsFor_span{EWM_SPAN}'] , ae[f'EWM_WinRate_span{EWM_SPAN}'],
        a_last_res, a_streak, a_rest, a_adv,
        h2h_hw, h2h_aw, h2h_d,
        h[f'AvgGoalsFor_L{WINDOW}'] - a[f'AvgGoalsFor_L{WINDOW}'], # FormDiff_Goals (fallback)
        he[f'EWM_GoalsFor_span{EWM_SPAN}'] - ae[f'EWM_GoalsFor_span{EWM_SPAN}']
    ]

    # Map columns to features list (support backward compatibility)
    cols_for_input = features.copy()
    X_input = pd.DataFrame([feat_vals], columns=cols_for_input)

    probs = model.predict_proba(X_input)[0]
    print(f"Info Form (Home): WinRate={h[f'WinRate_L{WINDOW}']:.2f}, EWM_Goals={he[f'EWM_GoalsFor_span{EWM_SPAN}']:.2f}, LastResult={h_last_res}, RestDays={h_rest}")
    print(f"Info Form (Away): WinRate={a[f'WinRate_L{WINDOW}']:.2f}, EWM_Goals={ae[f'EWM_GoalsFor_span{EWM_SPAN}']:.2f}, LastResult={a_last_res}, RestDays={a_rest}")
    print(f"Info H2H: {home} menang {h2h_hw}, {away} menang {h2h_aw}, seri {h2h_d}")
    print(f"\nProbabilities -> Draw: {probs[0]*100:.2f}%, Home: {probs[1]*100:.2f}%, Away: {probs[2]*100:.2f}%")
    pred_class = np.argmax(probs)
    label_map = {0: 'Seri', 1: home_team_input + ' Menang', 2: away_team_input + ' Menang'}
    print('Kesimpulan:', label_map[pred_class])
    return probs

# contoh prediksi (aman jika team ada di data latest)
try:
    predict_match('Aston Villa', 'Burnley')
    predict_match('Wolves', 'Brighton')
except Exception as e:
    print('Info: contoh prediksi gagal -', e)

# Simpan model
joblib.dump(pipeline, 'model_epl_final_v3.joblib')
print('\n💾 Model tersimpan sebagai "model_epl_final_v3.joblib"')
