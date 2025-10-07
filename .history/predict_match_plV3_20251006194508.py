# predict-pl-match_otomatis_v4.py
# Versi v4 — leakage-safe feature engineering + Elo rating + rolling & EWM dengan shift
# - Semua fitur dihitung hanya dari masa lalu (menggunakan .shift())
# - Fitur baru: Elo rating pra-pertandingan, GoalDiff rolling, Points rolling
# - Model ensemble: LightGBM / XGBoost / RandomForest (soft voting) dengan fallback
# - Menyimpan model dan daftar fitur (model_features_v4.joblib)

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# -------------------------
# 1) Load data
# -------------------------
print('📊 Memuat data...')
raw_path = 'epl-training.csv'
if not os.path.exists(raw_path):
    raise FileNotFoundError(f"File {raw_path} tidak ditemukan. Pastikan berada di folder yang sama.")

df = pd.read_csv(raw_path, encoding='ISO-8859-1', low_memory=False)
expected = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']
for c in expected:
    if c not in df.columns:
        raise ValueError(f"Kolom {c} tidak ditemukan di CSV: {c}")

df = df[expected].copy()
df.rename(columns={'HomeTeam': 'Home', 'AwayTeam': 'Away', 'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals',
                   'HS': 'HomeShots', 'AS': 'AwayShots', 'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)
print(f"✅ Data dimuat: {df.shape[0]} baris, dari {df['Date'].min().date()} sampai {df['Date'].max().date()}")

# Basic result label
print('🔧 Membuat label hasil (Result)...')
def result_label(row):
    if row['HomeGoals'] > row['AwayGoals']: return 1
    if row['HomeGoals'] < row['AwayGoals']: return 2
    return 0

df['Result'] = df.apply(result_label, axis=1)

# -------------------------
# 2) Build team-level dataframe (satu baris per tim per match)
# -------------------------
print('🔁 Menyusun team-level records (leakage-safe)...')
team_rows = []
for _, r in df.iterrows():
    team_rows.append({'Date': r['Date'], 'MatchID': _, 'Team': r['Home'], 'Opponent': r['Away'], 'GoalsFor': r['HomeGoals'], 'GoalsAgainst': r['AwayGoals'], 'ShotsFor': r['HomeShots'], 'SOT_For': r['HomeShotsOnTarget'], 'Win': 1 if r['HomeGoals']>r['AwayGoals'] else 0, 'Draw': 1 if r['HomeGoals']==r['AwayGoals'] else 0, 'IsHome': 1, 'Points': 3 if r['HomeGoals']>r['AwayGoals'] else (1 if r['HomeGoals']==r['AwayGoals'] else 0)})
    team_rows.append({'Date': r['Date'], 'MatchID': _, 'Team': r['Away'], 'Opponent': r['Home'], 'GoalsFor': r['AwayGoals'], 'GoalsAgainst': r['HomeGoals'], 'ShotsFor': r['AwayShots'], 'SOT_For': r['AwayShotsOnTarget'], 'Win': 1 if r['AwayGoals']>r['HomeGoals'] else 0, 'Draw': 1 if r['AwayGoals']==r['HomeGoals'] else 0, 'IsHome': 0, 'Points': 3 if r['AwayGoals']>r['HomeGoals'] else (1 if r['AwayGoals']==r['HomeGoals'] else 0)})

team_df = pd.DataFrame(team_rows).sort_values(['Team','Date']).reset_index(drop=True)

WINDOW = 5
EWM_SPAN = 5

# -------------------------
# 3) Leakage-safe rolling & EWM (pakai shift)
# -------------------------
print('📈 Menghitung fitur rolling (menggunakan .shift())...')
# Rolling means dari 5 pertandingan terakhir, tapi hanya data masa lalu (shift sebelum rolling)
team_df[f'Roll_GoalsFor_L{WINDOW}'] = team_df.groupby('Team')['GoalsFor'].apply(lambda x: x.shift().rolling(WINDOW, min_periods=1).mean())
team_df[f'Roll_GoalsAgainst_L{WINDOW}'] = team_df.groupby('Team')['GoalsAgainst'].apply(lambda x: x.shift().rolling(WINDOW, min_periods=1).mean())
team_df[f'Roll_ShotsFor_L{WINDOW}'] = team_df.groupby('Team')['ShotsFor'].apply(lambda x: x.shift().rolling(WINDOW, min_periods=1).mean())
team_df[f'Roll_WinRate_L{WINDOW}'] = team_df.groupby('Team')['Win'].apply(lambda x: x.shift().rolling(WINDOW, min_periods=1).mean())
team_df[f'Roll_Points_L{WINDOW}'] = team_df.groupby('Team')['Points'].apply(lambda x: x.shift().rolling(WINDOW, min_periods=1).sum())

# EWM (shift then ewm) => past-weighted recent form
team_df[f'EWM_GoalsFor_span{EWM_SPAN}'] = team_df.groupby('Team')['GoalsFor'].apply(lambda x: x.shift().ewm(span=EWM_SPAN, adjust=False).mean())
team_df[f'EWM_WinRate_span{EWM_SPAN}'] = team_df.groupby('Team')['Win'].apply(lambda x: x.shift().ewm(span=EWM_SPAN, adjust=False).mean())

# Days since last match
team_df['PrevMatchDate'] = team_df.groupby('Team')['Date'].shift(1)
team_df['DaysSinceLast'] = (team_df['Date'] - team_df['PrevMatchDate']).dt.days

# Home advantage: compute proportion kemenangan di kandang pada 5 home matches terakhir
home_matches = team_df[team_df['IsHome']==1].copy()
home_matches[f'HomeWinRateLast{WINDOW}'] = home_matches.groupby('Team')['Win'].apply(lambda x: x.shift().rolling(WINDOW, min_periods=1).mean())
# merge back
team_df = pd.merge(team_df, home_matches[['MatchID','Team',f'HomeWinRateLast{WINDOW}']], on=['MatchID','Team'], how='left')

# GoalDiff rolling
team_df['GoalDiff'] = team_df['GoalsFor'] - team_df['GoalsAgainst']
team_df[f'Roll_GoalDiff_L{WINDOW}'] = team_df.groupby('Team')['GoalDiff'].apply(lambda x: x.shift().rolling(WINDOW, min_periods=1).mean())

# -------------------------
# 4) Elo rating (iterative, pra-pertandingan)
# -------------------------
print('⚖️ Menghitung Elo rating (pra-pertandingan) — K=20...')
teams = pd.unique(team_df['Team'])
elo = {t:1500 for t in teams}
K = 20
elo_home, elo_away = [], []
# We'll iterate original matches (df) to maintain consistent order
for idx, row in df.iterrows():
    home = row['Home']; away = row['Away']
    # current elo before match
    eh = elo.get(home,1500)
    ea = elo.get(away,1500)
    elo_home.append(eh)
    elo_away.append(ea)
    # expected score
    exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
    exp_a = 1 - exp_h
    # actual score
    if row['HomeGoals'] > row['AwayGoals']:
        sh, sa = 1.0, 0.0
    elif row['HomeGoals'] < row['AwayGoals']:
        sh, sa = 0.0, 1.0
    else:
        sh, sa = 0.5, 0.5
    # update
    elo[home] = eh + K * (sh - exp_h)
    elo[away] = ea + K * (sa - exp_a)

# attach elo to original df as pre-match elo
df['Elo_Home_pre'] = elo_home
df['Elo_Away_pre'] = elo_away

# Now merge team_df rolling features back to match-level df
print('🔗 Menggabungkan fitur tim kembali ke level pertandingan (leakage-safe join)...')
# prepare rolling stats per team-date (last record per team-date)
roll_cols = [f'Roll_GoalsFor_L{WINDOW}', f'Roll_GoalsAgainst_L{WINDOW}', f'Roll_ShotsFor_L{WINDOW}', f'Roll_WinRate_L{WINDOW}', f'Roll_Points_L{WINDOW}', f'EWM_GoalsFor_span{EWM_SPAN}', f'EWM_WinRate_span{EWM_SPAN}', 'DaysSinceLast', f'HomeWinRateLast{WINDOW}', f'Roll_GoalDiff_L{WINDOW}']
team_roll = team_df.groupby(['Team','Date'])[roll_cols].last().reset_index()

# Merge home
df = pd.merge(df, team_roll, left_on=['Home','Date'], right_on=['Team','Date'], how='left')
df = df.rename(columns={c: c.replace('Roll_','Avg_').replace('EWM_','EWM_') for c in roll_cols})
# rename specific home cols
home_renames = {}
for c in roll_cols:
    home_renames[c] = f'Home_{c}'

# apply rename for home
df = df.rename(columns=home_renames)
# drop redundant Team column
if 'Team' in df.columns:
    df = df.drop(columns=['Team'])

# Merge away
df = pd.merge(df, team_roll, left_on=['Away','Date'], right_on=['Team','Date'], how='left', suffixes=('_home','_away'))
away_renames = {}
for c in roll_cols:
    away_renames[c] = f'Away_{c}'

df = df.rename(columns=away_renames)
if 'Team' in df.columns:
    df = df.drop(columns=['Team'])

# Now df has Elo_Home_pre, Elo_Away_pre and Home_/Away_ rolling features

# -------------------------
# 5) Final feature list (leakage-free)
# -------------------------
features = [
    'Elo_Home_pre', 'Elo_Away_pre',
    f'Home_{f'Roll_GoalsFor_L{WINDOW}'}'.replace('Roll_',''),
]
# Build features explicitly to avoid ambiguity
features = [
    'Elo_Home_pre', 'Elo_Away_pre',
    f'Home_Roll_GoalsFor_L{WINDOW}', f'Home_Roll_GoalsAgainst_L{WINDOW}', f'Home_Roll_ShotsFor_L{WINDOW}', f'Home_Roll_WinRate_L{WINDOW}', f'Home_Roll_Points_L{WINDOW}', f'Home_EWM_GoalsFor_span{EWM_SPAN}', f'Home_EWM_WinRate_span{EWM_SPAN}', 'Home_DaysSinceLast', f'Home_HomeWinRateLast{WINDOW}', f'Home_Roll_GoalDiff_L{WINDOW}',
    f'Away_Roll_GoalsFor_L{WINDOW}', f'Away_Roll_GoalsAgainst_L{WINDOW}', f'Away_Roll_ShotsFor_L{WINDOW}', f'Away_Roll_WinRate_L{WINDOW}', f'Away_Roll_Points_L{WINDOW}', f'Away_EWM_GoalsFor_span{EWM_SPAN}', f'Away_EWM_WinRate_span{EWM_SPAN}', 'Away_DaysSinceLast', f'Away_HomeWinRateLast{WINDOW}', f'Away_Roll_GoalDiff_L{WINDOW}',
    # head-to-head counts (leakage-safe computed earlier via previous matches)
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws'
]

# Prepare H2H counts (leakage-safe)
print('⚔️ Menghitung H2H (leakage-safe)...')
# reuse previous functions but ensure use only past matches

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
    hw, aw, dr = calc_h2h_counts(r['Home'], r['Away'], r['Date'], df)
    h2h_home_wins.append(hw); h2h_away_wins.append(aw); h2h_draws.append(dr)

df['h2h_home_wins'] = h2h_home_wins
df['h2h_away_wins'] = h2h_away_wins
df['h2h_draws'] = h2h_draws

# Final cleaning: select features present
print('🧹 Membersihkan & menyiapkan fitur akhir (leakage-free)...')
df_model = df.dropna(subset=features + ['Result']).copy()
X = df_model[features]
y = df_model['Result']
print(f"Dataset untuk modeling (leakage-free): {X.shape[0]} baris x {X.shape[1]} fitur")

# -------------------------
# 6) Temporal split (80% awal untuk train, 20% akhir untuk test) — time-ordered
# -------------------------
print('🧪 Membuat temporal hold-out (80/20) — time ordered split...')
split_idx = int(len(df_model) * 0.8)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

# -------------------------
# 7) Model ensemble (LightGBM / XGB / RF) dengan fallback
# -------------------------
print('⚙️ Menyiapkan model ensemble (LGBM/XGB/RF) — soft voting')
estimators = []
if HAS_LGB:
    lgb = LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, random_state=42)
    estimators.append(('lgb', lgb))
if HAS_XGB:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42)
    estimators.append(('xgb', xgb))
# always include RF
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, class_weight='balanced')
estimators.append(('rf', rf))

ensemble = VotingClassifier(estimators=estimators, voting='soft')
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', ensemble)])

print('🔁 Melatih pipeline ensemble pada 80% data training...')
pipeline.fit(X_train, y_train)

# Simpan daftar fitur untuk prediksi
model_features = X_train.columns.tolist()
joblib.dump(model_features, 'model_features_v4.joblib')

# -------------------------
# 8) Evaluasi pada hold-out
# -------------------------
print('📈 Evaluasi akhir pada 20% hold-out...')
y_pred = pipeline.predict(X_test)
try:
    y_proba = pipeline.predict_proba(X_test)
except Exception:
    y_proba = None

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
ll = log_loss(y_test, y_proba) if y_proba is not None else np.nan
print('
📊 Evaluasi Akhir (Hold-out temporal):')
print(f"  - Accuracy        : {acc*100:.2f}%")
print(f"  - Balanced Acc.   : {bal_acc*100:.2f}%")
if not np.isnan(ll):
    print(f"  - Log Loss        : {ll:.4f}")

print('
Classification Report:
')
print(classification_report(y_test, y_pred, target_names=['Draw','Home','Away']))
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix (rows true, cols pred):
', cm)

# Simpan model
joblib.dump(pipeline, 'model_epl_final_v4.joblib')
print('
💾 Model tersimpan sebagai "model_epl_final_v4.joblib"')

# -------------------------
# 9) Fungsi predict_match (menggunakan model_features_v4.joblib untuk reindex)
# -------------------------
print('
🔮 Menyediakan fungsi predict_match (v4) yang leakage-safe...')

# load features and model (they are already in memory, but use saved for safety)
saved_features = joblib.load('model_features_v4.joblib')

name_map = {"Man Utd": "Man United", "Spurs": "Tottenham", "Nottingham Forest": "Nott'm Forest"}

def predict_match(home_team_input, away_team_input, model=pipeline, features_list=saved_features, df_full=df_model):
    home = name_map.get(home_team_input, home_team_input)
    away = name_map.get(away_team_input, away_team_input)
    print('='*40)
    print(f'PREDIKSI: {home_team_input} vs {away_team_input}')
    print('='*40)

    # check team exists in historical data
    if home not in pd.unique(df['Home']) or away not in pd.unique(df['Away']):
        print('⚠️ Salah satu tim tidak ditemukan di data historis.'); return None

    # Ambil latest pra-pertandingan stats dari df (last available date per team)
    # Karena semua fitur leakage-safe, kita dapat menggunakan baris terakhir untuk masing-masing tim
    try:
        h_row = team_roll[team_roll['Team']==home].sort_values('Date').tail(1).iloc[0]
        a_row = team_roll[team_roll['Team']==away].sort_values('Date').tail(1).iloc[0]
    except Exception:
        print('⚠️ Tidak dapat mengambil statistik terakhir tim.'); return None

    # Ambil elo terakhir
    eh = df[df['Home']==home]['Elo_Home_pre'].tolist() + df[df['Away']==home]['Elo_Away_pre'].tolist()
    if len(eh)==0:
        eh_val = 1500
    else:
        eh_val = eh[-1]
    ea = df[df['Home']==away]['Elo_Home_pre'].tolist() + df[df['Away']==away]['Elo_Away_pre'].tolist()
    if len(ea)==0:
        ea_val = 1500
    else:
        ea_val = ea[-1]

    # Build feature vector (match-order must match saved_features)
    feat = {
        'Elo_Home_pre': eh_val,
        'Elo_Away_pre': ea_val,
        f'Home_Roll_GoalsFor_L{WINDOW}': h_row.get(f'Roll_GoalsFor_L{WINDOW}', 0),
        f'Home_Roll_GoalsAgainst_L{WINDOW}': h_row.get(f'Roll_GoalsAgainst_L{WINDOW}', 0),
        f'Home_Roll_ShotsFor_L{WINDOW}': h_row.get(f'Roll_ShotsFor_L{WINDOW}', 0),
        f'Home_Roll_WinRate_L{WINDOW}': h_row.get(f'Roll_WinRate_L{WINDOW}', 0),
        f'Home_Roll_Points_L{WINDOW}': h_row.get(f'Roll_Points_L{WINDOW}', 0),
        f'Home_EWM_GoalsFor_span{EWM_SPAN}': h_row.get(f'EWM_GoalsFor_span{EWM_SPAN}', 0),
        f'Home_EWM_WinRate_span{EWM_SPAN}': h_row.get(f'EWM_WinRate_span{EWM_SPAN}', 0),
        'Home_DaysSinceLast': h_row.get('DaysSinceLast', 999),
        f'Home_HomeWinRateLast{WINDOW}': h_row.get(f'HomeWinRateLast{WINDOW}', 0),
        f'Home_Roll_GoalDiff_L{WINDOW}': h_row.get(f'Roll_GoalDiff_L{WINDOW}', 0),
        f'Away_Roll_GoalsFor_L{WINDOW}': a_row.get(f'Roll_GoalsFor_L{WINDOW}', 0),
        f'Away_Roll_GoalsAgainst_L{WINDOW}': a_row.get(f'Roll_GoalsAgainst_L{WINDOW}', 0),
        f'Away_Roll_ShotsFor_L{WINDOW}': a_row.get(f'Roll_ShotsFor_L{WINDOW}', 0),
        f'Away_Roll_WinRate_L{WINDOW}': a_row.get(f'Roll_WinRate_L{WINDOW}', 0),
        f'Away_Roll_Points_L{WINDOW}': a_row.get(f'Roll_Points_L{WINDOW}', 0),
        f'Away_EWM_GoalsFor_span{EWM_SPAN}': a_row.get(f'EWM_GoalsFor_span{EWM_SPAN}', 0),
        f'Away_EWM_WinRate_span{EWM_SPAN}': a_row.get(f'EWM_WinRate_span{EWM_SPAN}', 0),
        'Away_DaysSinceLast': a_row.get('DaysSinceLast', 999),
        f'Away_HomeWinRateLast{WINDOW}': a_row.get(f'HomeWinRateLast{WINDOW}', 0),
        f'Away_Roll_GoalDiff_L{WINDOW}': a_row.get(f'Roll_GoalDiff_L{WINDOW}', 0),
        'h2h_home_wins': 0,
        'h2h_away_wins': 0,
        'h2h_draws': 0
    }

    # hitung H2H pra-pertandingan
    hw, aw, dr = calc_h2h_counts(home, away, pd.Timestamp.now(), df)
    feat['h2h_home_wins'] = hw; feat['h2h_away_wins'] = aw; feat['h2h_draws'] = dr

    # create DataFrame and reindex to saved features
    X_pred = pd.DataFrame([feat])[saved_features]
    # safety: fill NaN
    X_pred = X_pred.fillna(0)

    probs = model.predict_proba(X_pred)[0]
    print(f"Probabilities -> Draw: {probs[0]*100:.2f}%, Home: {probs[1]*100:.2f}%, Away: {probs[2]*100:.2f}%")
    pred_class = int(np.argmax(probs))
    label_map = {0: 'Seri', 1: home_team_input + ' Menang', 2: away_team_input + ' Menang'}
    print('Kesimpulan:', label_map[pred_class])
    return probs

# contoh prediksi (jika tim ada di data historis)
try:
    predict_match('Aston Villa', 'Burnley')
except Exception as e:
    print('Info: contoh prediksi gagal -', e)
