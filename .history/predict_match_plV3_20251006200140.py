# ==========================================
# Premier League Match Predictor v3 (Stable)
# ==========================================
# by decoder + GPT-5
# Leakage-safe | Rolling Stats | Elo Rating | H2H | RF Model
# Target: Result (0=Away, 1=Draw, 2=Home)
# Accuracy goal: 85–95%

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. LOAD DATA
# ==============================
print("📊 Memuat data...")
df = pd.read_csv('historical_matches.csv', parse_dates=['Date'])
df = df.sort_values('Date')
print(f"✅ Data dimuat: {len(df)} baris, dari {df['Date'].min().date()} sampai {df['Date'].max().date()}")

# ==============================
# 2. BUAT LABEL RESULT
# ==============================
print("🔧 Membuat label hasil (Result)...")
def get_result(row):
    if row['FTHG'] > row['FTAG']:
        return 2  # Home Win
    elif row['FTHG'] < row['FTAG']:
        return 0  # Away Win
    else:
        return 1  # Draw

df['Result'] = df.apply(get_result, axis=1)

# ==============================
# 3. TEAM-LEVEL RECORDS (LEAKAGE SAFE)
# ==============================
print("🔁 Menyusun team-level records (leakage-safe)...")
team_stats = []

for team in pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel()):
    team_df = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
    team_df['is_home'] = team_df['HomeTeam'] == team
    team_df['GoalsFor'] = np.where(team_df['is_home'], team_df['FTHG'], team_df['FTAG'])
    team_df['GoalsAgainst'] = np.where(team_df['is_home'], team_df['FTAG'], team_df['FTHG'])
    team_df['Points'] = np.where(team_df['GoalsFor'] > team_df['GoalsAgainst'], 3,
                                 np.where(team_df['GoalsFor'] == team_df['GoalsAgainst'], 1, 0))
    team_df['Win'] = np.where(team_df['Points'] == 3, 1, 0)
    team_df['Loss'] = np.where(team_df['Points'] == 0, 1, 0)
    team_df['Draw'] = np.where(team_df['Points'] == 1, 1, 0)
    team_df['Team'] = team
    team_df = team_df.sort_values('Date')

    # ==============================
    # 4. ROLLING FORM FEATURES (L5)
    # ==============================
    WINDOW = 5
    for col in ['GoalsFor', 'GoalsAgainst', 'Win', 'Points']:
        team_df[f'Roll_{col}_L{WINDOW}'] = (
            team_df[col].shift(1).rolling(WINDOW, min_periods=1).mean()
        )

    team_df[f'Roll_WinRate_L{WINDOW}'] = (
        team_df['Win'].shift(1).rolling(WINDOW, min_periods=1).mean()
    )
    team_df[f'Roll_GoalDiff_L{WINDOW}'] = (
        (team_df['GoalsFor'] - team_df['GoalsAgainst']).shift(1).rolling(WINDOW, min_periods=1).mean()
    )

    team_stats.append(team_df)

team_stats = pd.concat(team_stats)

# ==============================
# 5. ELO RATING SYSTEM
# ==============================
print("⚖️ Menghitung Elo rating (pra-pertandingan) — K=20...")
K = 20
elos = {t: 1500 for t in df['HomeTeam'].unique()}

elo_records = []
for _, row in df.iterrows():
    home, away = row['HomeTeam'], row['AwayTeam']
    home_elo, away_elo = elos[home], elos[away]
    exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    exp_away = 1 - exp_home

    if row['FTHG'] > row['FTAG']:
        score_home, score_away = 1, 0
    elif row['FTHG'] < row['FTAG']:
        score_home, score_away = 0, 1
    else:
        score_home, score_away = 0.5, 0.5

    elos[home] = home_elo + K * (score_home - exp_home)
    elos[away] = away_elo + K * (score_away - exp_away)

    elo_records.append({
        'Date': row['Date'],
        'HomeTeam': home,
        'AwayTeam': away,
        'HomeElo': home_elo,
        'AwayElo': away_elo
    })

elo_df = pd.DataFrame(elo_records)

# ==============================
# 6. MERGE ALL FEATURES
# ==============================
print("🔗 Menggabungkan fitur tim kembali ke level pertandingan (leakage-safe join)...")

def join_team_features(df_base, team_stats, side):
    rename_dict = {col: f'{side}_{col}' for col in team_stats.columns if col not in ['Date', 'Team']}
    merged = df_base.merge(
        team_stats.rename(columns=rename_dict),
        how='left',
        left_on=['Date', f'{side}Team'],
        right_on=['Date', f'{side}_Team']
    )
    return merged

df = join_team_features(df, team_stats, 'Home')
df = join_team_features(df, team_stats, 'Away')
df = df.merge(elo_df, on=['Date', 'HomeTeam', 'AwayTeam'], how='left')

# ==============================
# 7. HEAD-TO-HEAD FEATURES
# ==============================
print("⚔️ Menghitung H2H (leakage-safe)...")
df['H2H_HomeWins'] = df.groupby(['HomeTeam', 'AwayTeam'])['Result'].apply(lambda x: (x.shift(1) == 2).cumsum())
df['H2H_AwayWins'] = df.groupby(['HomeTeam', 'AwayTeam'])['Result'].apply(lambda x: (x.shift(1) == 0).cumsum())

# ==============================
# 8. FINAL CLEANING
# ==============================
print("🧹 Membersihkan & menyiapkan fitur akhir (leakage-free)...")
features = [
    'HomeElo', 'AwayElo', 'H2H_HomeWins', 'H2H_AwayWins',
    'Home_Roll_GoalsFor_L5', 'Home_Roll_GoalsAgainst_L5', 'Home_Roll_WinRate_L5',
    'Home_Roll_Points_L5', 'Home_Roll_GoalDiff_L5',
    'Away_Roll_GoalsFor_L5', 'Away_Roll_GoalsAgainst_L5', 'Away_Roll_WinRate_L5',
    'Away_Roll_Points_L5', 'Away_Roll_GoalDiff_L5'
]

df_model = df.dropna(subset=features + ['Result']).copy()

X = df_model[features]
y = df_model['Result']

# ==============================
# 9. SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ==============================
# 10. TRAIN MODEL
# ==============================
print("🤖 Melatih model RandomForest (n_estimators=200)...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced_subsample'
)
model.fit(X_train, y_train)

# ==============================
# 11. EVALUASI
# ==============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Akurasi: {acc*100:.2f}%")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==============================
# 12. SIMPAN MODEL
# ==============================
import joblib
joblib.dump(model, 'rf_match_predictor_v3.pkl')
print("\n💾 Model disimpan sebagai rf_match_predictor_v3.pkl")
