import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ================================
# Bagian 1, 2, 3, 4 (Memuat & Feature Engineering) tidak berubah
# ================================
print("📊 Memuat data pertandingan historis dari Kaggle...")
try:
    hist_df = pd.read_csv("epl-training.csv")
except FileNotFoundError:
    print("❌ Gagal: File 'epl-training.csv' tidak ditemukan."); exit()
print(f"✅ Berhasil memuat data historis: {hist_df.shape[0]} pertandingan")

print("\n✨ Menyesuaikan data dari Kaggle...")
relevant_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY']
hist_df = hist_df[relevant_cols]
hist_df.rename(columns={'HomeTeam': 'Home', 'AwayTeam': 'Away', 'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals', 'HS': 'HomeShots', 'AS': 'AwayShots', 'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget', 'HC': 'HomeCorners', 'AC': 'AwayCorners', 'HF': 'HomeFouls', 'AF': 'AwayFouls', 'HY': 'HomeYellowCards', 'AY': 'AwayYellowCards'}, inplace=True)
stats_cols = ['HomeGoals', 'AwayGoals', 'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners', 'AwayCorners', 'HomeFouls', 'AwayFouls', 'HomeYellowCards', 'AwayYellowCards']
for col in stats_cols:
    hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')
hist_df.dropna(subset=stats_cols, inplace=True)
hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='%d/%m/%Y')
hist_df = hist_df.sort_values(by='Date').reset_index(drop=True)
print("✅ Data berhasil dibersihkan dan disesuaikan.")

print("\n🚀 Membuat fitur 'Team Form' (Performa Terkini)...")
team_stats = []
for index, row in hist_df.iterrows():
    team_stats.append({'Date': row['Date'], 'Team': row['Home'], 'GoalsFor': row['HomeGoals'], 'GoalsAgainst': row['AwayGoals'], 'ShotsFor': row['HomeShots'], 'SOT_For': row['HomeShotsOnTarget'], 'CornersFor': row['HomeCorners'], 'FoulsCommitted': row['HomeFouls'], 'Yellows': row['HomeYellowCards']})
    team_stats.append({'Date': row['Date'], 'Team': row['Away'], 'GoalsFor': row['AwayGoals'], 'GoalsAgainst': row['HomeGoals'], 'ShotsFor': row['AwayShots'], 'SOT_For': row['AwayShotsOnTarget'], 'CornersFor': row['AwayCorners'], 'FoulsCommitted': row['AwayFouls'], 'Yellows': row['AwayYellowCards']})
team_stats_df = pd.DataFrame(team_stats).sort_values(by=['Team', 'Date'])
rolling_stats = team_stats_df.groupby('Team').rolling(window=5, on='Date').mean().reset_index()
rolling_stats.rename(columns={'GoalsFor': 'AvgGoalsFor_L5', 'GoalsAgainst': 'AvgGoalsAgainst_L5', 'ShotsFor': 'AvgShotsFor_L5', 'SOT_For': 'AvgSOT_L5', 'CornersFor': 'AvgCorners_L5', 'FoulsCommitted': 'AvgFouls_L5', 'Yellows': 'AvgYellows_L5'}, inplace=True)
rolling_stats = rolling_stats[['Team', 'Date', 'AvgGoalsFor_L5', 'AvgGoalsAgainst_L5', 'AvgShotsFor_L5', 'AvgSOT_L5', 'AvgCorners_L5', 'AvgFouls_L5', 'AvgYellows_L5']]
home_rolling_stats = rolling_stats.copy()
away_rolling_stats = rolling_stats.copy()
home_rolling_stats.columns = ['Team', 'Date', 'AvgGoalsFor_Home_L5', 'AvgGoalsAgainst_Home_L5', 'AvgShotsFor_Home_L5', 'AvgSOT_Home_L5', 'AvgCorners_Home_L5', 'AvgFouls_Home_L5', 'AvgYellows_Home_L5']
away_rolling_stats.columns = ['Team', 'Date', 'AvgGoalsFor_Away_L5', 'AvgGoalsAgainst_Away_L5', 'AvgShotsFor_Away_L5', 'AvgSOT_Away_L5', 'AvgCorners_Away_L5', 'AvgFouls_Away_L5', 'AvgYellows_Away_L5']
hist_df = pd.merge(hist_df, home_rolling_stats, left_on=['Date', 'Home'], right_on=['Date', 'Team'], how='left').drop('Team', axis=1)
hist_df = pd.merge(hist_df, away_rolling_stats, left_on=['Date', 'Away'], right_on=['Date', 'Team'], how='left').drop('Team', axis=1)
hist_df.dropna(inplace=True)
print("✅ Fitur 'Team Form' yang lebih lengkap berhasil dibuat.")

print("\n🛠️  Membuat fitur Head-to-Head (H2H)...")
def calculate_h2h_stats(home_team, away_team, past_matches_df):
    h2h_matches = past_matches_df[((past_matches_df['Home'] == home_team) & (past_matches_df['Away'] == away_team)) | ((past_matches_df['Home'] == away_team) & (past_matches_df['Away'] == home_team))]
    home_wins, away_wins, draws = 0, 0, 0
    if h2h_matches.empty: return {'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0}
    for _, row in h2h_matches.iterrows():
        if row['HomeGoals'] > row['AwayGoals']:
            if row['Home'] == home_team: home_wins += 1; else: away_wins += 1
        elif row['AwayGoals'] > row['HomeGoals']:
            if row['Away'] == away_team: away_wins += 1; else: home_wins += 1
        else: draws += 1
    return {'h2h_home_wins': home_wins, 'h2h_away_wins': away_wins, 'h2h_draws': draws}
h2h_results = []
for index in hist_df.index:
    row = hist_df.loc[index]
    past_matches = hist_df.loc[hist_df.index < index]
    stats = calculate_h2h_stats(row['Home'], row['Away'], past_matches)
    h2h_results.append(stats)
hist_df = pd.concat([hist_df, pd.DataFrame(h2h_results, index=hist_df.index)], axis=1)
print("✅ Fitur H2H berhasil ditambahkan.")


# ================================
# 5️⃣ Training & Evaluasi dengan MODEL YANG DI-TUNING
# ================================
print("\n⚙️ Mempersiapkan data untuk training model...")
def get_result(row):
    if row["HomeGoals"] > row["AwayGoals"]: return 1;
    elif row["HomeGoals"] < row["AwayGoals"]: return 0;
    else: return 2
hist_df["Result"] = hist_df.apply(get_result, axis=1)

features = [
    'AvgGoalsFor_Home_L5', 'AvgGoalsAgainst_Home_L5', 'AvgShotsFor_Home_L5', 'AvgSOT_Home_L5', 'AvgCorners_Home_L5', 'AvgFouls_Home_L5', 'AvgYellows_Home_L5',
    'AvgGoalsFor_Away_L5', 'AvgGoalsAgainst_Away_L5', 'AvgShotsFor_Away_L5', 'AvgSOT_Away_L5', 'AvgCorners_Away_L5', 'AvgFouls_Away_L5', 'AvgYellows_Away_L5',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws'
]
X = hist_df[features]
y = hist_df["Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Fitur berhasil di-scale.")

print("\n🧠 Melatih model XGBoost yang sudah di-Tuning...")
# === PERUBAHAN: Menambahkan Hyperparameter untuk Tuning ===
model = xgb.XGBClassifier(
    n_estimators=1000,         # Tambah jumlah pohon
    learning_rate=0.01,        # Perkecil learning rate agar lebih hati-hati
    max_depth=3,               # Batasi kedalaman pohon
    subsample=0.8,             # Gunakan 80% data per pohon
    colsample_bytree=0.8,      # Gunakan 80% fitur per pohon
    gamma=0.1,                 # Regularisasi untuk mencegah overfitting
    random_state=42,
    objective='multi:softprob',
    eval_metric='mlogloss',
    use_label_encoder=False
)
model.fit(X_train_scaled, y_train)
print("✅ Model berhasil dilatih.")
# =======================================================

print("\n⚖️ Mengevaluasi akurasi model yang sudah di-Tuning...")
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi Model Secara Keseluruhan: {acc*100:.2f}%")
print("\nLaporan Detail Kinerja Model:\n")
print(classification_report(y_test, y_pred, target_names=['Menang Tamu', 'Menang Tuan Rumah', 'Seri']))


# ================================
# 6️⃣ & 7️⃣ Fungsi Prediksi & Contoh (Tidak berubah)
# ================================
def predict_match(home_team_input, away_team_input, historical_df, model, scaler):
    # ... (Fungsi ini sama persis seperti sebelumnya)
    print("\n" + "="*40); print(f"🔮 PREDIKSI: {home_team_input} vs {away_team_input}"); print("="*40)
    name_map = {"Man Utd": "Man United", "Spurs": "Tottenham", "Chealsea": "Chelsea"}; home_team = name_map.get(home_team_input, home_team_input); away_team = name_map.get(away_team_input, away_team_input)
    home_form = rolling_stats[rolling_stats['Team'] == home_team].tail(1); away_form = rolling_stats[rolling_stats['Team'] == away_team].tail(1)
    if home_form.empty or away_form.empty: print(f"   ⚠️ Peringatan: Data form tidak ditemukan."); return
    form_values = [home_form['AvgGoalsFor_L5'].values[0], home_form['AvgGoalsAgainst_L5'].values[0], home_form['AvgShotsFor_L5'].values[0], home_form['AvgSOT_L5'].values[0], home_form['AvgCorners_L5'].values[0], home_form['AvgFouls_L5'].values[0], home_form['AvgYellows_L5'].values[0], away_form['AvgGoalsFor_L5'].values[0], away_form['AvgGoalsAgainst_L5'].values[0], away_form['AvgShotsFor_L5'].values[0], away_form['AvgSOT_L5'].values[0], away_form['AvgCorners_L5'].values[0], away_form['AvgFouls_L5'].values[0], away_form['AvgYellows_L5'].values[0]]
    h2h_stats = calculate_h2h_stats(home_team, away_team, historical_df); h2h_values = [h2h_stats['h2h_home_wins'], h2h_stats['h2h_away_wins'], h2h_stats['h2h_draws']]
    print(f"   - Rekor H2H: {home_team} Menang ({h2h_values[0]}), {away_team} Menang ({h2h_values[1]}), Seri ({h2h_values[2]})"); print(f"   - Form Gol (5 Laga): {home_team} ({form_values[0]:.2f}), {away_team} ({form_values[7]:.2f})"); print(f"   - Form Corners (5 Laga): {home_team} ({form_values[4]:.2f}), {away_team} ({form_values[11]:.2f})")
    input_features_raw = np.array([form_values + h2h_values]); input_features_scaled = scaler.transform(input_features_raw)
    probs = model.predict_proba(input_features_scaled)[0]; p_away, p_home, p_draw = probs[0], probs[1], probs[2]
    print("\n📈 Probabilitas Hasil:"); print(f"   - 🏠 {home_team_input} Menang: {p_home*100:.2f}%"); print(f"   - 🚗 {away_team_input} Menang: {p_away*100:.2f}%"); print(f"   - 🤝 Seri            : {p_draw*100:.2f}%")
    if p_home > max(p_draw, p_away): conclusion = f"{home_team_input} kemungkinan besar akan menang."; elif p_away > max(p_home, p_draw): conclusion = f"{away_team_input} kemungkinan besar akan menang."; else: conclusion = "Pertandingan ini kemungkinan besar akan berakhir seri."
    print(f"\n💡 Kesimpulan: {conclusion}")

predict_match("Arsenal", "Man Utd", hist_df, model, scaler)
predict_match("Liverpool", "Man City", hist_df, model, scaler)
predict_match("Chelsea", "Tottenham", hist_df, model, scaler)