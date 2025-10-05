import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ================================
# 1️⃣ & 2️⃣ (Memuat & Menyesuaikan Data)
# ================================
print("📊 Memuat data pertandingan historis dari Kaggle...")
try:
    hist_df = pd.read_csv("epl-training.csv")
    print(f"✅ Berhasil memuat data historis: {hist_df.shape[0]} pertandingan")
except FileNotFoundError:
    print("❌ Gagal: File 'epl-training.csv' tidak ditemukan.")
    exit()

print("\n✨ Menyesuaikan data dari Kaggle...")
try:
    relevant_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']
    hist_df = hist_df[relevant_cols]
    hist_df.rename(columns={'HomeTeam': 'Home', 'AwayTeam': 'Away', 'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals', 'HS': 'HomeShots', 'AS': 'AwayShots', 'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget'}, inplace=True)
    hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='%d/%m/%Y')
    hist_df = hist_df.sort_values(by='Date').reset_index(drop=True)
    print("✅ Nama kolom dan format tanggal berhasil disesuaikan.")
except KeyError:
    print("❌ Error: Kolom yang dibutuhkan tidak ditemukan di 'epl-training.csv'.")
    exit()

# ================================
# 3️⃣ FEATURE ENGINEERING TINGKAT LANJUT: "TEAM FORM"
# ================================
print("\n🚀 Membuat fitur 'Team Form' (Performa Terkini)...")

# Mengubah dataframe menjadi format yang lebih mudah untuk dihitung per tim
team_stats = []
for index, row in hist_df.iterrows():
    team_stats.append({'Date': row['Date'], 'Team': row['Home'], 'GoalsFor': row['HomeGoals'], 'GoalsAgainst': row['AwayGoals'], 'ShotsFor': row['HomeShots'], 'ShotsAgainst': row['AwayShots'], 'SOT_For': row['HomeShotsOnTarget'], 'SOT_Against': row['AwayShotsOnTarget']})
    team_stats.append({'Date': row['Date'], 'Team': row['Away'], 'GoalsFor': row['AwayGoals'], 'GoalsAgainst': row['HomeGoals'], 'ShotsFor': row['AwayShots'], 'ShotsAgainst': row['HomeShots'], 'SOT_For': row['AwayShotsOnTarget'], 'SOT_Against': row['HomeShotsOnTarget']})
team_stats_df = pd.DataFrame(team_stats).sort_values(by=['Team', 'Date'])

# Menghitung rata-rata performa dari 5 pertandingan terakhir (rolling average)
rolling_stats = team_stats_df.groupby('Team').rolling(window=5, on='Date').mean().reset_index()
rolling_stats.rename(columns={'GoalsFor': 'AvgGoalsFor_L5', 'GoalsAgainst': 'AvgGoalsAgainst_L5', 'ShotsFor': 'AvgShotsFor_L5', 'SOT_For': 'AvgSOT_For_L5'}, inplace=True)
rolling_stats = rolling_stats[['Team', 'Date', 'AvgGoalsFor_L5', 'AvgGoalsAgainst_L5', 'AvgShotsFor_L5', 'AvgSOT_For_L5']]

# Menggabungkan statistik form ini kembali ke dataframe utama
hist_df = pd.merge(hist_df, rolling_stats, left_on=['Date', 'Home'], right_on=['Date', 'Team'], how='left').rename(columns=lambda x: x.replace('_L5', '_Home_L5')).drop('Team', axis=1)
hist_df = pd.merge(hist_df, rolling_stats, left_on=['Date', 'Away'], right_on=['Date', 'Team'], how='left').rename(columns=lambda x: x.replace('_L5', '_Away_L5')).drop('Team', axis=1)
hist_df.dropna(inplace=True) # Hapus baris di awal yang belum punya data 5 laga
print("✅ Fitur 'Team Form' berhasil dibuat.")


# ================================
# 4️⃣ Feature Engineering H2H (Tetap digunakan)
# ================================
print("\n🛠️  Membuat fitur Head-to-Head (H2H)...")
def calculate_h2h_stats(home_team, away_team, past_matches_df):
    h2h_matches = past_matches_df[((past_matches_df['Home'] == home_team) & (past_matches_df['Away'] == away_team)) | ((past_matches_df['Home'] == away_team) & (past_matches_df['Away'] == home_team))]
    home_wins, away_wins, draws = 0, 0, 0
    if h2h_matches.empty: return {'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0}
    for _, row in h2h_matches.iterrows():
        if row['HomeGoals'] > row['AwayGoals']:
            if row['Home'] == home_team: home_wins += 1
            else: away_wins += 1
        elif row['AwayGoals'] > row['HomeGoals']:
            if row['Away'] == away_team: away_wins += 1
            else: home_wins += 1
        else: draws += 1
    return {'h2h_home_wins': home_wins, 'h2h_away_wins': away_wins, 'h2h_draws': draws}
h2h_results = []
for index, row in hist_df.iterrows():
    past_matches = hist_df.loc[hist_df.index < index]
    stats = calculate_h2h_stats(row['Home'], row['Away'], past_matches)
    h2h_results.append(stats)
hist_df = pd.concat([hist_df, pd.DataFrame(h2h_results, index=hist_df.index)], axis=1)
print("✅ Fitur H2H berhasil ditambahkan.")


# ================================
# 5️⃣ Training & Evaluasi Model dengan FITUR PRO
# ================================
print("\n⚙️ Mempersiapkan data untuk training model...")
def get_result(row):
    if row["HomeGoals"] > row["AwayGoals"]: return 1
    elif row["HomeGoals"] < row["AwayGoals"]: return 2
    else: return 0
hist_df["Result"] = hist_df.apply(get_result, axis=1)

# === FITUR PALING LENGKAP & CANGGIH ===
features = [
    'AvgGoalsFor_Home_L5', 'AvgGoalsAgainst_Home_L5', 'AvgShotsFor_Home_L5', 'AvgSOT_For_Home_L5',
    'AvgGoalsFor_Away_L5', 'AvgGoalsAgainst_Away_L5', 'AvgShotsFor_Away_L5', 'AvgSOT_For_Away_L5',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws'
]
X = hist_df[features]
y = hist_df["Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🧠 Melatih model PRO RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("✅ Model berhasil dilatih.")

print("\n⚖️ Mengevaluasi akurasi model PRO...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi Model Secara Keseluruhan: {acc*100:.2f}%")
print("\nLaporan Detail Kinerja Model:\n")
print(classification_report(y_test, y_pred, target_names=['Seri', 'Menang Tuan Rumah', 'Menang Tamu']))


# ================================
# 6️⃣ Fungsi Prediksi yang Menggunakan "Team Form"
# ================================
def predict_match(home_team_input, away_team_input, historical_df, model):
    print("\n" + "="*40)
    print(f"🔮 PREDIKSI: {home_team_input} vs {away_team_input}")
    print("="*40)

    # Memperbaiki typo "Chealsea" menjadi "Chelsea"
    name_map = {"Man Utd": "Man United", "Spurs": "Tottenham", "Chealsea": "Chelsea"}
    home_team = name_map.get(home_team_input, home_team_input)
    away_team = name_map.get(away_team_input, away_team_input)
    
    # Mendapatkan data form dari 5 laga terakhir
    home_form = rolling_stats[rolling_stats['Team'] == home_team].tail(1)
    away_form = rolling_stats[rolling_stats['Team'] == away_team].tail(1)

    if home_form.empty or away_form.empty:
        print(f"   ⚠️ Peringatan: Data form tidak ditemukan untuk salah satu tim. Prediksi mungkin tidak akurat.")
        return

    # Mengambil nilai rata-rata dari data form
    form_values = [
        home_form['AvgGoalsFor_L5'].values[0], home_form['AvgGoalsAgainst_L5'].values[0],
        home_form['AvgShotsFor_L5'].values[0], home_form['AvgSOT_For_L5'].values[0],
        away_form['AvgGoalsFor_L5'].values[0], away_form['AvgGoalsAgainst_L5'].values[0],
        away_form['AvgShotsFor_L5'].values[0], away_form['AvgSOT_For_L5'].values[0]
    ]
    
    h2h_stats = calculate_h2h_stats(home_team, away_team, historical_df)
    h2h_values = [h2h_stats['h2h_home_wins'], h2h_stats['h2h_away_wins'], h2h_stats['h2h_draws']]
    print(f"   - Rekor H2H: {home_team} Menang ({h2h_values[0]}), {away_team} Menang ({h2h_values[1]}), Seri ({h2h_values[2]})")
    print(f"   - Form Gol (5 Laga): {home_team} ({form_values[0]:.2f}), {away_team} ({form_values[4]:.2f})")

    input_features = np.array([form_values + h2h_values])
    probs = model.predict_proba(input_features)[0]
    p_draw, p_home, p_away = probs[0], probs[1], probs[2]

    print("\n📈 Probabilitas Hasil:")
    print(f"   - 🏠 {home_team_input} Menang: {p_home*100:.2f}%")
    print(f"   - 🚗 {away_team_input} Menang: {p_away*100:.2f}%")
    print(f"   - 🤝 Seri            : {p_draw*100:.2f}%")
    
    if p_home > max(p_draw, p_away): conclusion = f"{home_team_input} kemungkinan besar akan menang."
    elif p_away > max(p_home, p_draw): conclusion = f"{away_team_input} kemungkinan besar akan menang."
    else: conclusion = "Pertandingan ini kemungkinan besar akan berakhir seri."
    print(f"\n💡 Kesimpulan: {conclusion}")

# ================================
# 7️⃣ Jalankan Contoh Prediksi
# ================================
predict_match("Arsenal", "Man Utd", hist_df, model)
predict_match("Liverpool", "Man City", hist_df, model)
predict_match("Chelsea", "Tottenham", hist_df, model)
predict_match("Man City", "Liverpool", hist_df, model)
predict_match("Man Utd", "Arsenal", hist_df, model)
predict_match("Tottenham", "Chealsea", hist_df, model) # Uji perbaikan typo