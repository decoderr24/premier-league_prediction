import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report # Pastikan ini di-import
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ================================
# 1️⃣ Memuat Data
# ================================
print("📊 Memuat data statistik passing pemain (untuk prediksi)...")
try:
    passing_df = pd.read_csv("premier_league_player_passing.csv")
except FileNotFoundError:
    print("❌ Gagal: File 'premier_league_player_passing.csv' tidak ditemukan.")
    exit()

print("\n📊 Memuat data pertandingan historis dari Kaggle...")
try:
    hist_df = pd.read_csv("epl-training.csv")
    print(f"✅ Berhasil memuat data historis: {hist_df.shape[0]} pertandingan")
except FileNotFoundError:
    print("❌ Gagal: File 'epl-training.csv' tidak ditemukan.")
    exit()

# ================================
# 2️⃣ Preprocessing & Penyesuaian Data
# ================================
print("\n✨ Menyesuaikan data dari Kaggle...")
try:
    hist_df.rename(columns={'HomeTeam': 'Home', 'AwayTeam': 'Away', 'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'}, inplace=True)
    hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='%d/%m/%Y')
    hist_df = hist_df.sort_values(by='Date').reset_index(drop=True)
    print("✅ Nama kolom dan format tanggal berhasil disesuaikan.")
    
    pass_map = passing_df.groupby('Squad')['Total_Cmp%'].mean()
    hist_df['HomePass%'] = hist_df['Home'].map(pass_map)
    hist_df['AwayPass%'] = hist_df['Away'].map(pass_map)
    average_pass_perc = passing_df['Total_Cmp%'].mean()
    hist_df['HomePass%'].fillna(average_pass_perc, inplace=True)
    hist_df['AwayPass%'].fillna(average_pass_perc, inplace=True)
    print("✅ Fitur passing berhasil dibuat untuk data historis.")
except KeyError:
    print("❌ Error: Kolom yang dibutuhkan tidak ditemukan di 'epl-training.csv'.")
    exit()

# ================================
# 3️⃣ Feature Engineering H2H
# ================================
print("\n🛠️  Membuat fitur Head-to-Head (H2H)...")
def calculate_h2h_stats(home_team, away_team, past_matches_df):
    h2h_matches = past_matches_df[
        ((past_matches_df['Home'] == home_team) & (past_matches_df['Away'] == away_team)) |
        ((past_matches_df['Home'] == away_team) & (past_matches_df['Away'] == home_team))
    ]
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
    past_matches = hist_df.iloc[:index]
    stats = calculate_h2h_stats(row['Home'], row['Away'], past_matches)
    h2h_results.append(stats)
hist_df = pd.concat([hist_df, pd.DataFrame(h2h_results)], axis=1)
print("✅ Fitur H2H berhasil ditambahkan.")

# ================================
# 4️⃣ Training & Evaluasi Model
# ================================
print("\n⚙️ Mempersiapkan data untuk training model...")
def get_result(row):
    if row["HomeGoals"] > row["AwayGoals"]: return 1
    elif row["HomeGoals"] < row["AwayGoals"]: return 2
    else: return 0
hist_df["Result"] = hist_df.apply(get_result, axis=1)
features = ["HomePass%", "AwayPass%", "h2h_home_wins", "h2h_away_wins", "h2h_draws"]
X = hist_df[features]
y = hist_df["Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🧠 Melatih model RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("✅ Model berhasil dilatih.")

print("\n⚖️ Mengevaluasi akurasi model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi Model Secara Keseluruhan: {acc*100:.2f}%")

# === BAGIAN INI DIAKTIFKAN UNTUK CEK AKURASI DETAIL ===
print("\nLaporan Detail Kinerja Model:\n")
print(classification_report(y_test, y_pred, target_names=['Seri', 'Menang Tuan Rumah', 'Menang Tamu']))
# =======================================================


# ================================
# 5️⃣ & 6️⃣ Fungsi Prediksi & Contoh
# ================================
def predict_match(home_team, away_team, passing_df, historical_df, model):
    print("\n" + "="*40)
    print(f"🔮 PREDIKSI: {home_team} vs {away_team}")
    print("="*40)
    # ... (sisa fungsi ini tidak berubah)
    def avg_pass_perc(team_name, passing_stats_df):
        name_map = {"Man City": "Manchester City", "Man Utd": "Man United", "Spurs": "Tottenham"}
        standard_name = name_map.get(team_name, team_name)
        team_df = passing_stats_df[passing_stats_df["Squad"].str.contains(standard_name, case=False, na=False)]
        if team_df.empty: return passing_stats_df["Total_Cmp%"].astype(float).mean()
        return team_df["Total_Cmp%"].astype(float).mean()
    home_pass = avg_pass_perc(home_team, passing_df)
    away_pass = avg_pass_perc(away_team, passing_df)
    h2h_stats = calculate_h2h_stats(home_team, away_team, historical_df)
    h2h_home, h2h_away, h2h_draw = h2h_stats['h2h_home_wins'], h2h_stats['h2h_away_wins'], h2h_stats['h2h_draws']
    print(f"   - Rekor H2H dari dataset: {home_team} Menang ({h2h_home}), {away_team} Menang ({h2h_away}), Seri ({h2h_draw})")
    input_features = np.array([[home_pass, away_pass, h2h_home, h2h_away, h2h_draw]])
    probs = model.predict_proba(input_features)[0]
    p_draw, p_home, p_away = probs[0], probs[1], probs[2]
    print("\n📈 Probabilitas Hasil:")
    print(f"   - 🏠 {home_team} Menang: {p_home*100:.2f}%")
    print(f"   - 🚗 {away_team} Menang: {p_away*100:.2f}%")
    print(f"   - 🤝 Seri            : {p_draw*100:.2f}%")
    if p_home > max(p_draw, p_away): conclusion = f"{home_team} kemungkinan besar akan menang."
    elif p_away > max(p_home, p_draw): conclusion = f"{away_team} kemungkinan besar akan menang."
    else: conclusion = "Pertandingan ini kemungkinan besar akan berakhir seri."
    print(f"\n💡 Kesimpulan: {conclusion}")

predict_match("Arsenal", "Man Utd", passing_df, hist_df, model)
predict_match("Liverpool", "Man City", passing_df, hist_df, model)
predict_match("Chelsea", "Tottenham", passing_df, hist_df, model)