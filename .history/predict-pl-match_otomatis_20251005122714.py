import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ================================
# 1️⃣ Memuat Data Historis dari Kaggle
# ================================
print("📊 Memuat data pertandingan historis dari Kaggle...")
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
    # Memilih hanya kolom yang akan kita gunakan
    relevant_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']
    hist_df = hist_df[relevant_cols]
    
    hist_df.rename(columns={
        'HomeTeam': 'Home', 'AwayTeam': 'Away',
        'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals',
        'HS': 'HomeShots', 'AS': 'AwayShots',
        'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget'
    }, inplace=True)
    
    hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='%d/%m/%Y')
    hist_df = hist_df.sort_values(by='Date').reset_index(drop=True)
    print("✅ Nama kolom dan format tanggal berhasil disesuaikan.")

except KeyError:
    print("❌ Error: Kolom yang dibutuhkan (misal: 'HomeTeam', 'HS') tidak ditemukan di 'epl-training.csv'.")
    exit()

# ================================
# 3️⃣ Feature Engineering H2H (Tetap kita gunakan karena penting)
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
    past_matches = hist_df.iloc[:index]
    stats = calculate_h2h_stats(row['Home'], row['Away'], past_matches)
    h2h_results.append(stats)
hist_df = pd.concat([hist_df, pd.DataFrame(h2h_results)], axis=1)
print("✅ Fitur H2H berhasil ditambahkan.")

# ================================
# 4️⃣ Training & Evaluasi Model dengan FITUR BARU
# ================================
print("\n⚙️ Mempersiapkan data untuk training model...")
def get_result(row):
    if row["HomeGoals"] > row["AwayGoals"]: return 1
    elif row["HomeGoals"] < row["AwayGoals"]: return 2
    else: return 0
hist_df["Result"] = hist_df.apply(get_result, axis=1)

# === FITUR BARU & LEBIH BAIK ===
features = [
    'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws'
]
X = hist_df[features]
y = hist_df["Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🧠 Melatih model RandomForestClassifier dengan fitur baru...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("✅ Model berhasil dilatih.")

print("\n⚖️ Mengevaluasi akurasi model baru...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi Model Secara Keseluruhan: {acc*100:.2f}%")
print("\nLaporan Detail Kinerja Model:\n")
print(classification_report(y_test, y_pred, target_names=['Seri', 'Menang Tuan Rumah', 'Menang Tamu']))

# ================================
# 5️⃣ Fungsi Prediksi yang Lebih Cerdas
# ================================
def predict_match(home_team_input, away_team_input, historical_df, model):
    print("\n" + "="*40)
    print(f"🔮 PREDIKSI: {home_team_input} vs {away_team_input}")
    print("="*40)

    name_map_kaggle = {"Man Utd": "Man United", "Spurs": "Tottenham"}
    home_team = name_map_kaggle.get(home_team_input, home_team_input)
    away_team = name_map_kaggle.get(away_team_input, away_team_input)
    
    # Hitung rata-rata statistik historis untuk kedua tim
    home_stats = historical_df[historical_df['Home'] == home_team]
    away_stats = historical_df[historical_df['Away'] == away_team]
    
    avg_home_shots = home_stats['HomeShots'].mean()
    avg_away_shots = away_stats['AwayShots'].mean()
    avg_home_sot = home_stats['HomeShotsOnTarget'].mean()
    avg_away_sot = away_stats['AwayShotsOnTarget'].mean()

    print(f"   - Rata-rata Shots (Kandang): {home_team} ({avg_home_shots:.2f}), (Tandang): {away_team} ({avg_away_shots:.2f})")
    print(f"   - Rata-rata SOT (Kandang): {home_team} ({avg_home_sot:.2f}), (Tandang): {away_team} ({avg_away_sot:.2f})")

    h2h_stats = calculate_h2h_stats(home_team, away_team, historical_df)
    h2h_home, h2h_away, h2h_draw = h2h_stats['h2h_home_wins'], h2h_stats['h2h_away_wins'], h2h_stats['h2h_draws']
    print(f"   - Rekor H2H dari dataset: {home_team} Menang ({h2h_home}), {away_team} Menang ({h2h_away}), Seri ({h2h_draw})")

    # Gunakan rata-rata stats sebagai input untuk prediksi
    input_features = np.array([[
        avg_home_shots, avg_away_shots, avg_home_sot, avg_away_sot,
        h2h_home, h2h_away, h2h_draw
    ]])
    
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
# 6️⃣ Jalankan Contoh Prediksi
# ================================
predict_match("Arsenal", "Man Utd", hist_df, model)
predict_match("Liverpool", "Man City", hist_df, model)
predict_match("Chelsea", "Tottenham", hist_df, model)