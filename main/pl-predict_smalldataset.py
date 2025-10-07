import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ================================
# 1️⃣ Memuat Data
# ================================
print("📊 Memuat data statistik passing pemain...")
try:
    passing_df = pd.read_csv("premier_league_player_passing.csv")
except FileNotFoundError:
    print("❌ Gagal: File 'premier_league_player_passing.csv' tidak ditemukan.")
    exit()

print("\n📊 Memuat data pertandingan historis...")
try:
    hist_df = pd.read_csv("historical_matches.csv")
    print(f"✅ Berhasil memuat data historis: {hist_df.shape[0]} pertandingan")
except (FileNotFoundError, pd.errors.EmptyDataError):
    print("❌ Gagal: File 'historical_matches.csv' tidak ditemukan atau kosong.")
    print("   Pastikan Anda sudah menjalankan script 'create_historical_data.py' versi multi-season.")
    exit()


# ================================
# 2️⃣ PERBAIKAN: Feature Engineering H2H Tanpa Kebocoran Data
# ================================
print("\n🛠️  Membuat fitur H2H (tanpa kebocoran data)...")

# Pastikan kolom Tanggal berformat datetime dan urutkan
try:
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    hist_df = hist_df.sort_values(by='Date').reset_index(drop=True)
except Exception as e:
    print(f"❌ Error memproses kolom tanggal: {e}")
    print("   Pastikan format tanggal di CSV Anda benar (misal: YYYY-MM-DD).")
    exit()


# Fungsi untuk menghitung statistik H2H (tidak berubah)
def calculate_h2h_stats(home_team, away_team, past_matches_df):
    h2h_matches = past_matches_df[
        ((past_matches_df['Home'] == home_team) & (past_matches_df['Away'] == away_team)) |
        ((past_matches_df['Home'] == away_team) & (past_matches_df['Away'] == home_team))
    ]
    
    home_team_wins, away_team_wins, draws = 0, 0, 0
    
    if h2h_matches.empty:
        return {'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0}

    for _, row in h2h_matches.iterrows():
        if row['HomeGoals'] > row['AwayGoals']:
            if row['Home'] == home_team: home_team_wins += 1
            else: away_team_wins += 1
        elif row['AwayGoals'] > row['HomeGoals']:
            if row['Away'] == away_team: away_team_wins += 1
            else: home_team_wins += 1
        else:
            draws += 1
            
    return {'h2h_home_wins': home_team_wins, 'h2h_away_wins': away_team_wins, 'h2h_draws': draws}

# === INI BAGIAN PENTING PERBAIKANNYA ===
# Kita akan loop melalui setiap pertandingan dan hanya menghitung H2H dari laga SEBELUMNYA.
h2h_results = []
for index, row in hist_df.iterrows():
    # Ambil semua pertandingan yang terjadi SEBELUM tanggal pertandingan saat ini
    past_matches = hist_df.iloc[:index]
    # Hitung H2H hanya berdasarkan data masa lalu tersebut
    stats = calculate_h2h_stats(row['Home'], row['Away'], past_matches)
    h2h_results.append(stats)

# Gabungkan fitur-fitur baru ini ke dataframe utama
hist_df = pd.concat([hist_df, pd.DataFrame(h2h_results)], axis=1)
print("✅ Fitur H2H yang benar berhasil ditambahkan.")

# ================================
# 3️⃣ Preprocessing & Persiapan Model
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
print("✅ Data siap.")

# ================================
# 4️⃣ Melatih & Mengevaluasi Model
# ================================
print("\n🧠 Melatih model dengan fitur yang benar...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("✅ Model berhasil dilatih.")

print("\n⚖️ Mengevaluasi akurasi model baru yang realistis...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi Model Secara Keseluruhan: {acc*100:.2f}%")
print("\nLaporan Detail Kinerja Model:\n")
print(classification_report(y_test, y_pred, target_names=['Seri', 'Menang Tuan Rumah', 'Menang Tamu']))

# ================================
# 5️⃣ Fungsi Prediksi (Sudah Benar)
# ================================
def predict_match(home_team, away_team, passing_df, historical_df, model):
    print(f"\n🔮 Prediksi Pertandingan: {home_team} vs {away_team}")
    
    def avg_pass_perc(team_name, passing_stats_df):
        name_map = {"Man City": "Manchester City", "Man Utd": "Manchester United", "Spurs": "Tottenham Hotspur"}
        standard_name = name_map.get(team_name, team_name)
        team_df = passing_stats_df[passing_stats_df["Squad"].str.contains(standard_name, case=False, na=False)]
        if team_df.empty: return passing_stats_df["Total_Cmp%"].astype(float).mean()
        return team_df["Total_Cmp%"].astype(float).mean()

    home_pass = avg_pass_perc(home_team, passing_df)
    away_pass = avg_pass_perc(away_team, passing_df)
    print(f"   - Rata-rata Pass: {home_team} ({home_pass:.2f}%) vs {away_team} ({away_pass:.2f}%)")

    # Saat memprediksi laga baru, kita gunakan SEMUA data historis, ini sudah benar.
    h2h_stats = calculate_h2h_stats(home_team, away_team, historical_df)
    h2h_home_wins, h2h_away_wins, h2h_draws = h2h_stats['h2h_home_wins'], h2h_stats['h2h_away_wins'], h2h_stats['h2h_draws']
    print(f"   - Rekor H2H: {home_team} Menang ({h2h_home_wins}), {away_team} Menang ({h2h_away_wins}), Seri ({h2h_draws})")

    input_features = np.array([[home_pass, away_pass, h2h_home_wins, h2h_away_wins, h2h_draws]])
    probs = model.predict_proba(input_features)[0]
    p_draw, p_home, p_away = probs[0], probs[1], probs[2]

    print("\n📈 Probabilitas Hasil:")
    print(f"   - 🏠 {home_team} Menang: {p_home*100:.2f}%")
    print(f"   - 🚗 {away_team} Menang: {p_away*100:.2f}%")
    print(f"   - 🤝 Seri            : {p_draw*100:.2f}%")
    
    # ... (sisa kode sama)

# ================================
# 6️⃣ Jalankan Contoh Prediksi
# ================================
predict_match("Brentford", "Man City", passing_df, hist_df, model)
predict_match("Liverpool", "Everton", passing_df, hist_df, model)
predict_match("Arsenal", "Man Utd", passing_df, hist_df, model)
predict_match("Liverpool", "Man City", passing_df, hist_df, model)