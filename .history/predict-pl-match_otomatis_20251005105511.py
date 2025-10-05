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
except FileNotFoundError:
    print("❌ Gagal: File 'historical_matches.csv' tidak ditemukan.")
    exit()

# ================================
# 2️⃣ Feature Engineering: Menambahkan Data Head-to-Head (H2H)
# ================================
print("\n🛠️  Membuat fitur baru dari data Head-to-Head (H2H)...")

# Fungsi untuk menghitung statistik H2H antara dua tim
def calculate_h2h_stats(home_team, away_team, all_matches_df):
    # Filter semua pertandingan antara kedua tim
    h2h_matches = all_matches_df[
        ((all_matches_df['Home'] == home_team) & (all_matches_df['Away'] == away_team)) |
        ((all_matches_df['Home'] == away_team) & (all_matches_df['Away'] == home_team))
    ]
    
    home_team_wins = 0
    away_team_wins = 0
    draws = 0
    
    if h2h_matches.empty:
        return {'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0}

    for index, row in h2h_matches.iterrows():
        if row['HomeGoals'] > row['AwayGoals']:
            if row['Home'] == home_team:
                home_team_wins += 1
            else:
                away_team_wins += 1
        elif row['AwayGoals'] > row['HomeGoals']:
            if row['Away'] == away_team:
                away_team_wins += 1
            else:
                home_team_wins += 1
        else:
            draws += 1
            
    return {'h2h_home_wins': home_team_wins, 'h2h_away_wins': away_team_wins, 'h2h_draws': draws}

# Terapkan fungsi ini ke setiap baris di dataframe historis untuk membuat fitur baru
h2h_features = hist_df.apply(
    lambda row: calculate_h2h_stats(row['Home'], row['Away'], hist_df),
    axis=1
)

# Gabungkan fitur-fitur baru ini ke dataframe utama
hist_df = pd.concat([hist_df, h2h_features.apply(pd.Series)], axis=1)
print("✅ Fitur H2H berhasil ditambahkan.")

# ================================
# 3️⃣ Preprocessing & Persiapan Model
# ================================
print("\n⚙️ Mempersiapkan data untuk training model...")

def get_result(row):
    if row["HomeGoals"] > row["AwayGoals"]:
        return 1
    elif row["HomeGoals"] < row["AwayGoals"]:
        return 2
    else:
        return 0

hist_df["Result"] = hist_df.apply(get_result, axis=1)

# === PERBARUI FITUR YANG DIGUNAKAN MODEL ===
# Sekarang kita gunakan 5 fitur, bukan hanya 2
features = ["HomePass%", "AwayPass%", "h2h_home_wins", "h2h_away_wins", "h2h_draws"]
X = hist_df[features]
y = hist_df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data siap.")

# ================================
# 4️⃣ Melatih & Mengevaluasi Model
# ================================
print("\n🧠 Melatih model dengan fitur H2H...")
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
# 5️⃣ Fungsi Prediksi (Diperbarui dengan H2H)
# ================================
def predict_match(home_team, away_team, passing_df, historical_df, model):
    print(f"\n🔮 Prediksi Pertandingan: {home_team} vs {away_team}")

    # --- Bagian 1: Dapatkan statistik passing (sama seperti sebelumnya) ---
    def avg_pass_perc(team_name, passing_stats_df):
        name_map = {"Man City": "Manchester City", "Man Utd": "Manchester United", "Spurs": "Tottenham Hotspur"}
        standard_name = name_map.get(team_name, team_name)
        team_df = passing_stats_df[passing_stats_df["Squad"].str.contains(standard_name, case=False, na=False)]
        if team_df.empty: return passing_stats_df["Total_Cmp%"].astype(float).mean()
        return team_df["Total_Cmp%"].astype(float).mean()

    home_pass = avg_pass_perc(home_team, passing_df)
    away_pass = avg_pass_perc(away_team, passing_df)
    print(f"   - Rata-rata Pass: {home_team} ({home_pass:.2f}%) vs {away_team} ({away_pass:.2f}%)")

    # --- Bagian 2: Dapatkan statistik H2H ---
    h2h_stats = calculate_h2h_stats(home_team, away_team, historical_df)
    h2h_home_wins = h2h_stats['h2h_home_wins']
    h2h_away_wins = h2h_stats['h2h_away_wins']
    h2h_draws = h2h_stats['h2h_draws']
    print(f"   - Rekor H2H: {home_team} Menang ({h2h_home_wins}), {away_team} Menang ({h2h_away_wins}), Seri ({h2h_draws})")

    # --- Bagian 3: Buat prediksi menggunakan SEMUA fitur ---
    input_features = np.array([[home_pass, away_pass, h2h_home_wins, h2h_away_wins, h2h_draws]])
    probs = model.predict_proba(input_features)[0]
    p_draw, p_home, p_away = probs[0], probs[1], probs[2]

    print("\n📈 Probabilitas Hasil:")
    print(f"   - 🏠 {home_team} Menang: {p_home*100:.2f}%")
    print(f"   - 🚗 {away_team} Menang: {p_away*100:.2f}%")
    print(f"   - 🤝 Seri            : {p_draw*100:.2f}%")
    
    if p_home > max(p_draw, p_away):
        result = f"{home_team} kemungkinan besar akan menang."
    elif p_away > max(p_home, p_draw):
        result = f"{away_team} kemungkinan besar akan menang."
    else:
        result = "Pertandingan ini kemungkinan besar akan berakhir seri."

    print(f"\n💡 Kesimpulan: {result}")
    print("-" * 40)

# ================================
# 6️⃣ Jalankan Contoh Prediksi
# ================================
predict_match("Brentford", "Man City", passing_df, hist_df, model)
predict_match("Liverpool", "Everton", passing_df, hist_df, model) # Merseyside Derby
predict_match("Arsenal", "Man Utd", passing_df, hist_df, model)