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
print("📊 Memuat data statistik passing pemain (untuk prediksi)...")
try:
    passing_df = pd.read_csv("premier_league_player_passing.csv")
except FileNotFoundError:
    print("❌ Gagal: File 'premier_league_player_passing.csv' tidak ditemukan.")
    exit()

print("\n📊 Memuat data pertandingan historis dari Kaggle...")
try:
    # --- PERUBAHAN 1: Membaca file epl-training.csv ---
    hist_df = pd.read_csv("epl-training.csv")
    print(f"✅ Berhasil memuat data historis: {hist_df.shape[0]} pertandingan")
except FileNotFoundError:
    print("❌ Gagal: File 'epl-training.csv' tidak ditemukan.")
    exit()

# ================================
# 2️⃣ Preprocessing & Penyesuaian Data
# ================================
print("\n✨ Menyesuaikan data dari Kaggle...")

# --- PERUBAHAN 2: Mengubah nama kolom agar sesuai ---
# FTHG = Full Time Home Goals, FTAG = Full Time Away Goals
try:
    hist_df.rename(columns={
        'HomeTeam': 'Home',
        'AwayTeam': 'Away',
        'FTHG': 'HomeGoals',
        'FTAG': 'AwayGoals'
    }, inplace=True)

    # --- PERUBAHAN 3: Menyesuaikan format tanggal ---
    # Mengubah format DD/MM/YYYY menjadi format standar yang bisa diurutkan
    hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='%d/%m/%Y')
    hist_df = hist_df.sort_values(by='Date').reset_index(drop=True)
    print("✅ Nama kolom dan format tanggal berhasil disesuaikan.")
    
    # === CATATAN PENTING TENTANG FITUR PASSING ===
    # Dataset historis ini sangat besar, namun data passing (passing_df) kemungkinan
    # hanya dari musim terbaru. Jadi kita akan membuat fitur passing berdasarkan data terbaru ini.
    # Untuk model yang lebih canggih, kita bisa menggunakan fitur lain dari dataset Kaggle
    # seperti Shots (HS, AS) atau Corners (HC, AC).
    
    # Membuat fitur HomePass% dan AwayPass% untuk data historis
    # Ini adalah sebuah penyederhanaan, kita asumsikan %passing tim relatif stabil
    pass_map = passing_df.groupby('Squad')['Total_Cmp%'].mean()
    hist_df['HomePass%'] = hist_df['Home'].map(pass_map)
    hist_df['AwayPass%'] = hist_df['Away'].map(pass_map)
    # Isi data tim yang tidak ada di data passing (misal, tim lama) dengan rata-rata
    average_pass_perc = passing_df['Total_Cmp%'].mean()
    hist_df['HomePass%'].fillna(average_pass_perc, inplace=True)
    hist_df['AwayPass%'].fillna(average_pass_perc, inplace=True)
    print("✅ Fitur passing berhasil dibuat untuk data historis.")

except KeyError:
    print("❌ Error: Kolom yang dibutuhkan (misal: 'HomeTeam', 'FTHG') tidak ditemukan di 'epl-training.csv'.")
    exit()


# ================================
# 3️⃣ Feature Engineering H2H (Logika Tetap Sama)
# ================================
print("\n🛠️  Membuat fitur Head-to-Head (H2H)...")
def calculate_h2h_stats(home_team, away_team, past_matches_df):
    h2h_matches = past_matches_df[
        ((past_matches_df['Home'] == home_team) & (past_matches_df['Away'] == away_team)) |
        ((past_matches_df['Home'] == away_team) & (past_matches_df['Away'] == home_team))
    ]
    home_team_wins, away_team_wins, draws = 0, 0, 0
    if h2h_matches.empty: return {'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0}
    for _, row in h2h_matches.iterrows():
        if row['HomeGoals'] > row['AwayGoals']:
            if row['Home'] == home_team: home_team_wins += 1
            else: away_team_wins += 1
        elif row['AwayGoals'] > row['HomeGoals']:
            if row['Away'] == away_team: away_team_wins += 1
            else: home_team_wins += 1
        else: draws += 1
    return {'h2h_home_wins': home_team_wins, 'h2h_away_wins': away_team_wins, 'h2h_draws': draws}

h2h_results = []
for index, row in hist_df.iterrows():
    past_matches = hist_df.iloc[:index]
    stats = calculate_h2h_stats(row['Home'], row['Away'], past_matches)
    h2h_results.append(stats)
hist_df = pd.concat([hist_df, pd.DataFrame(h2h_results)], axis=1)
print("✅ Fitur H2H berhasil ditambahkan.")


# ================================
# Sisa Script (Training, Evaluasi, Prediksi Massal) tidak berubah
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

print("\n🧠 Melatih model RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("✅ Model berhasil dilatih.")

print("\n⚖️ Mengevaluasi akurasi model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi Model Secara Keseluruhan: {acc*100:.2f}%")


print("\n" + "="*40)
print("🔮 MEMULAI PREDIKSI MASSAL DARI FILE JADWAL 🔮")
print("="*40)

def predict_match(home_team, away_team, passing_df, historical_df, model):
    # ... (Fungsi predict_match sama persis seperti sebelumnya)
    def avg_pass_perc(team_name, passing_stats_df):
        name_map = {"Man City": "Manchester City", "Man Utd": "Manchester United", "Spurs": "Tottenham Hotspur", "Nott'ham Forest": "Nottingham Forest", "Wolves": "Wolverhampton Wanderers", "West Ham": "West Ham United", "AFC Bournemouth": "Bournemouth"}
        standard_name = name_map.get(team_name, team_name)
        team_df = passing_stats_df[passing_stats_df["Squad"].str.contains(standard_name, case=False, na=False)]
        if team_df.empty: return passing_stats_df["Total_Cmp%"].astype(float).mean()
        return team_df["Total_Cmp%"].astype(float).mean()

    home_pass = avg_pass_perc(home_team, passing_df)
    away_pass = avg_pass_perc(away_team, passing_df)
    h2h_stats = calculate_h2h_stats(home_team, away_team, historical_df)
    h2h_home_wins, h2h_away_wins, h2h_draws = h2h_stats['h2h_home_wins'], h2h_stats['h2h_away_wins'], h2h_stats['h2h_draws']
    input_features = np.array([[home_pass, away_pass, h2h_home_wins, h2h_away_wins, h2h_draws]])
    probs = model.predict_proba(input_features)[0]
    p_draw, p_home, p_away = probs[0], probs[1], probs[2]
    if p_home > max(p_draw, p_away): conclusion = f"{home_team} Menang"
    elif p_away > max(p_home, p_draw): conclusion = f"{away_team} Menang"
    else: conclusion = "Seri"
    return {"Tim Tuan Rumah": home_team, "Tim Tamu": away_team, "Peluang Menang Tuan Rumah (%)": round(p_home * 100, 2), "Peluang Seri (%)": round(p_draw * 100, 2), "Peluang Menang Tamu (%)": round(p_away * 100, 2), "Kesimpulan": conclusion}

try:
    schedule_df = pd.read_csv("epl-test.csv")
    print(f"✅ Berhasil memuat {len(schedule_df)} pertandingan dari 'epl-test.csv'")
except FileNotFoundError:
    print("❌ Gagal: File 'epl-test.csv' tidak ditemukan. Prediksi massal dibatalkan.")
    schedule_df = pd.DataFrame()

predictions = []
if not schedule_df.empty:
    for index, row in schedule_df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        print(f"   - Memprediksi: {home} vs {away}")
        prediction_result = predict_match(home, away, passing_df, hist_df, model)
        predictions.append(prediction_result)
    predictions_df = pd.DataFrame(predictions)
    print("\n\n✅ PREDIKSI SELESAI. HASIL LENGKAP:\n")
    print(predictions_df.to_string())
    try:
        output_filename = "hasil_prediksi_epl.csv"
        predictions_df.to_csv(output_filename, index=False)
        print(f"\n\n💾 Hasil prediksi telah disimpan ke file: '{output_filename}'")
    except Exception as e:
        print(f"\n\n❌ Gagal menyimpan hasil prediksi ke file CSV: {e}")