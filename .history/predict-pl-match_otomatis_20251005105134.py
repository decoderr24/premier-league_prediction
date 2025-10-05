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
    print(f"✅ Berhasil memuat statistik: {passing_df.shape[0]} pemain")
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
# 2️⃣ Preprocessing & Persiapan Model
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
X = hist_df[["HomePass%", "AwayPass%"]]
y = hist_df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data siap.")

# ================================
# 3️⃣ Melatih Model Machine Learning
# ================================
print("\n🧠 Melatih model RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("✅ Model berhasil dilatih.")

# ================================
# 4️⃣ Mengevaluasi Kinerja Model
# ================================
print("\n⚖️ Mengevaluasi akurasi model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi Model Secara Keseluruhan: {acc*100:.2f}%")

# === BAGIAN YANG DITAMBAHKAN UNTUK DETAIL ===
print("\nLaporan Detail Kinerja Model:\n")
# Menampilkan laporan presisi, recall, dan f1-score untuk setiap kategori hasil
print(classification_report(y_test, y_pred, target_names=['Seri', 'Menang Tuan Rumah', 'Menang Tamu']))
# ============================================

# ================================
# 5️⃣ Fungsi untuk Prediksi Pertandingan
# ================================
def predict_match(home_team, away_team, passing_df, model):
    print(f"\n🔮 Prediksi Pertandingan: {home_team} vs {away_team}")

    def avg_pass_perc(team_name):
        team_df = passing_df[passing_df["Squad"].str.contains(team_name, case=False, na=False)]
        if team_df.empty:
            print(f"   ⚠️ Peringatan: Tidak ditemukan data untuk '{team_name}'. Menggunakan rata-rata liga.")
            return passing_df["Total_Cmp%"].astype(float).mean()
        return team_df["Total_Cmp%"].astype(float).mean()

    home_pass = avg_pass_perc(home_team)
    away_pass = avg_pass_perc(away_team)
    print(f"   - Rata-rata Pass {home_team}: {home_pass:.2f}%")
    print(f"   - Rata-rata Pass {away_team}: {away_pass:.2f}%")

    probs = model.predict_proba([[home_pass, away_pass]])[0]
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
predict_match("Arsenal", "Manchester Utd", passing_df, model)
predict_match("Liverpool", "Man City", passing_df, model)