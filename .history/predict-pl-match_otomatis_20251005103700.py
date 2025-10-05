import pandas as pd
import numpy as np

# ===============================
# 1️⃣ Load Data Passing
# ===============================
def load_team_stats(csv_path="premier_league_player_passing.csv"):
    df = pd.read_csv(csv_path)
    # Ambil rata-rata passing per tim
    team_stats = df.groupby("Squad").agg(
        Total_Cmp_percent=pd.NamedAgg(column="Total_Cmp%", aggfunc="mean"),
        Total_Cmp=pd.NamedAgg(column="Total_Cmp", aggfunc="sum"),
        Total_Att=pd.NamedAgg(column="Total_Att", aggfunc="sum")
    ).reset_index()
    return team_stats

# ===============================
# 2️⃣ Hitung skor prediksi
# ===============================
def predict_match(home_team, away_team, team_stats):
    print(f"\n🔮 Memprediksi pertandingan: {home_team} vs {away_team}")

    # Ambil data passing
    home = team_stats[team_stats["Squad"]==home_team].squeeze()
    away = team_stats[team_stats["Squad"]==away_team].squeeze()

    if home.empty or away.empty:
        print("⚠️ Salah satu tim tidak ditemukan di data, pakai default")
        home_pass = 0.82
        away_pass = 0.78
    else:
        home_pass = home["Total_Cmp_percent"] / 100
        away_pass = away["Total_Cmp_percent"] / 100

    # Asumsi form & home advantage
    home_adv = 1.1
    away_adv = 1.0
    home_form = home_pass * 0.6 + 0.4
    away_form = away_pass * 0.6 + 0.4

    # Skor probabilitas
    home_score = home_form * home_adv
    away_score = away_form * away_adv

    p_home = home_score / (home_score + away_score)
    p_away = away_score / (home_score + away_score)
    p_draw = 0.15

    print("\n📈 Probabilitas Prediksi:")
    print(f"  🏠 {home_team} menang : {p_home:.2%}")
    print(f"  🚗 {away_team} menang : {p_away:.2%}")
    print(f"  🤝 Seri               : {p_draw:.2%}")

    # Prediksi skor (menggunakan proporsi probabilitas + random)
    exp_home_goals = max(round(p_home * 3 + np.random.uniform(0, 1)), 0)
    exp_away_goals = max(round(p_away * 3 + np.random.uniform(0, 1)), 0)

    print(f"\n⚽ Prediksi Skor: {home_team} {exp_home_goals} – {exp_away_goals} {away_team}")

    # Analisis akhir
    if p_home > p_away:
        result = f"{home_team} berpeluang menang"
    elif p_away > p_home:
        result = f"{away_team} berpeluang menang"
    else:
        result = "kemungkinan besar imbang"
    print(f"\n🧠 Analisis Akhir: {result}\n")
    print("✅ Prediksi selesai.")


# ===============================
# 3️⃣ Main
# ===============================
def main():
    team_stats = load_team_stats()
    print(f"📊 Loaded team stats: {team_stats.shape[0]} teams")
    
    # Ubah tim sesuai yang ingin diprediksi
    home_team = "Aston Villa"
    away_team = "Man City"
    predict_match(home_team, away_team, team_stats)

if __name__ == "__main__":
    main()
