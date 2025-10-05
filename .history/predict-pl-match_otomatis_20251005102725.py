# predict-pl-match_otomatis.py
import pandas as pd
import numpy as np

# ==================================================
# 1️⃣ Load Player Passing CSV
# ==================================================
def load_passing_data(csv_file="premier_league_player_passing.csv"):
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded passing data: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"⚠️ File {csv_file} tidak ditemukan.")
        return None

# ==================================================
# 2️⃣ Hitung statistik tim otomatis dari CSV
# ==================================================
def get_team_stats(passing_df):
    teams = passing_df['Squad'].unique()
    stats = {}
    for team in teams:
        team_df = passing_df[passing_df['Squad'] == team]
        if team_df.empty:
            continue
        try:
            avg_cmp_perc = team_df['Total_Cmp%'].astype(float).mean() / 100
        except:
            avg_cmp_perc = 0.8
        # Contoh parameter sederhana: form, goals, concede, home_adv
        stats[team] = {
            "form": avg_cmp_perc,       # ganti sesuai logika
            "goals": team_df['Total_TotDist'].astype(float).mean()/50,  # dummy goals proxy
            "concede": 1.0,             # bisa diganti CSV hasil gol kebobolan
            "home_adv": 1.1 if team in ["Brighton","Man City","Arsenal"] else 1.0
        }
    return stats

# ==================================================
# 3️⃣ Prediksi pertandingan
# ==================================================
def predict_match(home_team, away_team, team_stats):
    print(f"\n🔮 Prediksi pertandingan: {home_team} vs {away_team}")

    if home_team not in team_stats or away_team not in team_stats:
        print("⚠️ Data form tidak ditemukan, menggunakan default.")
        team_stats[home_team] = {"form": 0.5, "goals": 1.0, "concede": 1.0, "home_adv": 1.0}
        team_stats[away_team] = {"form": 0.5, "goals": 1.0, "concede": 1.0, "home_adv": 1.0}

    home = team_stats[home_team]
    away = team_stats[away_team]

    # Perhitungan skor probabilitas sederhana
    home_score = home["form"]*0.5 + home["goals"]/(away["concede"]+0.1)*0.3 + home["home_adv"]*0.2
    away_score = away["form"]*0.5 + away["goals"]/(home["concede"]+0.1)*0.3 + away["home_adv"]*0.2

    p_home = home_score / (home_score + away_score)
    p_away = away_score / (home_score + away_score)
    p_draw = 0.15

    print("\n📈 Probabilitas Prediksi:")
    print(f"  🏠 {home_team} menang : {p_home:.2%}")
    print(f"  🚗 {away_team} menang : {p_away:.2%}")
    print(f"  🤝 Seri               : {p_draw:.2%}")

    exp_home_goals = round(home["goals"] * p_home + np.random.uniform(0, 0.5))
    exp_away_goals = round(away["goals"] * p_away + np.random.uniform(0, 0.3))

    print(f"\n⚽ Prediksi Skor: {home_team} {exp_home_goals} – {exp_away_goals} {away_team}")

    result = home_team if p_home>p_away else away_team if p_away>p_home else "Seri"
    print(f"\n🧠 Analisis Akhir: {result} berpeluang menang\n")
    print("✅ Prediksi selesai.")

# ==================================================
# 4️⃣ MAIN
# ==================================================
def main():
    passing_df = load_passing_data()
    if passing_df is None:
        return

    team_stats = get_team_stats(passing_df)

    # Ganti tim di sini
    home_team = "Brighton"
    away_team = "Wolves"

    predict_match(home_team, away_team, team_stats)

if __name__ == "__main__":
    main()
