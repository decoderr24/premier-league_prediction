# predict-pl-match_realistic.py
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
# 2️⃣ Load Team Goals & Concede CSV (optional)
# ==================================================
def load_team_goals(csv_file="premier_league_team_goals.csv"):
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded team goals/concede data: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"⚠️ File {csv_file} tidak ditemukan. Default digunakan.")
        return None

# ==================================================
# 3️⃣ Hitung statistik tim
# ==================================================
def get_team_stats(passing_df, goals_df=None):
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

        if goals_df is not None and team in goals_df['Team'].values:
            g = goals_df[goals_df['Team']==team]
            goals = float(g['Goals_For'])
            concede = float(g['Goals_Against'])
        else:
            goals = team_df['Total_TotDist'].astype(float).mean()/50
            concede = 1.0

        stats[team] = {
            "form": avg_cmp_perc,
            "goals": goals,
            "concede": concede,
            "home_adv": 1.1 if team in ["Brighton","Man City","Arsenal"] else 1.0
        }
    return stats

# ==================================================
# 4️⃣ Prediksi pertandingan
# ==================================================
def predict_match(home_team, away_team, team_stats):
    print(f"\n🔮 Prediksi pertandingan: {home_team} vs {away_team}")

    if home_team not in team_stats or away_team not in team_stats:
        print("⚠️ Data form tidak ditemukan, menggunakan default.")
        team_stats[home_team] = {"form": 0.5, "goals": 1.0, "concede": 1.0, "home_adv": 1.0}
        team_stats[away_team] = {"form": 0.5, "goals": 1.0, "concede": 1.0, "home_adv": 1.0}

    home = team_stats[home_team]
    away = team_stats[away_team]

    # Perhitungan skor probabilitas berbasis form, goals & concede historis
    home_score = home["form"]*0.4 + home["goals"]/(away["concede"]+0.1)*0.4 + home["home_adv"]*0.2
    away_score = away["form"]*0.4 + away["goals"]/(home["concede"]+0.1)*0.4 + away["home_adv"]*0.2

    p_home = home_score / (home_score + away_score)
    p_away = away_score / (home_score + away_score)
    p_draw = 0.15

    print("\n📈 Probabilitas Prediksi:")
    print(f"  🏠 {home_team} menang : {p_home:.2%}")
    print(f"  🚗 {away_team} menang : {p_away:.2%}")
    print(f"  🤝 Seri               : {p_draw:.2%}")

    # Prediksi skor realistis
    exp_home_goals = max(0, round(home["goals"] * p_home + np.random.uniform(0, 0.5)))
    exp_away_goals = max(0, round(away["goals"] * p_away + np.random.uniform(0, 0.5)))

    print(f"\n⚽ Prediksi Skor: {home_team} {exp_home_goals} – {exp_away_goals} {away_team}")

    result = home_team if p_home>p_away else away_team if p_away>p_home else "Seri"
    print(f"\n🧠 Analisis Akhir: {result} berpeluang menang\n")
    print("✅ Prediksi selesai.")

# ==================================================
# 5️⃣ MAIN
# ==================================================
def main():
    passing_df = load_passing_data()
    goals_df = load_team_goals()  # optional, buat CSV: Team, Goals_For, Goals_Against

    if passing_df is None:
        return

    team_stats = get_team_stats(passing_df, goals_df)

    # Ganti tim yang mau diprediksi
    home_team = "Man City"
    away_team = "Brentford"

    predict_match(home_team, away_team, team_stats)

if __name__ == "__main__":
    main()
