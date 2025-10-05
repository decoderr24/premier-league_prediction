import pandas as pd
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# =========================
# 1️⃣ Scrape Jadwal EPL
# =========================
def scrape_premier_league_schedule():
    url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    print(f"🌍 Scraping data from {url} ...")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        driver.get(url)
        time.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        table = soup.find("table", id=lambda x: x and "schedule" in x.lower())
        if not table:
            print("❌ Table not found")
            return None

        rows = table.find_all("tr")
        data = []
        for row in rows:
            cols = [c.text.strip() for c in row.find_all(["th", "td"])]
            if len(cols) >= 8:
                data.append(cols[:8])

        df = pd.DataFrame(data, columns=["Date", "Time", "Home", "xG_H", "Score", "xG_A", "Away", "Attendance"])
        df = df[df["Home"].notna()]
        print(f"✅ Match table loaded: {len(df)} rows")
        return df
    finally:
        driver.quit()


# =========================
# 2️⃣ Load Player Passing Data
# =========================
def load_passing_data():
    try:
        df = pd.read_csv("premier_league_player_passing.csv")
        print(f"📊 Loaded passing data: {df.shape}")
        return df
    except FileNotFoundError:
        print("⚠️ premier_league_player_passing.csv not found")
        return None


# =========================
# 3️⃣ Hitung Statistik Tim Otomatis
# =========================
def get_team_stats(schedule_df, passing_df):
    teams = schedule_df["Home"].unique()
    team_stats = {}

    for team in teams:
        # Formasi: menang=1, draw=0.5, kalah=0
        team_games = schedule_df[(schedule_df["Home"] == team) | (schedule_df["Away"] == team)]
        total_games = len(team_games)
        if total_games == 0:
            continue

        wins, draws, losses = 0, 0, 0
        goals_for, goals_against = 0, 0
        for _, row in team_games.iterrows():
            if pd.isna(row["Score"]) or "-" not in row["Score"]:
                continue
            try:
                h, a = row["Score"].split("–")
                h, a = int(h), int(a)
            except:
                continue

            if row["Home"] == team:
                goals_for += h
                goals_against += a
                if h > a:
                    wins += 1
                elif h == a:
                    draws += 1
                else:
                    losses += 1
            else:
                goals_for += a
                goals_against += h
                if a > h:
                    wins += 1
                elif a == h:
                    draws += 1
                else:
                    losses += 1

        form = (wins + 0.5 * draws) / (wins + draws + losses) if (wins + draws + losses) > 0 else 0.5
        avg_goals = goals_for / total_games if total_games > 0 else 1.0
        avg_concede = goals_against / total_games if total_games > 0 else 1.0

        # Passing accuracy rata-rata pemain
        team_pass = passing_df[passing_df["Squad"].str.contains(team, case=False, na=False)]
        if not team_pass.empty:
            try:
                avg_pass = team_pass["Total_Cmp%"].astype(float).mean() / 100
            except:
                avg_pass = 0.8
        else:
            avg_pass = 0.8

        # Faktor home advantage default
        home_adv = 1.1
        team_stats[team] = {"form": form, "goals": avg_goals, "concede": avg_concede, "home_adv": home_adv, "passing": avg_pass}

    return team_stats


# =========================
# 4️⃣ Prediksi Pertandingan
# =========================
def predict_match(home_team, away_team, team_stats):
    print(f"\n🔮 Predicting match: {home_team} vs {away_team}")

    if home_team not in team_stats or away_team not in team_stats:
        print("⚠️ Missing team data, using defaults")
        defaults = {"form": 0.5, "goals": 1.0, "concede": 1.0, "home_adv": 1.0, "passing": 0.8}
        if home_team not in team_stats:
            team_stats[home_team] = defaults
        if away_team not in team_stats:
            team_stats[away_team] = defaults

    ht = team_stats[home_team]
    at = team_stats[away_team]

    home_score = ht["form"] * 0.5 + ht["goals"] / (at["concede"] + 0.1) * 0.3 + ht["passing"] * 0.1 + ht["home_adv"] * 0.1
    away_score = at["form"] * 0.5 + at["goals"] / (ht["concede"] + 0.1) * 0.3 + at["passing"] * 0.1 + at["home_adv"] * 0.1

    p_home = home_score / (home_score + away_score)
    p_away = away_score / (home_score + away_score)
    p_draw = 0.15

    print("\n📈 Probabilities:")
    print(f"  🏠 {home_team} win : {p_home:.2%}")
    print(f"  🚗 {away_team} win : {p_away:.2%}")
    print(f"  🤝 Draw           : {p_draw:.2%}")

    exp_home_goals = round(ht["goals"] * p_home + np.random.uniform(0, 0.5))
    exp_away_goals = round(at["goals"] * p_away + np.random.uniform(0, 0.3))

    print(f"\n⚽ Predicted Score: {home_team} {exp_home_goals} – {exp_away_goals} {away_team}")

    if p_home > p_away:
        result = f"{home_team} berpeluang menang"
    elif p_away > p_home:
        result = f"{away_team} berpeluang menang"
    else:
        result = "kemungkinan besar imbang"

    print(f"\n🧠 Final Analysis: {result}")


# =========================
# 5️⃣ MAIN
# =========================
def main():
    schedule_df = scrape_premier_league_schedule()
    passing_df = load_passing_data()

    if schedule_df is None or passing_df is None:
        print("❌ Cannot continue, missing data")
        return

    team_stats = get_team_stats(schedule_df, passing_df)

    # Ganti nama tim yang ingin diprediksi di sini
    home_team = "Brentford"
    away_team = "Man City"

    predict_match(home_team, away_team, team_stats)


if __name__ == "__main__":
    main()
