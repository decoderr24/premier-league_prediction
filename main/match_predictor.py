import pandas as pd
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager


# ==============================================
# 1️⃣ FUNGSI UNTUK SCRAPE JADWAL PREMIER LEAGUE
# ==============================================
def scrape_premier_league_schedule():
    url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    print(f"🌍 Opening browser to scrape data from: {url}")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(url)
        time.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        table = soup.find("table", id=lambda x: x and "schedule" in x.lower())
        if not table:
            table = soup.find("table", class_=lambda x: x and "stats_table" in x.lower())

        if not table:
            print("❌ Match table not found. Saving debug page...")
            with open("debug_schedule.html", "w", encoding="utf-8") as f:
                f.write(soup.prettify())
            driver.quit()
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

    except Exception as e:
        print(f"❌ Gagal mengakses halaman FBref: {e}")
        return None
    finally:
        driver.quit()


# ==================================================
# 2️⃣ FUNGSI UNTUK MEMUAT DATA PASSING (DARI CSV)
# ==================================================
def load_passing_data():
    try:
        df = pd.read_csv("premier_league_player_passing.csv")
        print(f"📊 Loaded passing data: {df.shape}")
        return df
    except FileNotFoundError:
        print("⚠️ Tidak ditemukan file premier_league_player_passing.csv.")
        return None


# ==================================================
# 3️⃣ MODEL PREDIKSI BERBASIS STATISTIK
# ==================================================
def predict_match(home_team, away_team, passing_df=None):
    print(f"\n🔮 Memprediksi pertandingan: {home_team} vs {away_team}")
    
    # --- Data dasar asumsi form ---
    team_stats = {
        "Man City": {"form": 0.85, "goals": 2.3, "concede": 0.7, "home_adv": 1.2},
        "Brentford": {"form": 0.45, "goals": 1.1, "concede": 1.5, "home_adv": 1.0},
    }

    if home_team not in team_stats or away_team not in team_stats:
        print("⚠️ Data form tidak ditemukan, menggunakan nilai default.")
        team_stats[home_team] = {"form": 0.5, "goals": 1.0, "concede": 1.0, "home_adv": 1.0}
        team_stats[away_team] = {"form": 0.5, "goals": 1.0, "concede": 1.0, "home_adv": 1.0}

    # --- Faktor performa passing jika tersedia ---
    if passing_df is not None:
        home_pass = passing_df[passing_df["Squad"].str.contains(home_team, case=False, na=False)]
        away_pass = passing_df[passing_df["Squad"].str.contains(away_team, case=False, na=False)]

        def avg_pass(df):
            if df.empty:
                return 0
            try:
                return df["Total_Cmp%"].astype(float).mean() / 100
            except:
                return 0.8

        home_passing = avg_pass(home_pass)
        away_passing = avg_pass(away_pass)
    else:
        home_passing, away_passing = 0.82, 0.76  # default rata-rata passing EPL

    # --- Perhitungan skor probabilitas ---
    home_score = (
        team_stats[home_team]["form"] * 0.5 +
        team_stats[home_team]["goals"] / (team_stats[away_team]["concede"] + 0.1) * 0.3 +
        home_passing * 0.1 +
        team_stats[home_team]["home_adv"] * 0.1
    )

    away_score = (
        team_stats[away_team]["form"] * 0.5 +
        team_stats[away_team]["goals"] / (team_stats[home_team]["concede"] + 0.1) * 0.3 +
        away_passing * 0.1 +
        team_stats[away_team]["home_adv"] * 0.1
    )

    p_home = home_score / (home_score + away_score)
    p_away = away_score / (home_score + away_score)
    p_draw = 0.15

    print("\n📈 Probabilitas Prediksi:")
    print(f"  🏠 {home_team} menang : {p_home:.2%}")
    print(f"  🚗 {away_team} menang : {p_away:.2%}")
    print(f"  🤝 Seri               : {p_draw:.2%}")

    # --- Prediksi skor ---
    exp_home_goals = round(team_stats[home_team]["goals"] * p_home + np.random.uniform(0, 0.5))
    exp_away_goals = round(team_stats[away_team]["goals"] * p_away + np.random.uniform(0, 0.3))

    print(f"\n⚽ Prediksi Skor: {home_team} {exp_home_goals} – {exp_away_goals} {away_team}")

    if p_home > p_away:
        result = f"{home_team} berpeluang menang"
    elif p_away > p_home:
        result = f"{away_team} berpeluang menang"
    else:
        result = "kemungkinan besar imbang"

    print(f"\n🧠 Analisis Akhir: {result}\n")
    print("✅ Prediksi selesai.")


# ==================================================
# 4️⃣ MAIN PROGRAM
# ==================================================
def main():
    schedule_df = scrape_premier_league_schedule()
    passing_df = load_passing_data()

    if schedule_df is None:
        print("❌ Gagal mendapatkan data pertandingan.")
        return

    # Cek apakah ada Brighton vs Wolves
    mask = (
        (schedule_df["Home"].str.contains("Brighton", na=False) & schedule_df["Away"].str.contains("Wolves", na=False)) |
        (schedule_df["Home"].str.contains("Wolves", na=False) & schedule_df["Away"].str.contains("Brighton", na=False))
    )

    match_df = schedule_df[mask]
    if match_df.empty:
        print("⚠️ Tidak ditemukan pertandingan Brighton vs Wolves dalam jadwal.")
    else:
        print("\n📅 Jadwal Pertandingan Ditemukan:")
        print(match_df.tail(1)[["Date", "Home", "Away"]].to_string(index=False))

    predict_match("Brentford", "Man City", passing_df)


if __name__ == "__main__":
    main()
