import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


# ===================================================
# 1️⃣ SCRAPER MENGGUNAKAN SELENIUM (ANTI CLOUDFLARE)
# ===================================================
def get_fixtures_table(url):
    print(f"🌍 Opening browser to scrape data from: {url}")

    # Setup browser
    options = ChromeOptions()
    options.add_argument("--headless=new")  # Jalankan tanpa GUI
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.get(url)

    # Tunggu halaman load
    time.sleep(5)

    try:
        # Tunggu tabel schedule muncul
        wait = WebDriverWait(driver, 15)
        div_element = wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@id, 'sched') and .//table]")))
        print(f"✅ Ditemukan tabel jadwal dengan id: {div_element.get_attribute('id')}")

        html = div_element.get_attribute("outerHTML")
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        df = pd.read_html(str(table))[0]
        driver.quit()
        print(f"✅ Table ditemukan dengan shape: {df.shape}")
        return df

    except TimeoutException:
        print("❌ Match table not found. Saving debug page...")
        with open("debug_schedule.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        driver.quit()
        return None


# ===================================================
# 2️⃣ FUNGSI PREDIKSI HASIL PERTANDINGAN
# ===================================================
def predict_match(df, team1, team2):
    print(f"\n🔮 Memprediksi hasil: {team1} vs {team2}")

    df = df.dropna(subset=['Home', 'Away'])
    df = df[df['Score'].notna()]

    # Ambil 5 laga terakhir
    team1_recent = df[(df['Home'] == team1) | (df['Away'] == team1)].tail(5)
    team2_recent = df[(df['Home'] == team2) | (df['Away'] == team2)].tail(5)

    if team1_recent.empty or team2_recent.empty:
        print("⚠️ Salah satu tim belum punya cukup data.")
        return None

    def get_stats(team_df, name):
        win = lose = draw = 0
        goals_scored = goals_conceded = 0
        for _, row in team_df.iterrows():
            try:
                home_goals, away_goals = map(int, str(row['Score']).replace('-', '–').split('–'))
            except:
                continue
            if row['Home'] == name:
                goals_scored += home_goals
                goals_conceded += away_goals
                if home_goals > away_goals:
                    win += 1
                elif home_goals < away_goals:
                    lose += 1
                else:
                    draw += 1
            else:
                goals_scored += away_goals
                goals_conceded += home_goals
                if away_goals > home_goals:
                    win += 1
                elif away_goals < home_goals:
                    lose += 1
                else:
                    draw += 1

        total = win + lose + draw if (win + lose + draw) > 0 else 1
        return {
            "win_rate": win / total,
            "avg_goals": goals_scored / total,
            "avg_conceded": goals_conceded / total
        }

    # Hitung statistik
    t1 = get_stats(team1_recent, team1)
    t2 = get_stats(team2_recent, team2)

    print(f"\n📈 Statistik {team1}: {t1}")
    print(f"📉 Statistik {team2}: {t2}")

    # Hitung kekuatan relatif
    s1 = t1['win_rate'] * 0.6 + t1['avg_goals'] * 0.3 - t1['avg_conceded'] * 0.1
    s2 = t2['win_rate'] * 0.6 + t2['avg_goals'] * 0.3 - t2['avg_conceded'] * 0.1

    p1 = round((s1 / (s1 + s2)) * 100, 1)
    p2 = round(100 - p1, 1)

    print(f"\n📊 Probabilitas Prediksi:")
    print(f"⚽ {team1} menang: {p1}%")
    print(f"⚽ {team2} menang: {p2}%")

    # Hasil
    if p1 > 55:
        print(f"\n🏆 Prediksi: {team1} Menang\n")
    elif p2 > 55:
        print(f"\n🏆 Prediksi: {team2} Menang\n")
    else:
        print("\n🤝 Prediksi: Seri\n")


# ===================================================
# 3️⃣ MAIN PROGRAM
# ===================================================
if __name__ == "__main__":
    url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    df = get_fixtures_table(url)
    if df is not None:
        predict_match(df, "Brighton", "Wolves")
    else:
        print("❌ Gagal mendapatkan data pertandingan.")
