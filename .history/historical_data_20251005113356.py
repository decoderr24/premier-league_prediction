import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from io import StringIO
import time
import sys
import random
import numpy as np

# Fungsi-fungsi (calculate_team_passing_avg, scrape_season_data) tidak perlu diubah
def calculate_team_passing_avg(passing_stats_file):
    try:
        df_pass = pd.read_csv(passing_stats_file)
        if "Squad" not in df_pass.columns or "Total_Cmp%" not in df_pass.columns:
            print(f"❌ Error: Kolom 'Squad' atau 'Total_Cmp%' tidak ditemukan di {passing_stats_file}")
            return None

        df_pass['Total_Cmp%'] = pd.to_numeric(df_pass['Total_Cmp%'], errors='coerce')
        team_avg_pass = df_pass.groupby('Squad')['Total_Cmp%'].mean().reset_index()
        team_avg_pass.rename(columns={'Total_Cmp%': 'AvgPass%'}, inplace=True)
        print("✅ Berhasil menghitung rata-rata passing % per tim.")
        return team_avg_pass
    except FileNotFoundError:
        print(f"❌ Error: File '{passing_stats_file}' tidak ditemukan.")
        return None

def scrape_season_data(url, driver):
    print(f"\n   - Mengakses: {url}")
    try:
        driver.get(url)
        print("     ⏳ Menunggu tabel data muncul...")
        table_element = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "table.stats_table")))
        print("     ✅ Tabel ditemukan, mengambil data HTML...")
        html_source = table_element.get_attribute('outerHTML')
        df_season = pd.read_html(StringIO(html_source))[0]
        print(f"     👍 Berhasil mengambil data untuk musim ini.")
        return df_season
    except Exception as e:
        print(f"     ❌ Gagal mengambil data dari URL ini."); return None

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    PASSING_STATS_FILE = "premier_league_player_passing.csv"
    OUTPUT_FILE = "historical_matches.csv"

    team_pass_avg_df = calculate_team_passing_avg(PASSING_STATS_FILE)
    if team_pass_avg_df is None:
        sys.exit()

    seasons_urls = [
        "https://fbref.com/en/comps/9/schedule/2023-2024/Premier-League-Scores-and-Fixtures",
        "https://fbref.com/en/comps/9/schedule/2022-2023/Premier-League-Scores-and-Fixtures",
        "https://fbref.com/en/comps/9/schedule/2021-2022/Premier-League-Scores-and-Fixtures",
        "https://fbref.com/en/comps/9/schedule/2020-2021/Premier-League-Scores-and-Fixtures"
    ]
    all_matches_df = pd.DataFrame()

    print("\n🌐 Memulai proses pengambilan data dari 4 musim terakhir...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = None
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver.get("https://fbref.com")
        try:
            wait = WebDriverWait(driver, 5)
            accept_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[text()="Accept All"]')))
            accept_button.click()
            print("✅ Cookie banner diterima.")
        except:
            print("ℹ️ Tidak ada cookie banner atau sudah diterima.")

        for url in seasons_urls:
            season_df = scrape_season_data(url, driver)
            if season_df is not None:
                all_matches_df = pd.concat([all_matches_df, season_df], ignore_index=True)
            jeda = random.randint(5, 10)
            print(f"     ☕ Istirahat sejenak selama {jeda} detik...")
            time.sleep(jeda)
    finally:
        if driver:
            driver.quit()
            print("\n✅ Browser ditutup.")

    if all_matches_df.empty:
        print("❌ Tidak ada data yang berhasil diambil. Proses dihentikan.")
        sys.exit()

    print("\n⚙️ Memproses semua data pertandingan...")
    df_matches = all_matches_df[['Date', 'Home', 'Score', 'Away']]
    df_matches.dropna(subset=['Score'], inplace=True)
    df_matches = df_matches[df_matches['Score'].str.contains('–', na=False)]
    scores = df_matches['Score'].str.split('–', expand=True)
    df_matches['HomeGoals'] = pd.to_numeric(scores[0])
    df_matches['AwayGoals'] = pd.to_numeric(scores[1])

    print("🔄 Menggabungkan data pertandingan dengan data passing...")
    pass_map = {row['Squad']: row['AvgPass%'] for index, row in team_pass_avg_df.iterrows()}

    def get_pass_perc(team_name):
        for team, perc in pass_map.items():
            if team_name in team:
                return perc
        print(f"❌ Tidak dapat menemukan data passing untuk tim: {team_name}")
        return np.nan  # Kembalikan NaN jika tidak ditemukan

    df_matches['HomePass%'] = df_matches['Home'].apply(get_pass_perc)
    df_matches['AwayPass%'] = df_matches['Away'].apply(get_pass_perc)

    final_df = df_matches[['Date', 'Home', 'Away', 'HomeGoals', 'AwayGoals', 'HomePass%', 'AwayPass%']]
    final_df = final_df.round(1)

    try:
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n🎉 SUKSES! File '{OUTPUT_FILE}' berhasil dibuat dengan {len(final_df)} data pertandingan.")
        print("   Sekarang Anda bisa menjalankan script prediksi utama Anda.")
    except Exception as e:
        print(f"❌ Gagal menyimpan file CSV: {e}")