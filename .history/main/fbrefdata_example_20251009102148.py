import time
import pandas as pd
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def pull_premier_league_passing():
    """
    Ambil data passing (otomatis deteksi: tim atau pemain)
    dari halaman FBref Premier League terbaru.
    """
    # URL utama
    url = "https://fbref.com/en/comps/9/passing/Premier-League-Stats"
    print(f"🌐 Opening browser to download passing stats from {url} ...")

    # --- Setup browser Chrome ---
    options = ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--headless")  # aktifkan jika ingin tanpa tampilan browser
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.get(url)
f
    # --- Handle cookie banner (jika muncul) ---
    try:
        wait = WebDriverWait(driver, 10)
        accept_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Accept All Cookies')]")
        ))
        accept_button.click()
        print("🍪 Cookie banner accepted.")
    except TimeoutException:
        print("No cookie banner found or it took too long.")

    # --- Coba deteksi tabel TIM terlebih dahulu ---
    table_html = None
    try:
        wait = WebDriverWait(driver, 15)
        div_team = wait.until(EC.presence_of_element_located((By.ID, "all_stats_passing_team")))
        print("✅ Team passing table found.")
        table_html = div_team.get_attribute("outerHTML")
        table_type = "team"
    except TimeoutException:
        print("⚠️ Team passing table not found. Trying player table...")

        # --- Fallback ke tabel pemain ---
        try:
            div_player = wait.until(EC.presence_of_element_located((By.ID, "all_stats_passing")))
            print("✅ Player passing table found.")
            table_html = div_player.get_attribute("outerHTML")
            table_type = "player"
        except TimeoutException:
            print("❌ No passing table found at all. Saving debug files...")
            driver.save_screenshot('debug_screenshot.png')
            with open('debug_page.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            driver.quit()
            return None

    driver.quit()
    print("📄 Data downloaded. Processing with pandas...")

    # --- Parse HTML table ke DataFrame ---
    df = pd.read_html(StringIO(table_html))[0]
    print(f"✅ Table found with shape: {df.shape}")

    # Gabungkan header dua baris (jika ada)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Pilih kolom relevan
    cols_to_use = [c for c in df.columns if any(x in c for x in ['Squad', 'Player', 'Cmp', 'Att', 'Cmp%', 'TotDist'])]
    df = df[cols_to_use]

    # Normalisasi nama kolom
    rename_map = {}
    for c in df.columns:
        if 'Squad' in c: rename_map[c] = 'Squad'
        elif 'Player' in c: rename_map[c] = 'Player'
        elif 'Cmp%' in c: rename_map[c] = 'Total_Cmp%'
        elif 'Cmp' in c and 'Cmp%' not in c: rename_map[c] = 'Total_Cmp'
        elif 'Att' in c: rename_map[c] = 'Total_Att'
        elif 'TotDist' in c: rename_map[c] = 'Total_TotDist'
    df.rename(columns=rename_map, inplace=True)

    # Bersihkan baris kosong / header duplikat
    if 'Squad' in df.columns:
        df = df[df['Squad'].notna()]
        df = df[~df['Squad'].str.contains("Squad|Rk", na=False)]

    print(f"✅ Cleaned dataframe shape: {df.shape}")
    return df, table_type


def filter_teams(df, teams):
    """Filter baris berdasarkan nama tim"""
    if "Squad" not in df.columns:
        print("⚠️ 'Squad' column not found, skipping team filter.")
        return df
    return df[df["Squad"].isin(teams)]


def main():
    df, table_type = pull_premier_league_passing()
    if df is not None:
        # Simpan hasil
        filename = f"premier_league_{table_type}_passing.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Saved to {filename}")

        # Filter contoh tim
        teams = ["Arsenal", "Wolves", "Brighton"]
        df_filtered = filter_teams(df, teams)
        print(f"\n📊 Passing Stats ({table_type.title()} Level) for selected teams")
        print("=" * 80)
        print(df_filtered.head())


if __name__ == "__main__":
    main()
