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


def pull_premier_league_team_passing():
    url = "https://fbref.com/en/comps/9/passing/Premier-League-Stats"
    print(f"Opening browser to download team passing stats from {url} ...")

    options = ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--headless")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        wait = WebDriverWait(driver, 10)
        accept_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept All Cookies')]")))
        accept_button.click()
        print("Cookie banner accepted.")
    except TimeoutException:
        print("No cookie banner found or it took too long.")

    try:
        wait = WebDriverWait(driver, 15)
        wait.until(EC.visibility_of_element_located((By.ID, "stats_passing_team")))
        print("Team stats table is visible.")
        html_source = driver.page_source
    except TimeoutException:
        print("❌ The team stats table could not be found on the page. Saving debug files...")
        driver.save_screenshot('debug_screenshot.png')
        with open('debug_page.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        driver.quit()
        return None

    driver.quit()
    print("Data downloaded. Processing with pandas...")

    # Ambil hanya tabel team passing
    all_tables = pd.read_html(StringIO(html_source))
    team_df = None
    for df in all_tables:
        if 'Squad' in df.columns:
            team_df = df
            break

    if team_df is None:
        print("❌ No team table found.")
        return None

    print(f"✅ Found team table with shape: {team_df.shape}")

    # Bersihkan kolom header ganda jika ada
    if isinstance(team_df.columns, pd.MultiIndex):
        team_df.columns = ['_'.join(col).strip() for col in team_df.columns.values]

    # Ambil kolom yang relevan
    cols_to_use = [c for c in team_df.columns if any(x in c for x in ['Squad', 'Cmp', 'Att', 'Cmp%', 'TotDist'])]
    team_df = team_df[cols_to_use]

    # Normalisasi nama kolom
    rename_map = {}
    for c in team_df.columns:
        if 'Squad' in c: rename_map[c] = 'Squad'
        elif 'Cmp%' in c: rename_map[c] = 'Total_Cmp%'
        elif 'Cmp' in c and 'Cmp%' not in c: rename_map[c] = 'Total_Cmp'
        elif 'Att' in c: rename_map[c] = 'Total_Att'
        elif 'TotDist' in c: rename_map[c] = 'Total_TotDist'
    team_df.rename(columns=rename_map, inplace=True)

    # Hapus baris duplikat atau NaN
    team_df = team_df[team_df['Squad'].notna()]
    team_df = team_df[~team_df['Squad'].str.contains("Squad|Rk", na=False)]

    return team_df


def filter_teams(df, teams):
    return df[df["Squad"].isin(teams)]


def main():
    df = pull_premier_league_team_passing()
    if df is not None:
        teams = ["Arsenal", "Nott'ham Forest"]
        df_filtered = filter_teams(df, teams)
        print("\n📊 Passing Stats for Arsenal & Nottingham Forest (Team Level)")
        print("=" * 70)
        print(df_filtered)


if __name__ == "__main__":
    main()
