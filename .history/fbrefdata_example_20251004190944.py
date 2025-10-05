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
        wait = WebDriverWait(driver, 10)
        wait.until(EC.visibility_of_element_located((By.ID, "div_stats_passing")))
        print("Stats table is now visible.")
        html_source = driver.page_source
        all_tables = pd.read_html(StringIO(html_source))
    except TimeoutException:
        print("The stats table could not be found on the page. Saving debug files...")
        driver.save_screenshot('debug_screenshot.png')
        with open('debug_page.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        driver.quit()
        return None

    driver.quit()
    print("Data downloaded. Processing with pandas...")

    # Ambil tabel utama (yang paling banyak kolomnya)
    main_df = max(all_tables, key=lambda df: len(df.columns))
    print(f"Main table selected with shape: {main_df.shape}")

    # Jika kolom multi-level (MultiIndex), kita gabungkan nama header-nya
    if isinstance(main_df.columns, pd.MultiIndex):
        main_df.columns = ['_'.join(col).strip() for col in main_df.columns.values]

    # Coba tampilkan beberapa kolom agar tahu nama sebenarnya
    print("Available columns:", main_df.columns[:10].tolist())

    # Cari kolom yang relevan untuk passing
    cols_to_use = [c for c in main_df.columns if any(x in c for x in ['Squad', 'Cmp', 'Att', 'Cmp%', 'TotDist'])]
    df = main_df[cols_to_use]

    # Normalisasi nama kolom agar lebih rapi
    df.columns = ['Squad', 'Total_Cmp', 'Total_Att', 'Total_Cmp%', 'Total_TotDist']
    df = df[df['Squad'].notna() & (df['Squad'] != 'Squad')]  # hapus header duplikat

    return df


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
