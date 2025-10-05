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
    options.add_argument("--headless") # Kita nyalakan lagi headless agar cepat
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
    except TimeoutException:
        print("The stats table could not be found on the page.")
        driver.quit()
        return None
    
    html_source = driver.page_source
    driver.quit()
    
    print("Data downloaded. Processing with pandas...")

    # ==============================================================================
    # === BAGIAN INVESTIGASI BARU ===
    # ==============================================================================
    # 1. Baca SEMUA tabel di halaman, jangan hanya ambil yang pertama [0]
    all_tables = pd.read_html(StringIO(html_source))
    print(f"\nDEBUG: Found {len(all_tables)} tables on the page.")

    # 2. Cetak ukuran (baris, kolom) dari setiap tabel yang ditemukan
    for i, table in enumerate(all_tables):
        print(f"DEBUG: Table [{i}] has shape: {table.shape}")
    
    # 3. Kita akan pilih tabel pertama untuk sementara agar bisa melihat output debug
    # Ini akan menyebabkan error lagi, tapi itu tidak apa-apa.
    df = all_tables[0]
    # ==============================================================================
    
    df = df[[1, 5, 6, 7, 8]]
    df.columns = ['Squad', 'Total_Cmp', 'Total_Att', 'Total_Cmp%', 'Total_TotDist']
    df = df.iloc[:-1]

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