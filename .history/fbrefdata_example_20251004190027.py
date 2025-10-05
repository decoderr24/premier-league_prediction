import time
import pandas as pd
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager

def pull_premier_league_team_passing():
    url = "https://fbref.com/en/comps/9/passing/Premier-League-Stats"
    print(f"Opening browser to download team passing stats from {url} ...")

    options = ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--headless") # Menjalankan browser di background agar tidak muncul jendela
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    
    driver.get(url)
    time.sleep(3) # Cukup 3 detik jika headless
    
    html_source = driver.page_source
    driver.quit()
    
    print("Data downloaded. Processing with pandas...")

    # Ambil tabel pertama dari HTML
    df = pd.read_html(StringIO(html_source))[0]

    # ==============================================================================
    # === BAGIAN LAMA DIHAPUS DAN DIGANTI DENGAN YANG LEBIH SEDERHANA INI ===
    # ==============================================================================
    # Berdasarkan struktur tabel di FBRef, kita tahu kolom yang kita mau ada di indeks:
    # 1: Squad, 5: Total Cmp, 6: Total Att, 7: Total Cmp%, 8: Total TotDist
    
    # 1. Pilih hanya kolom yang kita butuhkan berdasarkan nomor indeksnya
    df = df[[1, 5, 6, 7, 8]]
    
    # 2. Beri nama baru untuk kolom-kolom tersebut
    df.columns = ['Squad', 'Total_Cmp', 'Total_Att', 'Total_Cmp%', 'Total_TotDist']
    
    # 3. Hapus baris terakhir yang biasanya berisi total/rata-rata liga
    df = df.iloc[:-1]
    # ==============================================================================
    # ==============================================================================

    return df

def filter_teams(df, teams):
    # Fungsi ini sekarang akan berhasil karena kolom 'Squad' sudah ada
    return df[df["Squad"].isin(teams)]

def main():
    df = pull_premier_league_team_passing()

    teams = ["Arsenal", "Nott'ham Forest"]
    df_filtered = filter_teams(df, teams)

    print("\n📊 Passing Stats for Arsenal & Nottingham Forest (Team Level)")
    print("=" * 70)
    # Karena df_filtered sekarang hanya berisi kolom yang kita mau, kita bisa print langsung
    print(df_filtered)

if __name__ == "__main__":
    main()