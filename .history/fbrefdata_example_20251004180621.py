import time
import pandas as pd
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

def pull_premier_league_team_passing():
    url = "https://fbref.com/en/comps/9/passing/Premier-League-Stats"
    print(f"Opening browser to download team passing stats from {url} ...")

    # Inisialisasi driver Chrome secara otomatis
    # Browser akan terbuka, mengambil data, lalu menutup sendiri
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    
    # Buka URL
    driver.get(url)
    
    # Beri waktu 3 detik agar halaman dan semua elemennya (termasuk tabel)
    # termuat dengan sempurna
    time.sleep(3)
    
    # Ambil sumber HTML dari halaman yang sudah dimuat oleh browser
    html_source = driver.page_source
    
    # Tutup browser setelah selesai
    driver.quit()
    
    print("Data downloaded. Processing with pandas...")

    # Sekarang kita proses HTML yang didapat dengan pandas, sama seperti sebelumnya
    df = pd.read_html(StringIO(html_source))[0]

    # Meratakan kolom
    df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    # Mengganti nama kolom yang aneh
    df = df.rename(columns={
        "Unnamed: 0_level_0_Squad": "Squad",
        "Unnamed: 1_level_0_# Pl": "Players",
        "Unnamed: 2_level_0_90s": "90s",
        "Unnamed: 17_level_0_Ast": "Ast",
        "Unnamed: 18_level_0_xAG": "xAG",
        "Unnamed: 21_level_0_KP": "KP",
        "Unnamed: 22_level_0_1/3": "1/3",
        "Unnamed: 23_level_0_PPA": "PPA",
        "Unnamed: 24_level_0_CrsPA": "CrsPA",
        "Unnamed: 25_level_0_PrgP": "PrgP"
    })

    return df

def filter_teams(df, teams):
    return df[df["Squad"].isin(teams)]

def main():
    df = pull_premier_league_team_passing()

    teams = ["Arsenal", "Nott'ham Forest"]
    df_filtered = filter_teams(df, teams)

    print("\n📊 Passing Stats for Arsenal & Nottingham Forest (Team Level)")
    print("=" * 70)
    print(df_filtered[["Squad", "Total_Cmp", "Total_Att", "Total_Cmp%", "Total_TotDist"]])

if __name__ == "__main__":
    main()