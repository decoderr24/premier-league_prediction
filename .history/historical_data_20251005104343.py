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

# --- FUNGSI UNTUK MENGHITUNG RATA-RATA PASSING % PER TIM ---
def calculate_team_passing_avg(passing_stats_file):
    """
    Membaca file statistik passing pemain dan menghitung rata-rata
    persentase passing ('Total_Cmp%') untuk setiap tim.
    """
    try:
        df_pass = pd.read_csv(passing_stats_file)
        if "Squad" not in df_pass.columns or "Total_Cmp%" not in df_pass.columns:
            print(f"❌ Error: Kolom 'Squad' atau 'Total_Cmp%' tidak ditemukan di {passing_stats_file}")
            return None

        # Mengubah tipe data dan menghitung rata-rata
        df_pass['Total_Cmp%'] = pd.to_numeric(df_pass['Total_Cmp%'], errors='coerce')
        team_avg_pass = df_pass.groupby('Squad')['Total_Cmp%'].mean().reset_index()
        team_avg_pass.rename(columns={'Total_Cmp%': 'AvgPass%'}, inplace=True)
        print("✅ Berhasil menghitung rata-rata passing % per tim.")
        return team_avg_pass

    except FileNotFoundError:
        print(f"❌ Error: File '{passing_stats_file}' tidak ditemukan.")
        print("   Pastikan file ini ada di folder yang sama.")
        return None
    except Exception as e:
        print(f"❌ Terjadi error saat memproses {passing_stats_file}: {e}")
        return None


# --- FUNGSI UTAMA UNTUK SCRAPING DATA PERTANDINGAN ---
def scrape_historical_matches():
    """
    Scrape data pertandingan historis dari FBref menggunakan Selenium.
    """
    # URL untuk data Premier League musim 2023-2024 yang sudah selesai
    url = "https://fbref.com/en/comps/9/schedule/2023-2024/Premier-League-Scores-and-Fixtures"
    print(f"🌐 Mengakses halaman: {url}")

    options = webdriver.ChromeOptions()
    options.add_argument("--headless") # Jalankan di background tanpa membuka browser
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

    driver = None
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver.get(url)

        # Coba klik cookie banner jika ada
        try:
            wait = WebDriverWait(driver, 5)
            accept_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[text()="Accept All"]')))
            accept_button.click()
            print("✅ Cookie banner diterima.")
            time.sleep(2)
        except:
            print("ℹ️ Tidak ada cookie banner atau sudah diterima.")

        # Ambil HTML dari tabel data pertandingan
        try:
            table_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "sched_2023-2024_9_1"))
            )
            html_source = table_element.get_attribute('outerHTML')
            print("✅ Berhasil mengambil tabel data pertandingan.")
            return html_source
        except Exception as e:
            print(f"❌ Gagal menemukan tabel pertandingan: {e}")
            return None
            
    finally:
        if driver:
            driver.quit()

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    PASSING_STATS_FILE = "premier_league_player_passing.csv"
    OUTPUT_FILE = "historical_matches.csv"

    # 1. Hitung rata-rata passing dari file yang sudah ada
    team_pass_avg_df = calculate_team_passing_avg(PASSING_STATS_FILE)
    if team_pass_avg_df is None:
        sys.exit()

    # 2. Scrape data historis pertandingan
    html_table = scrape_historical_matches()
    if html_table is None:
        sys.exit()

    # 3. Proses data hasil scrape
    print("⚙️ Memproses data pertandingan...")
    df_matches = pd.read_html(StringIO(html_table))[0]

    # Membersihkan data
    df_matches = df_matches[['Date', 'Home', 'Score', 'Away']]
    df_matches.dropna(subset=['Score'], inplace=True)
    df_matches = df_matches[df_matches['Score'].str.contains('–', na=False)]

    scores = df_matches['Score'].str.split('–', expand=True)
    df_matches['HomeGoals'] = pd.to_numeric(scores[0])
    df_matches['AwayGoals'] = pd.to_numeric(scores[1])
    
    print("🔄 Menggabungkan data pertandingan dengan data passing...")
    
    # Buat dictionary untuk mapping nama tim ke passing %
    pass_map = {row['Squad']: row['AvgPass%'] for index, row in team_pass_avg_df.iterrows()}
    
    def get_pass_perc(team_name):
        if team_name in pass_map:
            return pass_map[team_name]
        for squad_name, perc in pass_map.items():
            if team_name in squad_name or squad_name in team_name:
                return perc
        return team_pass_avg_df['AvgPass%'].mean()

    df_matches['HomePass%'] = df_matches['Home'].apply(get_pass_perc)
    df_matches['AwayPass%'] = df_matches['Away'].apply(get_pass_perc)
    
    # Finalisasi DataFrame
    final_df = df_matches[['Date', 'Home', 'Away', 'HomeGoals', 'AwayGoals', 'HomePass%', 'AwayPass%']]
    final_df = final_df.round(1)

    # 4. Simpan ke CSV
    try:
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n🎉 SUKSES! File '{OUTPUT_FILE}' berhasil dibuat dengan {len(final_df)} data pertandingan.")
        print("   Sekarang Anda bisa menjalankan script prediksi utama Anda.")
    except Exception as e:
        print(f"❌ Gagal menyimpan file CSV: {e}")