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
    url = "https://fbref.com/en/comps/9/schedule/2023-2024/Premier-League-Scores-and-Fixtures"
    print(f"🌐 Mengakses halaman: {url}")

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

    driver = None
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver.get(url)

        try:
            wait = WebDriverWait(driver, 5)
            accept_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[text()="Accept All"]')))
            accept_button.click()
            print("✅ Cookie banner diterima.")
            time.sleep(2)
        except:
            print("ℹ️ Tidak ada cookie banner atau sudah diterima.")

        # === PERUBAHAN DI SINI ===
        # Beri waktu 3 detik agar semua JavaScript selesai dimuat
        print("⏳ Memberi waktu ekstra agar halaman memuat sempurna...")
        time.sleep(3)
        
        # Ambil HTML dari tabel yang berisi data pertandingan
        try:
            # Menunggu hingga tabel benar-benar TERLIHAT, bukan hanya ada di DOM
            # Menggunakan selector class yang lebih umum: 'stats_table'
            table_element = WebDriverWait(driver, 15).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "table.stats_table"))
            )
            html_source = table_element.get_attribute('outerHTML')
            print("✅ Berhasil menemukan dan mengambil tabel data pertandingan.")
            return html_source
        except Exception as e:
            print(f"❌ Gagal menemukan tabel pertandingan setelah menunggu: {e}")
            return None
            
    finally:
        if driver:
            driver.quit()

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    # Ganti nama file sesuai dengan nama file Anda jika berbeda
    PASSING_STATS_FILE = "premier_league_player_passing.csv"
    OUTPUT_FILE = "historical_matches.csv"

    team_pass_avg_df = calculate_team_passing_avg(PASSING_STATS_FILE)
    if team_pass_avg_df is None:
        sys.exit()

    html_table = scrape_historical_matches()
    if html_table is None:
        sys.exit()

    print("⚙️ Memproses data pertandingan...")
    df_matches = pd.read_html(StringIO(html_table))[0]
    df_matches = df_matches[['Date', 'Home', 'Score', 'Away']]
    df_matches.dropna(subset=['Score'], inplace=True)
    df_matches = df_matches[df_matches['Score'].str.contains('–', na=False)]

    scores = df_matches['Score'].str.split('–', expand=True)
    df_matches['HomeGoals'] = pd.to_numeric(scores[0])
    df_matches['AwayGoals'] = pd.to_numeric(scores[1])
    
    print("🔄 Menggabungkan data pertandingan dengan data passing...")
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
    
    final_df = df_matches[['Date', 'Home', 'Away', 'HomeGoals', 'AwayGoals', 'HomePass%', 'AwayPass%']]
    final_df = final_df.round(1)

    try:
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n🎉 SUKSES! File '{OUTPUT_FILE}' berhasil dibuat dengan {len(final_df)} data pertandingan.")
        print("   Sekarang Anda bisa menjalankan script prediksi utama Anda.")
    except Exception as e:
        print(f"❌ Gagal menyimpan file CSV: {e}")