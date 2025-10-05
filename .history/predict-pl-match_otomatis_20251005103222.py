import pandas as pd
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------------
# Fungsi scrape team stats
# ---------------------------
def scrape_team_stats():
    url = "https://fbref.com/en/comps/9/passing/Premier-League-Stats"
    print(f"🌍 Scraping team stats from {url} ...")
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    
    try:
        wait = WebDriverWait(driver, 15)
        div_element = wait.until(EC.presence_of_element_located((By.ID, "div_stats_passing_team")))
        html = div_element.get_attribute("outerHTML")
        df = pd.read_html(html)[0]
        
        # Gabungkan header jika multiindex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        # Pilih kolom penting
        cols = [c for c in df.columns if any(x in c for x in ['Squad', 'Cmp', 'Att', 'Cmp%', 'TotDist'])]
        df = df[cols]
        
        # Normalisasi nama kolom
        rename_map = {}
        for c in df.columns:
            if 'Squad' in c: rename_map[c] = 'Squad'
            elif 'Cmp%' in c: rename_map[c] = 'Total_Cmp%'
            elif 'Cmp' in c and 'Cmp%' not in c: rename_map[c] = 'Total_Cmp'
            elif 'Att' in c: rename_map[c] = 'Total_Att'
            elif 'TotDist' in c: rename_map[c] = 'Total_TotDist'
        df.rename(columns=rename_map, inplace=True)
        
        # Bersihkan
        df = df[df['Squad'].notna()]
        df = df[~df['Squad'].str.contains("Squad|Rk", na=False)]
        
        print(f"✅ Team stats loaded: {df.shape}")
        driver.quit()
        return df
    except TimeoutException:
        print("❌ Failed to load team stats, saving debug file...")
        driver.save_screenshot('debug_team_stats.png')
        with open('debug_team_stats.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        driver.quit()
        return None

# ---------------------------
# Fungsi prediksi skor
# ---------------------------
def predict_match(home_team, away_team, df_stats):
    print(f"\n🔮 Predicting: {home_team} vs {away_team}")
    
    def get_team_pass(df, team_name):
        row = df[df['Squad'].str.contains(team_name, case=False, na=False)]
        if row.empty: return 0.8  # default
        try:
            return row['Total_Cmp%'].astype(float).mean() / 100
        except:
            return 0.8
    
    home_pass = get_team_pass(df_stats, home_team)
    away_pass = get_team_pass(df_stats, away_team)
    
    # Contoh data form dan home advantage (dapat disesuaikan)
    team_form = {
        home_team: {"form": 0.6, "goals": 1.7, "concede": 1.2, "home_adv": 1.1},
        away_team: {"form": 0.55, "goals": 1.5, "concede": 1.3, "home_adv": 1.0},
    }
    
    # Skor probabilitas
    home_score = (team_form[home_team]["form"] * 0.5 +
                  team_form[home_team]["goals"] / (team_form[away_team]["concede"] + 0.1) * 0.3 +
                  home_pass * 0.2)
    away_score = (team_form[away_team]["form"] * 0.5 +
                  team_form[away_team]["goals"] / (team_form[home_team]["concede"] + 0.1) * 0.3 +
                  away_pass * 0.2)
    
    p_home = home_score / (home_score + away_score)
    p_away = away_score / (home_score + away_score)
    p_draw = 0.15
    
    # Prediksi skor realistis
    exp_home_goals = round(team_form[home_team]["goals"] * p_home)
    exp_away_goals = round(team_form[away_team]["goals"] * p_away)
    
    print("\n📊 Probabilities:")
    print(f"  🏠 {home_team} win: {p_home:.2%}")
    print(f"  🚗 {away_team} win: {p_away:.2%}")
    print(f"  🤝 Draw: {p_draw:.2%}")
    print(f"\n⚽ Predicted Score: {home_team} {exp_home_goals} - {exp_away_goals} {away_team}")
    
# ---------------------------
# MAIN
# ---------------------------
def main():
    df_stats = scrape_team_stats()
    if df_stats is None:
        print("❌ Cannot continue, missing data")
        return
    
    # Masukkan tim yang ingin diprediksi
    home_team = "Brighton"
    away_team = "Wolves"
    
    predict_match(home_team, away_team, df_stats)

if __name__ == "__main__":
    main()
