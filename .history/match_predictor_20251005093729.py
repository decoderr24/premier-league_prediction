# ===============================================
# 🔮 Premier League Match Outcome Predictor
# By decoder & GPT-5
# ===============================================

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys


# =====================================================
# 🧩 1. SCRAPE MATCH RESULTS
# =====================================================
def scrape_match_results():
    url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    print(f"🌍 Opening browser to scrape data from: {url}")
    time.sleep(2)

    options = ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--headless")  # mode tanpa tampilan browser
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        wait = WebDriverWait(driver, 20)
        table_div = wait.until(EC.presence_of_element_located((By.ID, "div_sched_9_2025_2026")))
        print("✅ Match schedule table found.")
        html_source = table_div.get_attribute("outerHTML")
        driver.quit()
        return pd.read_html(StringIO(html_source))[0]
    except TimeoutException:
        print("❌ Match table not found. Saving debug page...")
        with open("debug_match_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        driver.quit()
        return None


# =====================================================
# 🧮 2. CLEAN & FEATURE ENGINEERING
# =====================================================
def clean_match_data(df):
    print("🧹 Cleaning and preparing dataset...")

    # Hapus kolom tidak relevan
    df = df[['Date', 'Home', 'Score', 'Away', 'xG', 'xG.1', 'Poss', 'Poss.1', 'Sh', 'Sh.1']]

    # Hapus baris kosong atau belum dimainkan
    df = df.dropna(subset=['Score', 'xG', 'xG.1'])

    # Pisahkan skor menjadi dua kolom
    df[['Home_Goals', 'Away_Goals']] = df['Score'].str.split('–', expand=True)
    df['Home_Goals'] = pd.to_numeric(df['Home_Goals'], errors='coerce')
    df['Away_Goals'] = pd.to_numeric(df['Away_Goals'], errors='coerce')

    # Tentukan hasil pertandingan (1 = home win, 0 = draw, -1 = away win)
    df['Result'] = df.apply(
        lambda x: 1 if x['Home_Goals'] > x['Away_Goals'] else (-1 if x['Home_Goals'] < x['Away_Goals'] else 0),
        axis=1
    )

    # Buat fitur selisih
    df['xG_diff'] = df['xG'] - df['xG.1']
    df['Poss_diff'] = df['Poss'] - df['Poss.1']
    df['Shots_diff'] = df['Sh'] - df['Sh.1']

    df.to_csv("cleaned_match_data.csv", index=False)
    print("💾 Saved cleaned dataset as cleaned_match_data.csv")
    return df


# =====================================================
# 🤖 3. TRAIN MODEL
# =====================================================
def train_model(df):
    print("⚙️ Training Random Forest model...")

    X = df[['xG_diff', 'Poss_diff', 'Shots_diff']]
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n📊 Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    return model


# =====================================================
# 🔮 4. PREDICT NEW MATCH
# =====================================================
def predict_match(model, home_team, away_team):
    print(f"\n🔮 Predicting result: {home_team} vs {away_team}")

    # Sementara pakai nilai rata-rata historis (dummy)
    # (Kamu bisa nanti ganti dengan data real per tim)
    team_stats = {
        'Arsenal': {'xG': 2.1, 'Poss': 62, 'Sh': 15},
        'Man City': {'xG': 2.4, 'Poss': 68, 'Sh': 17},
        'Liverpool': {'xG': 2.3, 'Poss': 65, 'Sh': 16},
        'Chelsea': {'xG': 1.9, 'Poss': 59, 'Sh': 14},
        'Wolves': {'xG': 1.1, 'Poss': 43, 'Sh': 9},
        'Brighton': {'xG': 1.7, 'Poss': 57, 'Sh': 12},
        'Spurs': {'xG': 2.0, 'Poss': 61, 'Sh': 15},
        'Aston Villa': {'xG': 1.8, 'Poss': 55, 'Sh': 13},
    }

    if home_team not in team_stats or away_team not in team_stats:
        print("⚠️ Tim belum ada di dictionary, silakan tambahkan dulu datanya.")
        return

    home = team_stats[home_team]
    away = team_stats[away_team]

    match_df = pd.DataFrame([{
        'xG_diff': home['xG'] - away['xG'],
        'Poss_diff': home['Poss'] - away['Poss'],
        'Shots_diff': home['Sh'] - away['Sh']
    }])

    pred = model.predict(match_df)[0]
    if pred == 1:
        print(f"🏆 Prediksi: {home_team} MENANG")
    elif pred == 0:
        print(f"🤝 Prediksi: SERI")
    else:
        print(f"⚡ Prediksi: {away_team} MENANG")


# =====================================================
# 🚀 5. MAIN EXECUTION
# =====================================================
def main():
    # Step 1: Scrape data pertandingan
    df_raw = scrape_match_results()
    if df_raw is None:
        print("❌ Gagal mendapatkan data pertandingan.")
        return

    # Step 2: Bersihkan data
    df_clean = clean_match_data(df_raw)

    # Step 3: Train model
    model = train_model(df_clean)

    # Step 4: Prediksi
    predict_match(model, "Arsenal", "Wolves")
    predict_match(model, "Liverpool", "Man City")
    predict_match(model, "Brighton", "Chelsea")


if __name__ == "__main__":
    main()
