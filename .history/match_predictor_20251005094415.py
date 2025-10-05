import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# URL jadwal Premier League
URL = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"

print(f"🌍 Opening browser to scrape data from: {URL}")

try:
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
except Exception as e:
    print(f"❌ Gagal mengakses halaman FBref: {e}")
    print("❌ Gagal mendapatkan data pertandingan.")
    exit()

soup = BeautifulSoup(response.text, "html.parser")

# --- Coba cari tabel utama ---
table = soup.find("table", id=lambda x: x and "schedule" in x.lower())  # pencarian fleksibel
if table is None:
    table = soup.find("table", class_=lambda x: x and "stats_table" in x.lower())

if table is None:
    print("⚠️ Tidak menemukan tabel dengan ID 'schedule'. Mencoba pencarian adaptif...")

    # cari tabel berdasarkan heading
    possible_tables = soup.find_all("table")
    for t in possible_tables:
        if "Scores & Fixtures" in t.get_text():
            table = t
            break

if table is None:
    with open("debug_schedule.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    print("❌ Match table not found. Saved debug_schedule.html for inspection.")
    exit()

# --- Parsing isi tabel ---
rows = table.find_all("tr")
data = []
for row in rows:
    cols = [c.text.strip() for c in row.find_all(["th", "td"])]
    if len(cols) > 6:  # baris valid
        data.append(cols[:8])  # ambil kolom pertama
df = pd.DataFrame(data, columns=["Date", "Time", "Home", "xG_H", "Score", "xG_A", "Away", "Attendance"])
print(f"✅ Match table loaded: {len(df)} rows")

# --- Filter pertandingan Brighton vs Wolves ---
mask1 = (df["Home"].str.contains("Brighton", na=False) & df["Away"].str.contains("Wolves", na=False))
mask2 = (df["Home"].str.contains("Wolves", na=False) & df["Away"].str.contains("Brighton", na=False))
match_df = df[mask1 | mask2]

if match_df.empty:
    print("⚠️ Tidak ditemukan pertandingan Brighton vs Wolves.")
else:
    print("\n📅 Pertandingan Brighton vs Wolves ditemukan:")
    print(match_df.tail(1)[["Date", "Home", "Away", "Score"]].to_string(index=False))

# --- Prediksi probabilitas sederhana ---
brighton_form = np.random.uniform(0.55, 0.75)  # performa
wolves_form = np.random.uniform(0.35, 0.55)

p_brighton = brighton_form / (brighton_form + wolves_form)
p_wolves = 1 - p_brighton
p_draw = 0.15

print("\n🔮 Prediksi probabilitas:")
print(f"  ⚫ Brighton menang: {p_brighton:.2%}")
print(f"  🟠 Wolves menang:   {p_wolves:.2%}")
print(f"  ⚪ Seri:             {p_draw:.2%}")

# --- Prediksi skor ---
if p_brighton > 0.6:
    score_pred = "2 - 1"
elif p_brighton > 0.5:
    score_pred = "1 - 0"
else:
    score_pred = "1 - 1"

print(f"\n📊 Prediksi Skor: Brighton {score_pred} Wolves")
print("✅ Analisis selesai.")
