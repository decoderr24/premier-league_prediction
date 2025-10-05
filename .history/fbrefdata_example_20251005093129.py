import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import re

# === 1. URL target (Premier League Passing Stats terbaru) ===
URL = "https://fbref.com/en/comps/9/passing/Premier-League-Stats"

print(f"📡 Mengambil data dari {URL} ...")

# === 2. Ambil HTML page ===
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
response = requests.get(URL, headers=headers)

if response.status_code != 200:
    raise Exception(f"Gagal mengunduh halaman (status code {response.status_code})")

html = response.text

# === 3. Tangani tabel yang tersembunyi dalam komentar HTML ===
soup = BeautifulSoup(html, "html.parser")

# FBref sering menyembunyikan tabel di dalam komentar <!-- ... -->
comments = soup.find_all(string=lambda text: isinstance(text, Comment))
passing_table_html = None

for c in comments:
    if 'table' in c and 'passing' in c:
        if 'id="stats_passing' in c:
            passing_table_html = c
            break

if not passing_table_html:
    raise Exception("❌ Tabel passing tidak ditemukan. Mungkin struktur halaman berubah.")

# === 4. Parse tabel dari komentar ===
passing_soup = BeautifulSoup(passing_table_html, "html.parser")
table = passing_soup.find("table")

if table is None:
    raise Exception("❌ Tidak bisa mem-parse tabel dari komentar HTML.")

# === 5. Konversi ke DataFrame ===
df = pd.read_html(str(table))[0]

# === 6. Bersihkan kolom ===
df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
df = df.dropna(how='all')  # hapus baris kosong

# === 7. Simpan ke CSV ===
csv_name = "premier_league_passing_2025.csv"
df.to_csv(csv_name, index=False)
print(f"✅ Data berhasil diunduh dan disimpan ke {csv_name}")

# === 8. Tampilkan preview ===
print("\n=== Preview Data ===")
print(df.head(10))
