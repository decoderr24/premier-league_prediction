import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_fixtures_table(url):
    print(f"🌍 Opening browser to scrape data from: {url}")
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code != 200:
        print("❌ Gagal mengakses halaman FBref.")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Coba cari table dengan ID dinamis (berubah setiap musim)
    table = None
    for div in soup.find_all('div', id=True):
        if 'sched' in div['id'] and 'Premier-League' not in div['id']:
            if div.find('table'):
                table = div.find('table')
                print(f"✅ Ditemukan tabel jadwal dengan id: {div['id']}")
                break

    # Jika tidak ditemukan sama sekali
    if table is None:
        print("❌ Match table not found. Saving debug page...")
        with open("debug_schedule.html", "w", encoding="utf-8") as f:
            f.write(soup.prettify())
        return None

    # Parse ke pandas
    df = pd.read_html(str(table))[0]
    print(f"✅ Table ditemukan dengan shape: {df.shape}")
    return df


def predict_match(df, team1, team2):
    print(f"\n🔮 Memprediksi hasil: {team1} vs {team2}")

    # Filter 5 pertandingan terakhir masing-masing
    df = df.dropna(subset=['Home', 'Away'])
    team1_recent = df[(df['Home'] == team1) | (df['Away'] == team1)].tail(5)
    team2_recent = df[(df['Home'] == team2) | (df['Away'] == team2)].tail(5)

    def get_stats(team_df, team_name):
        win = 0
        lose = 0
        draw = 0
        goals_scored = 0
        goals_conceded = 0
        for _, row in team_df.iterrows():
            if pd.isna(row['Score']):
                continue
            try:
                home_goals, away_goals = map(int, row['Score'].split('–'))
            except:
                continue
            if row['Home'] == team_name:
                goals_scored += home_goals
                goals_conceded += away_goals
                if home_goals > away_goals:
                    win += 1
                elif home_goals < away_goals:
                    lose += 1
                else:
                    draw += 1
            else:
                goals_scored += away_goals
                goals_conceded += home_goals
                if away_goals > home_goals:
                    win += 1
                elif away_goals < home_goals:
                    lose += 1
                else:
                    draw += 1
        total = win + lose + draw if (win + lose + draw) > 0 else 1
        return {
            "win_rate": win / total,
            "avg_goals": goals_scored / total,
            "avg_conceded": goals_conceded / total
        }

    team1_stats = get_stats(team1_recent, team1)
    team2_stats = get_stats(team2_recent, team2)

    # Hitung skor prediksi sederhana
    team1_strength = (team1_stats["win_rate"] * 0.6 +
                      team1_stats["avg_goals"] * 0.3 -
                      team1_stats["avg_conceded"] * 0.1)
    team2_strength = (team2_stats["win_rate"] * 0.6 +
                      team2_stats["avg_goals"] * 0.3 -
                      team2_stats["avg_conceded"] * 0.1)

    prob_team1 = round((team1_strength / (team1_strength + team2_strength)) * 100, 1)
    prob_team2 = round(100 - prob_team1, 1)

    print(f"\n📊 Probabilitas Prediksi:")
    print(f"{team1} menang: {prob_team1}%")
    print(f"{team2} menang: {prob_team2}%")

    if prob_team1 > 55:
        result = f"🏆 Prediksi: {team1} Menang"
    elif prob_team2 > 55:
        result = f"🏆 Prediksi: {team2} Menang"
    else:
        result = "🤝 Prediksi: Seri"

    print(f"\n{result}\n")
    return result


if __name__ == "__main__":
    url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    df = get_fixtures_table(url)
    if df is not None:
        predict_match(df, "Brighton", "Wolves")
    else:
        print("❌ Gagal mendapatkan data pertandingan.")
