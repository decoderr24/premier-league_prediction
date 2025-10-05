import requests
import pandas as pd
from io import StringIO

def pull_premier_league_team_passing():
    url = "https://fbref.com/en/comps/9/passing/Premier-League-Stats"
    print(f"Downloading team passing stats from {url} ...")

    # Gunakan headers yang lebih lengkap untuk meniru browser asli
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1'
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    df = pd.read_html(StringIO(response.text))[0]

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
    # Menampilkan kolom yang relevan dari DataFrame yang sudah difilter
    print(df_filtered[["Squad", "Total_Cmp", "Total_Att", "Total_Cmp%", "Total_TotDist"]])

# Bagian ini PENTING untuk menjalankan fungsi main()
if __name__ == "__main__":
    main()