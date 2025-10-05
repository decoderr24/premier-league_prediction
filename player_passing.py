import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load data hasil scraping ---
df = pd.read_csv("premier_league_player_passing.csv")

print("✅ Data loaded successfully!")
print(df.head())

# --- 2. Bersihkan kolom angka (hapus simbol %, ubah ke float) ---
def clean_numeric(x):
    if isinstance(x, str):
        x = x.replace('%', '').replace(',', '').strip()
    return pd.to_numeric(x, errors='coerce')

df['Total_Cmp%'] = df['Total_Cmp%'].apply(clean_numeric)
df['Total_Att'] = df['Total_Att'].apply(clean_numeric)
df['Total_Cmp'] = df['Total_Cmp'].apply(clean_numeric)

# --- 3. 10 pemain dengan akurasi passing tertinggi ---
top_accuracy = df.sort_values('Total_Cmp%', ascending=False).head(10)[['Player', 'Squad', 'Total_Cmp%']]
print("\n🎯 Top 10 Players by Passing Accuracy:")
print(top_accuracy)

# --- 4. 10 pemain dengan jumlah umpan terbanyak ---
top_volume = df.sort_values('Total_Att', ascending=False).head(10)[['Player', 'Squad', 'Total_Att']]
print("\n📈 Top 10 Players by Total Passes Attempted:")
print(top_volume)

# --- 5. Rata-rata akurasi passing per klub ---
club_avg = df.groupby('Squad')['Total_Cmp%'].mean().sort_values(ascending=False).reset_index()
print("\n🏟️ Average Passing Accuracy per Team:")
print(club_avg.head(10))

# --- 6. Simpan hasil analisis ---
output_df = {
    'Top 10 Accuracy': top_accuracy,
    'Top 10 Volume': top_volume,
    'Team Average Accuracy': club_avg
}
with pd.ExcelWriter("premier_league_passing_analysis.xlsx") as writer:
    top_accuracy.to_excel(writer, sheet_name='Top_Accuracy', index=False)
    top_volume.to_excel(writer, sheet_name='Top_Volume', index=False)
    club_avg.to_excel(writer, sheet_name='Team_Average', index=False)

print("\n💾 Analysis results saved to premier_league_passing_analysis.xlsx")

# --- 7. Visualisasi: rata-rata akurasi passing antar tim ---
plt.figure(figsize=(10,6))
top10_clubs = club_avg.head(10)
plt.barh(top10_clubs['Squad'], top10_clubs['Total_Cmp%'], color='skyblue')
plt.xlabel("Average Passing Accuracy (%)")
plt.ylabel("Club")
plt.title("Top 10 Premier League Teams by Passing Accuracy (2025/2026)")
plt.gca().invert_yaxis()  # Biar ranking teratas di atas
plt.tight_layout()
plt.savefig("top10_passing_accuracy.png", dpi=300)
plt.show()

print("📊 Visualization saved as top10_passing_accuracy.png")
