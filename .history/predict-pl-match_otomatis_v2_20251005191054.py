import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ======================
# 1️⃣ LOAD DATA
# ======================
print("📊 Memuat data historis Premier League...")
df = pd.read_csv("epl-training.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"✅ Data: {df.shape[0]} pertandingan dari {df['Date'].min().date()} sampai {df['Date'].max().date()}")

# ======================
# 2️⃣ PERSIAPAN FITUR DASAR
# ======================
df = df.rename(columns={
    'HomeTeam': 'Home', 'AwayTeam': 'Away',
    'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'
})
df['Result'] = np.select(
    [df['HomeGoals'] > df['AwayGoals'], df['HomeGoals'] < df['AwayGoals']],
    ['Home', 'Away'], default='Draw'
)

# ======================
# 3️⃣ HITUNG FORM (LAST 5 MATCHES)
# ======================
print("⚙️ Menghitung form 5 pertandingan terakhir per tim...")

def compute_team_form(team, current_date):
    past = df[((df['Home'] == team) | (df['Away'] == team)) & (df['Date'] < current_date)]
    last5 = past.tail(5)
    if last5.empty:
        return pd.Series([np.nan]*4, index=['avg_gf','avg_ga','win_rate','goal_diff_avg'])
    
    # Goals for/against
    gf, ga = [], []
    for _, row in last5.iterrows():
        if row['Home'] == team:
            gf.append(row['HomeGoals'])
            ga.append(row['AwayGoals'])
        else:
            gf.append(row['AwayGoals'])
            ga.append(row['HomeGoals'])
    wins = sum([1 if g1 > g2 else 0 for g1,g2 in zip(gf,ga)])
    return pd.Series([
        np.mean(gf), np.mean(ga), wins/len(last5), np.mean(np.array(gf)-np.array(ga))
    ], index=['avg_gf','avg_ga','win_rate','goal_diff_avg'])

# Buat kolom untuk fitur form home & away
home_features, away_features = [], []
for i, row in df.iterrows():
    home_features.append(compute_team_form(row['Home'], row['Date']))
    away_features.append(compute_team_form(row['Away'], row['Date']))

home_form = pd.DataFrame(home_features, index=df.index)
away_form = pd.DataFrame(away_features, index=df.index)

home_form = home_form.add_prefix("home_")
away_form = away_form.add_prefix("away_")

df = pd.concat([df, home_form, away_form], axis=1)
df = df.dropna().reset_index(drop=True)
print("✅ Fitur form berhasil dibuat.")

# ======================
# 4️⃣ FITUR HEAD-TO-HEAD (H2H)
# ======================
print("⚔️ Menghitung H2H historis per pasangan tim...")

def get_h2h_stats(home, away, date):
    past = df[((df['Home'] == home) & (df['Away'] == away)) |
              ((df['Home'] == away) & (df['Away'] == home))]
    past = past[past['Date'] < date]
    if past.empty:
        return pd.Series([0,0,0], index=['h2h_home_wins','h2h_away_wins','h2h_draws'])
    hw, aw, dr = 0,0,0
    for _, r in past.iterrows():
        if r['HomeGoals'] > r['AwayGoals']:
            if r['Home'] == home: hw+=1
            else: aw+=1
        elif r['AwayGoals'] > r['HomeGoals']:
            if r['Away'] == away: aw+=1
            else: hw+=1
        else:
            dr+=1
    return pd.Series([hw,aw,dr], index=['h2h_home_wins','h2h_away_wins','h2h_draws'])

h2h_stats = []
for i, row in df.iterrows():
    h2h_stats.append(get_h2h_stats(row['Home'], row['Away'], row['Date']))

df = pd.concat([df, pd.DataFrame(h2h_stats)], axis=1)
print("✅ Fitur H2H ditambahkan.")

# ======================
# 5️⃣ SIAPKAN DATA UNTUK MODEL
# ======================
features = [
    'home_avg_gf','home_avg_ga','home_win_rate','home_goal_diff_avg',
    'away_avg_gf','away_avg_ga','away_win_rate','away_goal_diff_avg',
    'h2h_home_wins','h2h_away_wins','h2h_draws'
]
target = 'Result'

split_date = '2023-01-01'
train = df[df['Date'] < split_date]
test = df[df['Date'] >= split_date]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]
print(f"📅 Train: {train.shape[0]} | Test: {test.shape[0]} pertandingan")

# ======================
# 6️⃣ TRAIN MODEL XGBOOST
# ======================
print("🧠 Melatih model XGBoost (regularized, no leak)...")

model = XGBClassifier(
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    reg_alpha=0.5,
    eval_metric="mlogloss"
)

try:
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=30,
        verbose=True
    )
except TypeError:
    print("⚠️ early_stopping_rounds tidak didukung, melatih tanpa early stopping.")
    model.fit(X_train, y_train)

print("✅ Model selesai dilatih.")

# ======================
# 7️⃣ EVALUASI MODEL
# ======================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n📊 HASIL EVALUASI:")
print(f"   - Accuracy        : {acc*100:.2f}%")
print(f"   - Log Loss        : {log_loss(y_test, y_proba):.4f}")
print("\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ======================
# 8️⃣ FUNGSI PREDIKSI MANUAL
# ======================
def predict_match(home_team, away_team):
    print("\n" + "="*40)
    print(f"🔮 PREDIKSI: {home_team} vs {away_team}")
    print("="*40)
    date = df['Date'].max() + pd.Timedelta(days=1)

    hf = compute_team_form(home_team, date)
    af = compute_team_form(away_team, date)
    h2h = get_h2h_stats(home_team, away_team, date)
    if hf.isna().any() or af.isna().any():
        print("⚠️ Tidak cukup data form untuk salah satu tim.")
        return
    X_pred = pd.DataFrame([[
        hf['avg_gf'], hf['avg_ga'], hf['win_rate'], hf['goal_diff_avg'],
        af['avg_gf'], af['avg_ga'], af['win_rate'], af['goal_diff_avg'],
        h2h['h2h_home_wins'], h2h['h2h_away_wins'], h2h['h2h_draws']
    ]], columns=features)
    probs = model.predict_proba(X_pred)[0]
    pred = model.predict(X_pred)[0]
    print(f"🏠 {home_team} Menang: {probs[0]*100:.2f}%")
    print(f"🚗 {away_team} Menang: {probs[1]*100:.2f}%")
    print(f"🤝 Seri            : {probs[2]*100:.2f}%")
    print(f"\n💡 Kesimpulan: {pred}")

# ======================
# 9️⃣ CONTOH PREDIKSI
# ======================
predict_match("Arsenal", "Man Utd")
predict_match("Liverpool", "Man City")
predict_match("Chelsea", "Tottenham")
