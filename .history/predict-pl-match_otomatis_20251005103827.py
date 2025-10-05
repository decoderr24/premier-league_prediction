import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ================================
# 1️⃣ Load Data
# ================================
print("📊 Loading passing stats...")
try:
    passing_df = pd.read_csv("premier_league_player_passing.csv")
    print(f"✅ Loaded team stats: {passing_df.shape[0]} players")
except FileNotFoundError:
    print("❌ premier_league_player_passing.csv not found")
    exit()

# Contoh data historis untuk model (harus ada CSV historical_matches.csv)
# Columns: Date, Home, Away, HomeGoals, AwayGoals, HomePass%, AwayPass%
print("📊 Loading historical match data...")
try:
    hist_df = pd.read_csv("historical_matches.csv")
    print(f"✅ Historical matches loaded: {hist_df.shape[0]} rows")
except FileNotFoundError:
    print("❌ historical_matches.csv not found, cannot train model")
    exit()

# ================================
# 2️⃣ Preprocessing
# ================================
def get_result(row):
    if row["HomeGoals"] > row["AwayGoals"]:
        return 1  # Home Win
    elif row["HomeGoals"] < row["AwayGoals"]:
        return 2  # Away Win
    else:
        return 0  # Draw

hist_df["Result"] = hist_df.apply(get_result, axis=1)
X = hist_df[["HomePass%", "AwayPass%"]]
y = hist_df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 3️⃣ Train Model
# ================================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ================================
# 4️⃣ Evaluate Model
# ================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# 5️⃣ Predict Next Match
# ================================
def predict_match(home_team, away_team, passing_df):
    print(f"\n🔮 Predicting match: {home_team} vs {away_team}")

    def avg_pass(team_name):
        team_df = passing_df[passing_df["Squad"].str.contains(team_name, case=False, na=False)]
        if team_df.empty:
            return 0.8  # default
        try:
            return team_df["Total_Cmp%"].astype(float).mean() / 100
        except:
            return 0.8

    home_pass = avg_pass(home_team)
    away_pass = avg_pass(away_team)

    # Prediksi probabilitas dengan model
    probs = model.predict_proba([[home_pass, away_pass]])[0]

    p_home = probs[1]
    p_away = probs[2]
    p_draw = probs[0]

    print("\n📈 Probabilities:")
    print(f"  🏠 {home_team} win : {p_home*100:.2f}%")
    print(f"  🚗 {away_team} win : {p_away*100:.2f}%")
    print(f"  🤝 Draw           : {p_draw*100:.2f}%")

    # Skor perkiraan (proporsional)
    exp_home_goals = round(np.random.poisson(1.5) * p_home * 2)
    exp_away_goals = round(np.random.poisson(1.5) * p_away * 2)

    print(f"\n⚽ Predicted Score: {home_team} {exp_home_goals} – {exp_away_goals} {away_team}")

    if p_home > max(p_draw, p_away):
        result = f"{home_team} likely to win"
    elif p_away > max(p_home, p_draw):
        result = f"{away_team} likely to win"
    else:
        result = "Draw likely"

    print(f"\n🧠 Final Analysis: {result}")
    print("✅ Prediction complete.\n")

# ================================
# 6️⃣ Run Example
# ================================
predict_match("Brighton", "Wolves")
predict_match("Brentford", "Man City")
