# predict-pl-match_otomatis_v3.py
# Versi v3 — peningkatan fitur + tuning ringan + no-data-leakage
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from collections import deque, defaultdict

# ML
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import joblib
import random
random.seed(42)
np.random.seed(42)

# ---------------------------
# 0. Config
# ---------------------------
CSV = "epl-training.csv"
TEST_FROM_DATE = "2023-01-01"   # split temporal (train < date, test >= date)
ROLL_WINDOW = 5                 # jumlah match untuk rolling form
RANDOM_SEARCH_ITER = 20         # iterasi RandomizedSearch untuk tuning (ringan)
N_JOBS = -1

# ---------------------------
# 1. Load data & basic checks
# ---------------------------
if not os.path.exists(CSV):
    raise FileNotFoundError(f"File '{CSV}' tidak ditemukan. Letakkan file di folder yg sama.")

df_raw = pd.read_csv(CSV, low_memory=False)
# Normalisasi kolom tanggal (coba beberapa format)
for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
    try:
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], format=fmt)
        break
    except Exception:
        pass
if df_raw['Date'].dtype == object:
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

df_raw = df_raw.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# columns we expect (some optional)
required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
optional_shot_cols = {
    'HomeShots': 'HS', 'AwayShots': 'AS',
    'HomeSOT': 'HST', 'AwaySOT': 'AST'
}
# rename standard
df = df_raw.copy()
df = df.rename(columns={
    'HomeTeam': 'Home', 'AwayTeam': 'Away',
    'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals',
    optional_shot_cols['HomeShots']: 'HomeShots' if optional_shot_cols['HomeShots'] in df.columns else None,
})

# fix optional renames carefully
for std_name, orig in optional_shot_cols.items():
    if orig in df.columns:
        if std_name == 'HomeShots':
            df = df.rename(columns={orig: 'HomeShots'})
        elif std_name == 'AwayShots':
            df = df.rename(columns={orig: 'AwayShots'})
        elif std_name == 'HomeSOT':
            df = df.rename(columns={orig: 'HomeSOT'})
        elif std_name == 'AwaySOT':
            df = df.rename(columns={orig: 'AwaySOT'})

# ensure required present
for c in ['Home','Away','HomeGoals','AwayGoals']:
    if c not in df.columns:
        raise KeyError(f"Kolom {c} tidak ditemukan di dataset. Pastikan file sesuai (Kaggle EPL).")

print(f"📊 Data dimuat: {df.shape[0]} baris ({df['Date'].min().date()} → {df['Date'].max().date()})")

# ---------------------------
# 2. Build features incrementally (no leakage)
# ---------------------------
print("🔧 Membangun fitur secara incremental (no leakage)...")

# histories stores past matches per team as deque of dicts
histories = defaultdict(lambda: deque(maxlen=ROLL_WINDOW))
# h2h_counts stores counts for ordered pair (home,away) -> dict hwins, awins, draws
h2h_counts = defaultdict(lambda: {'home_wins':0, 'away_wins':0, 'draws':0})

rows = []
for idx, row in df.iterrows():
    date = row['Date']
    home = row['Home']
    away = row['Away']
    hg = int(row['HomeGoals'])
    ag = int(row['AwayGoals'])

    # compute home last5 stats using histories (only previous matches)
    def stats_from_history(team):
        past = list(histories[team])  # most recent last entries (old -> new)
        if len(past) == 0:
            return {'avg_gf': np.nan, 'avg_ga': np.nan, 'win_rate': np.nan, 'avg_gd': np.nan,
                    'avg_shots': np.nan, 'avg_sot': np.nan}
        gf = np.array([m['gf'] for m in past])
        ga = np.array([m['ga'] for m in past])
        res = np.array([m['win'] for m in past])  # 1 win else 0
        out = {
            'avg_gf': gf.mean(),
            'avg_ga': ga.mean(),
            'win_rate': res.mean(),
            'avg_gd': (gf - ga).mean()
        }
        # optional shots
        if 'shots' in past[0]:
            out['avg_shots'] = np.mean([m.get('shots', np.nan) for m in past])
        else:
            out['avg_shots'] = np.nan
        if 'sot' in past[0]:
            out['avg_sot'] = np.mean([m.get('sot', np.nan) for m in past])
        else:
            out['avg_sot'] = np.nan
        return out

    home_stats = stats_from_history(home)
    away_stats = stats_from_history(away)

    # h2h counts BEFORE this match (key sorted to preserve pair)
    pair_key = (home, away)  # orientation matters; we'll store as ordered key
    # get aggregated previous results for this specific pairing (both orientations)
    # We'll compute counts from h2h_counts[(min, max)] but it's easier to compute by scanning earlier matches?
    # To avoid heavy scanning, we also maintain a dict of pair->counts where pair is (home,away) normalized (A,B)
    norm_key = tuple(sorted((home, away)))
    pair_hist = h2h_counts[norm_key]  # counts aggregated irrespective of venue
    # But need orientation: how many times home (current 'home') won vs away won vs draws
    # We'll store aggregated as counts where home_wins means wins by the first element of norm_key
    # So to convert to current orientation:
    if norm_key[0] == home:
        # then pair_hist['home_wins'] refers to wins when norm_key[0] was home in those past matches
        # but we want wins by current 'home' regardless of which side in norm_key
        h_wins = pair_hist['home_wins']
        a_wins = pair_hist['away_wins']
    else:
        # norm_key[0] is away, so swap
        h_wins = pair_hist['away_wins']
        a_wins = pair_hist['home_wins']
    draws = pair_hist['draws']

    # Build feature row (use simple features)
    feat = {
        'Date': date,
        'Home': home, 'Away': away,
        # home features
        'home_avg_gf': home_stats['avg_gf'], 'home_avg_ga': home_stats['avg_ga'],
        'home_win_rate': home_stats['win_rate'], 'home_avg_gd': home_stats['avg_gd'],
        'home_avg_shots': home_stats['avg_shots'], 'home_avg_sot': home_stats['avg_sot'],
        # away features
        'away_avg_gf': away_stats['avg_gf'], 'away_avg_ga': away_stats['avg_ga'],
        'away_win_rate': away_stats['win_rate'], 'away_avg_gd': away_stats['avg_gd'],
        'away_avg_shots': away_stats['avg_shots'], 'away_avg_sot': away_stats['avg_sot'],
        # h2h
        'h2h_home_wins': h_wins, 'h2h_away_wins': a_wins, 'h2h_draws': draws,
        # target
        'HomeGoals': hg, 'AwayGoals': ag
    }
    rows.append(feat)

    # AFTER constructing features for this match, update histories & h2h_counts with this match
    # update home history entry (perspective of team)
    home_entry = {'gf': hg, 'ga': ag, 'win': 1 if hg>ag else 0, 'shots': row.get('HomeShots', np.nan), 'sot': row.get('HomeSOT', np.nan)}
    away_entry = {'gf': ag, 'ga': hg, 'win': 1 if ag>hg else 0, 'shots': row.get('AwayShots', np.nan), 'sot': row.get('AwaySOT', np.nan)}
    histories[home].append(home_entry)
    histories[away].append(away_entry)

    # update h2h aggregated counts at norm_key
    # determine result
    if hg > ag:
        # home (current home) won -> increment for the team who was home in this match orientation of norm_key
        if norm_key[0] == home:
            h2h_counts[norm_key]['home_wins'] += 1
        else:
            h2h_counts[norm_key]['away_wins'] += 1
    elif ag > hg:
        if norm_key[0] == home:
            h2h_counts[norm_key]['away_wins'] += 1
        else:
            h2h_counts[norm_key]['home_wins'] += 1
    else:
        h2h_counts[norm_key]['draws'] += 1

features_df = pd.DataFrame(rows)
# target label
features_df['Result'] = np.select(
    [features_df['HomeGoals'] > features_df['AwayGoals'],
     features_df['HomeGoals'] < features_df['AwayGoals']],
    ['Home','Away'], default='Draw'
)

# drop rows where features missing (teams with no previous history)
before = features_df.shape[0]
features_df = features_df.dropna(subset=[
    'home_avg_gf','home_avg_ga','home_win_rate','home_avg_gd',
    'away_avg_gf','away_avg_ga','away_win_rate','away_avg_gd'
]).reset_index(drop=True)
after = features_df.shape[0]
print(f"  -> Baris awal: {before}, setelah drop missing features: {after}")

# ---------------------------
# 3. Prepare X, y and temporal split
# ---------------------------
# Decide feature columns (exclude raw goals)
feat_cols = [
    'home_avg_gf','home_avg_ga','home_win_rate','home_avg_gd',
    'away_avg_gf','away_avg_ga','away_win_rate','away_avg_gd',
    'h2h_home_wins','h2h_away_wins','h2h_draws'
]
# optionally include shots if available (check columns)
if 'home_avg_shots' in features_df.columns and features_df['home_avg_shots'].notna().any():
    feat_cols += ['home_avg_shots','away_avg_shots']
if 'home_avg_sot' in features_df.columns and features_df['home_avg_sot'].notna().any():
    feat_cols += ['home_avg_sot','away_avg_sot']

df_model = features_df.copy()
df_model = df_model.sort_values('Date').reset_index(drop=True)

train_df = df_model[df_model['Date'] < pd.to_datetime(TEST_FROM_DATE)]
test_df = df_model[df_model['Date'] >= pd.to_datetime(TEST_FROM_DATE)]
print(f"📅 Train rows: {train_df.shape[0]} | Test rows: {test_df.shape[0]}")

X_train = train_df[feat_cols].astype(float)
X_test = test_df[feat_cols].astype(float)
y_train = train_df['Result'].astype(str)
y_test = test_df['Result'].astype(str)

# encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
print("🔁 LabelEncoder classes:", le.classes_)

# ---------------------------
# 4. Hyperparameter tuning (RandomizedSearchCV with TimeSeriesSplit)
# ---------------------------
print("⚙️ Randomized hyperparameter search (light)...")
base = XGBClassifier(use_label_encoder=False, objective='multi:softprob', eval_metric='mlogloss', random_state=42)

param_dist = {
    'n_estimators': [100,200,300,400],
    'max_depth': [3,4,5,6],
    'learning_rate': [0.01,0.03,0.05,0.1],
    'subsample': [0.6,0.8,1.0],
    'colsample_bytree': [0.6,0.8,1.0],
    'reg_lambda': [0.5,1.0,1.5,2.0],
    'reg_alpha': [0.0,0.25,0.5,1.0]
}

tscv = TimeSeriesSplit(n_splits=4)
rnd = RandomizedSearchCV(
    estimator=base,
    param_distributions=param_dist,
    n_iter=RANDOM_SEARCH_ITER,
    scoring='balanced_accuracy',
    cv=tscv,
    verbose=1,
    n_jobs=N_JOBS,
    random_state=42
)
rnd.fit(X_train, y_train_enc)
print("✅ Best params:", rnd.best_params_)
best = rnd.best_estimator_

# ---------------------------
# 5. Final train on full training set, evaluate on test
# ---------------------------
print("🧠 Melatih model final dengan best params pada seluruh training set...")
best.fit(X_train, y_train_enc)  # no early stopping here (we used CV)
y_pred_enc = best.predict(X_test)
y_proba = best.predict_proba(X_test)
y_pred = le.inverse_transform(y_pred_enc)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
ll = log_loss(y_test_enc, y_proba)
print("\n📊 Evaluasi (hold-out test temporal):")
print(f"  - Accuracy      : {acc*100:.2f}%")
print(f"  - Balanced acc. : {bal_acc*100:.2f}%")
print(f"  - Log loss      : {ll:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix (rows true, cols pred):\n", confusion_matrix(y_test, y_pred))

# feature importance (XGBoost builtin)
fi = best.get_booster().get_score(importance_type='weight')
# map to readable list
fi_list = sorted(fi.items(), key=lambda x: x[1], reverse=True)
print("\nFeature importance (by weight):")
for f,v in fi_list:
    print(f"  {f}: {v}")

# optional: plot top features
try:
    names = [f for f in feat_cols]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    # use sklearn style importance via gain if available
    importances = []
    booster = best.get_booster()
    gain = booster.get_score(importance_type='gain')
    # map fscore names 'f0','f1' to feature names by index
    # XGBoost's feature names are 'f0'.. so map
    for i, name in enumerate(feat_cols):
        key = f"f{i}"
        importances.append(gain.get(key, 0.0))
    idx = np.argsort(importances)[::-1]
    top_n = min(10, len(feat_cols))
    plt.bar([feat_cols[i] for i in idx[:top_n]], [importances[i] for i in idx[:top_n]])
    plt.xticks(rotation=45, ha='right')
    plt.title("Feature importance (gain) - top features")
    plt.tight_layout()
    plt.show()
except Exception:
    pass

# ---------------------------
# 6. Save model & encoder
# ---------------------------
joblib.dump(best, "model_epl_v3.joblib")
joblib.dump(le, "labelencoder_v3.joblib")
print("\n💾 Model & encoder disimpan: model_epl_v3.joblib, labelencoder_v3.joblib")

# ---------------------------
# 7. Predict function using built incremental summaries (safe)
# ---------------------------
# Rebuild histories quickly up to last date (we already had histories built while feature building)
# But simpler: use last known aggregated stats in train+test (df_model tail)
last_summary = df_model.groupby('Home').tail(1)  # not perfect; we'll build helper compute again from original df
# We'll implement compute_team_form_from_raw to compute based on raw df (only past matches)
def compute_team_form_from_raw(team, date, raw_df, window=ROLL_WINDOW):
    past = raw_df[((raw_df['HomeTeam'] == team) | (raw_df['AwayTeam'] == team)) & (raw_df['Date'] < date)].tail(window)
    if past.empty:
        return None
    gf, ga, wins = [], [], []
    for _, r in past.iterrows():
        if r['HomeTeam'] == team:
            gf.append(int(r['FTHG'])); ga.append(int(r['FTAG'])
)
            wins.append(1 if int(r['FTHG'])>int(r['FTAG']) else 0)
        else:
            gf.append(int(r['FTAG'])); ga.append(int(r['FTHG'])
)
            wins.append(1 if int(r['FTAG'])>int(r['FTHG']) else 0)
    return {
        'avg_gf': np.mean(gf), 'avg_ga': np.mean(ga),
        'win_rate': np.mean(wins), 'avg_gd': np.mean(np.array(gf)-np.array(ga))
    }

def predict_match(home, away, model=best, le=le, raw_df=df_raw):
    date = raw_df['Date'].max() + pd.Timedelta(days=1)
    h = compute_team_form_from_raw(home, date, raw_df)
    a = compute_team_form_from_raw(away, date, raw_df)
    if h is None or a is None:
        print("⚠️ Tidak cukup history untuk salah satu tim.")
        return None
    # h2h
    nk = tuple(sorted((home, away)))
    ph = h2h_counts.get(nk, {'home_wins':0,'away_wins':0,'draws':0})
    # convert orientation
    if nk[0] == home:
        h_wins = ph['home_wins']; a_wins = ph['away_wins']
    else:
        h_wins = ph['away_wins']; a_wins = ph['home_wins']
    feats = [
        h['avg_gf'], h['avg_ga'], h['win_rate'], h['avg_gd'],
        a['avg_gf'], a['avg_ga'], a['win_rate'], a['avg_gd'],
        h_wins, a_wins, ph['draws']
    ]
    # if shots included, extend (not implemented in this helper)
    Xp = np.array(feats).reshape(1, -1)
    probs = model.predict_proba(Xp)[0]
    pred_enc = np.argmax(probs)
    pred_label = le.inverse_transform([pred_enc])[0]
    print(f"PREDIKSI: {home} vs {away} -> {pred_label}")
    for idx, cls in enumerate(le.classes_):
        print(f"  {cls}: {probs[idx]*100:.2f}%")
    return pred_label, probs

# contoh prediksi
print("\n🔮 Contoh prediksi:")
try:
    predict_match("Arsenal", "Man Utd")
    predict_match("Liverpool", "Man City")
    predict_match("Chelsea", "Tottenham")
except Exception as e:
    print("Info: prediksi contoh gagal:", e)

print("\nSelesai.")
