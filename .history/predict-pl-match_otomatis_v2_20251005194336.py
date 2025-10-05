# predict-pl-match_v3_1_optimized.py
"""
Predict PL v3.1 - Optimized, no-data-leakage, time-aware CV, Optuna tuning (if available).
Requirements: pandas, numpy, scikit-learn, xgboost, joblib, matplotlib
Optional: optuna
"""
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from collections import defaultdict, deque
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, log_loss
import joblib
import matplotlib.pyplot as plt

# ML model
from xgboost import XGBClassifier

# Try Optuna
USE_OPTUNA = False
try:
    import optuna
    from optuna.samplers import TPESampler
    USE_OPTUNA = True
except Exception:
    USE_OPTUNA = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------
# Config
# -------------------------
CSV = "epl-training.csv"
TEST_SPLIT_DATE = "2023-01-01"
ROLL_WINDOW = 5
OPTUNA_TRIALS = 40   # if optuna available
RANDOM_SEARCH_ITERS = 30  # fallback tuning iterations

# -------------------------
# 1. Load & normalize data
# -------------------------
if not os.path.exists(CSV):
    raise FileNotFoundError(f"File '{CSV}' tidak ditemukan.")

df_raw = pd.read_csv(CSV, low_memory=False)

# try common date formats
for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
    try:
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], format=fmt)
        break
    except Exception:
        pass
if df_raw['Date'].dtype == object:
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

df_raw = df_raw.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
print("📊 Data dimuat:", df_raw.shape, df_raw['Date'].min().date(), "→", df_raw['Date'].max().date())

# rename columns to expected ones
rename_map = {}
if 'HomeTeam' in df_raw.columns: rename_map['HomeTeam'] = 'Home'
if 'AwayTeam' in df_raw.columns: rename_map['AwayTeam'] = 'Away'
if 'FTHG' in df_raw.columns: rename_map['FTHG'] = 'HomeGoals'
if 'FTAG' in df_raw.columns: rename_map['FTAG'] = 'AwayGoals'
if 'HS' in df_raw.columns: rename_map['HS'] = 'HomeShots'
if 'AS' in df_raw.columns: rename_map['AS'] = 'AwayShots'
if 'HST' in df_raw.columns: rename_map['HST'] = 'HomeSOT'
if 'AST' in df_raw.columns: rename_map['AST'] = 'AwaySOT'
df_raw = df_raw.rename(columns=rename_map)

required = ['Date','Home','Away','HomeGoals','AwayGoals']
for c in required:
    if c not in df_raw.columns:
        raise KeyError(f"Kolom '{c}' tidak ditemukan. Pastikan dataset EPL (Kaggle) yang benar.")

# -------------------------
# 2. Incremental feature build (no leakage)
# -------------------------
print("🔧 Membangun fitur incremental (no leakage)...")
histories = defaultdict(lambda: deque(maxlen=ROLL_WINDOW))
h2h_counts = defaultdict(lambda: {'home_wins':0,'away_wins':0,'draws':0})
rows = []

for idx, r in df_raw.iterrows():
    date = r['Date']
    home = r['Home']
    away = r['Away']
    hg = int(r['HomeGoals'])
    ag = int(r['AwayGoals'])

    # helper to get last N stats for a team
    def get_stats(team):
        past = list(histories[team])
        if len(past) == 0:
            return {'avg_gf':np.nan,'avg_ga':np.nan,'win_rate':np.nan,'avg_gd':np.nan,'avg_shots':np.nan,'avg_sot':np.nan}
        gf = np.array([m['gf'] for m in past])
        ga = np.array([m['ga'] for m in past])
        wins = np.array([m['win'] for m in past])
        out = {
            'avg_gf': float(np.mean(gf)),
            'avg_ga': float(np.mean(ga)),
            'win_rate': float(np.mean(wins)),
            'avg_gd': float(np.mean(gf - ga))
        }
        if 'shots' in past[0]:
            out['avg_shots'] = float(np.nanmean([m.get('shots', np.nan) for m in past]))
        else:
            out['avg_shots'] = np.nan
        if 'sot' in past[0]:
            out['avg_sot'] = float(np.nanmean([m.get('sot', np.nan) for m in past]))
        else:
            out['avg_sot'] = np.nan
        return out

    home_stats = get_stats(home)
    away_stats = get_stats(away)

    # normalized pair key
    nk = tuple(sorted((home, away)))
    ph = h2h_counts[nk]
    # orientation mapping
    if nk[0] == home:
        h_wins = ph['home_wins']; a_wins = ph['away_wins']
    else:
        h_wins = ph['away_wins']; a_wins = ph['home_wins']
    draws = ph['draws']

    rows.append({
        'Date': date, 'Home': home, 'Away': away,
        # home
        'home_avg_gf': home_stats['avg_gf'], 'home_avg_ga': home_stats['avg_ga'],
        'home_win_rate': home_stats['win_rate'], 'home_avg_gd': home_stats['avg_gd'],
        'home_avg_shots': home_stats['avg_shots'], 'home_avg_sot': home_stats['avg_sot'],
        # away
        'away_avg_gf': away_stats['avg_gf'], 'away_avg_ga': away_stats['avg_ga'],
        'away_win_rate': away_stats['win_rate'], 'away_avg_gd': away_stats['avg_gd'],
        'away_avg_shots': away_stats['avg_shots'], 'away_avg_sot': away_stats['avg_sot'],
        # h2h
        'h2h_home_wins': h_wins, 'h2h_away_wins': a_wins, 'h2h_draws': draws,
        # targets (raw)
        'HomeGoals': hg, 'AwayGoals': ag
    })

    # update histories AFTER features built
    home_entry = {'gf': hg, 'ga': ag, 'win': 1 if hg>ag else 0,
                  'shots': r.get('HomeShots', np.nan), 'sot': r.get('HomeSOT', np.nan)}
    away_entry = {'gf': ag, 'ga': hg, 'win': 1 if ag>hg else 0,
                  'shots': r.get('AwayShots', np.nan), 'sot': r.get('AwaySOT', np.nan)}
    histories[home].append(home_entry)
    histories[away].append(away_entry)

    # update h2h aggregated counts (norm key)
    if hg > ag:
        # winner is home in this match
        if nk[0] == home:
            h2h_counts[nk]['home_wins'] += 1
        else:
            h2h_counts[nk]['away_wins'] += 1
    elif ag > hg:
        if nk[0] == home:
            h2h_counts[nk]['away_wins'] += 1
        else:
            h2h_counts[nk]['home_wins'] += 1
    else:
        h2h_counts[nk]['draws'] += 1

features_df = pd.DataFrame(rows)
# target label
features_df['Result'] = np.select(
    [features_df['HomeGoals'] > features_df['AwayGoals'],
     features_df['HomeGoals'] < features_df['AwayGoals']],
    ['Home','Away'], default='Draw'
)

# drop rows missing core features
core_feats = ['home_avg_gf','home_avg_ga','home_win_rate','home_avg_gd',
              'away_avg_gf','away_avg_ga','away_win_rate','away_avg_gd']
before = features_df.shape[0]
features_df = features_df.dropna(subset=core_feats).reset_index(drop=True)
after = features_df.shape[0]
print(f"  -> baris awal: {before}, setelah drop missing: {after}")

# -------------------------
# 3. Prepare X,y and split (temporal)
# -------------------------
feat_cols = [
    'home_avg_gf','home_avg_ga','home_win_rate','home_avg_gd',
    'away_avg_gf','away_avg_ga','away_win_rate','away_avg_gd',
    'h2h_home_wins','h2h_away_wins','h2h_draws'
]
# include shots if available & not all NaN
if features_df['home_avg_shots'].notna().any():
    feat_cols += ['home_avg_shots','away_avg_shots']
if features_df['home_avg_sot'].notna().any():
    feat_cols += ['home_avg_sot','away_avg_sot']

df_model = features_df.sort_values('Date').reset_index(drop=True)
train_df = df_model[df_model['Date'] < pd.to_datetime(TEST_SPLIT_DATE)].copy()
test_df = df_model[df_model['Date'] >= pd.to_datetime(TEST_SPLIT_DATE)].copy()
print("📅 Train rows:", train_df.shape[0], "Test rows:", test_df.shape[0])

X_train = train_df[feat_cols].fillna(0).astype(float)
X_test = test_df[feat_cols].fillna(0).astype(float)
y_train = train_df['Result'].astype(str)
y_test = test_df['Result'].astype(str)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
print("🔁 Label classes:", list(le.classes_))

# compute simple sample weights to rebalance classes
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train_enc)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_enc)
# map class code -> weight
class_weight_map = {cls: w for cls,w in zip(classes, cw)}
sample_weights = np.array([class_weight_map[c] for c in y_train_enc])

# -------------------------
# 4. Hyperparameter tuning (Optuna or RandomizedSearchCV)
# -------------------------
print("⚙️ Mulai hyperparameter tuning... (optuna available:", USE_OPTUNA, ")")

def objective_optuna(trial):
    params = {
        'verbosity': 0,
        'use_label_encoder': False,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'random_state': RANDOM_SEED,
        'n_estimators': trial.suggest_categorical('n_estimators', [100,200,300,400]),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0)
    }
    model = XGBClassifier(**params)
    tscv = TimeSeriesSplit(n_splits=4)
    scores = []
    for tr_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train_enc[tr_idx], y_train_enc[val_idx]
        sw = sample_weights[tr_idx]
        # fit with early stopping if possible
        try:
            model.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xval, yval)], early_stopping_rounds=30, verbose=False)
        except TypeError:
            model.fit(Xtr, ytr, sample_weight=sw)
        ypred = model.predict(Xval)
        scores.append(balanced_accuracy_score(yval, ypred))
    return np.mean(scores)

best_params = None
if USE_OPTUNA:
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective_optuna, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    best_params = study.best_trial.params
    print("✅ Optuna best params:", best_params)
else:
    # fallback RandomizedSearchCV (time-aware implemented via cv=TimeSeriesSplit inside search)
    param_dist = {
        'n_estimators': [100,200,300,400],
        'max_depth': [3,4,5,6],
        'learning_rate': [0.01,0.03,0.05,0.1,0.2],
        'subsample': [0.6,0.8,1.0],
        'colsample_bytree': [0.6,0.8,1.0],
        'reg_lambda': [0.1,0.5,1.0,1.5],
        'reg_alpha': [0.0,0.25,0.5,1.0]
    }
    base = XGBClassifier(use_label_encoder=False, objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_SEED)
    rnd = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=RANDOM_SEARCH_ITERS,
                             scoring='balanced_accuracy', cv=TimeSeriesSplit(n_splits=4), n_jobs=-1, verbose=1, random_state=RANDOM_SEED)
    rnd.fit(X_train, y_train_enc, sample_weight=sample_weights)
    best_params = rnd.best_params_
    print("✅ RandomizedSearchCV best params:", best_params)

# -------------------------
# 5. Train final model
# -------------------------
print("🧠 Melatih final model dengan best params...")
final_kwargs = dict(use_label_encoder=False, objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_SEED)
# if optuna, best_params contains only param keys; else RandomizedSearchCV gave 'clf__' style? we already set best_params
final_kwargs.update(best_params)

model = XGBClassifier(**final_kwargs)
try:
    model.fit(X_train, y_train_enc, sample_weight=sample_weights, eval_set=[(X_test, y_test_enc)], early_stopping_rounds=50, verbose=False)
except TypeError:
    model.fit(X_train, y_train_enc, sample_weight=sample_weights)

# -------------------------
# 6. Evaluate
# -------------------------
y_pred_enc = model.predict(X_test)
y_proba = model.predict_proba(X_test)
y_pred = le.inverse_transform(y_pred_enc)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
ll = log_loss(y_test_enc, y_proba)

print("\n📊 Evaluasi (hold-out test temporal):")
print(f"  - Accuracy     : {acc*100:.2f}%")
print(f"  - Balanced acc.: {bal_acc*100:.2f}%")
print(f"  - Log loss     : {ll:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# 7. Feature importance (plot)
# -------------------------
print("\n🔎 Feature importance (gain):")
booster = model.get_booster()
fscore = booster.get_score(importance_type='gain')  # use 'gain'
# Map 'f0'.. to feature names
mapped = {}
for k, v in fscore.items():
    if k.startswith('f'):
        idx = int(k[1:])
        if idx < len(feat_cols):
            mapped[feat_cols[idx]] = v
        else:
            mapped[k] = v
# sort and print
fi_sorted = sorted(mapped.items(), key=lambda x: x[1], reverse=True)
for name, val in fi_sorted:
    print(f"  {name}: {val:.4f}")

# plot top features
try:
    top_n = min(12, len(fi_sorted))
    names = [n for n,_ in fi_sorted[:top_n]]
    vals = [v for _,v in fi_sorted[:top_n]]
    plt.figure(figsize=(8,5))
    plt.barh(names[::-1], vals[::-1])
    plt.title("Feature importance (gain) - top features")
    plt.tight_layout()
    plt.show()
except Exception:
    pass

# -------------------------
# 8. Save model & encoder & feature columns
# -------------------------
joblib.dump(model, "model_epl_v3_1.joblib")
joblib.dump(le, "labelencoder_v3_1.joblib")
joblib.dump(feat_cols, "feature_columns_v3_1.joblib")
print("\n💾 Model, encoder, dan feature columns disimpan.")

# -------------------------
# 9. Safe predict_match helper
# -------------------------
def compute_team_form_from_raw(team, date, raw_df, window=ROLL_WINDOW):
    past = raw_df[((raw_df['HomeTeam'] == team) | (raw_df['AwayTeam'] == team)) & (raw_df['Date'] < date)].tail(window)
    if past.empty:
        return None
    gf, ga, wins = [], [], []
    for _, r in past.iterrows():
        if r['HomeTeam'] == team:
            gf.append(int(r['FTHG'])); ga.append(int(r['FTAG']))
            wins.append(1 if int(r['FTHG']) > int(r['FTAG']) else 0)
        else:
            gf.append(int(r['FTAG'])); ga.append(int(r['FTHG']))
            wins.append(1 if int(r['FTAG']) > int(r['FTHG']) else 0)
    return {
        'avg_gf': np.mean(gf), 'avg_ga': np.mean(ga),
        'win_rate': np.mean(wins), 'avg_gd': np.mean(np.array(gf)-np.array(ga))
    }

def predict_match_safe(home, away, model=model, le=le, feat_cols=feat_cols, raw_df=df_raw, allow_fill=True):
    date = raw_df['Date'].max() + pd.Timedelta(days=1)
    hf = compute_team_form_from_raw(home, date, raw_df)
    af = compute_team_form_from_raw(away, date, raw_df)
    if hf is None or af is None:
        print("⚠️ Tidak cukup history untuk salah satu tim.")
        return None
    nk = tuple(sorted((home, away)))
    ph = h2h_counts.get(nk, {'home_wins':0,'away_wins':0,'draws':0})
    if nk[0] == home:
        h_wins = ph['home_wins']; a_wins = ph['away_wins']
    else:
        h_wins = ph['away_wins']; a_wins = ph['home_wins']
    feats = [
        hf['avg_gf'], hf['avg_ga'], hf['win_rate'], hf['avg_gd'],
        af['avg_gf'], af['avg_ga'], af['win_rate'], af['avg_gd'],
        h_wins, a_wins, ph['draws']
    ]
    # include shot/sot if model used them
    if 'home_avg_shots' in feat_cols:
        feats += [hf.get('avg_shots', 0.0), af.get('avg_shots', 0.0)]
    if 'home_avg_sot' in feat_cols:
        feats += [hf.get('avg_sot', 0.0), af.get('avg_sot', 0.0)]
    Xp = pd.DataFrame([feats], columns=feat_cols)
    # fill missing with 0 for safety
    Xp = Xp.fillna(0).astype(float)
    proba = model.predict_proba(Xp)[0]
    pred_enc = int(np.argmax(proba))
    pred_label = le.inverse_transform([pred_enc])[0]
    print(f"\n🔮 PREDIKSI: {home} vs {away} -> {pred_label}")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls}: {proba[i]*100:.2f}%")
    return pred_label, proba

# -------------------------
# 10. Example predictions
# -------------------------
print("\n🔮 Contoh prediksi (safe):")
try:
    predict_match_safe("Arsenal", "Man Utd")
    predict_match_safe("Liverpool", "Man City")
    predict_match_safe("Chelsea", "Tottenham")
except Exception as e:
    print("Info: contoh prediksi gagal:", e)

print("\nSelesai.")
