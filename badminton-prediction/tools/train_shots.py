"""Shot-type classifier on ShuttleSet22 strokes, using only what Badminton Vision measures from a video.
Evaluated with GroupKFold by match (no match is in both train and test)."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report

# usage: python train_shots.py <strokes.csv>                          -> match-wise CV of 3 models
#        python train_shots.py <strokes.csv> <out.json> <acc> <top2>  -> export the final MLP (models/shots.json)
df = pd.read_csv(sys.argv[1])
sys.argv = sys.argv[1:]
df['type18'] = df.type18.replace({'driven flight': 'drive'})  # 26 samples: merged
df['prev'] = df.prev.replace({'driven flight': 'drive'})
CLASSES = sorted(df.type18.unique())
PREV = ['none'] + CLASSES
NET = 6.7

def features(d):
    """Same inputs the app has at the end of a shot: positions (m, hitter near side), flight time, stroke number,
    previous shot type, backhand flag."""
    T = d['T'].fillna(-1)  # -1 = last stroke of the rally (no next hit)
    X = pd.DataFrame({
        'hx': d.hx, 'hy': d.hy, 'ox': d.ox, 'oy': d.oy, 'lx': d.lx, 'ly': d.ly,
        'hit_net': d.hy - NET, 'land_net': NET - d.ly, 'side': d.lx - d.hx,
        'dist': np.hypot(d.lx - d.hx, d.ly - d.hy), 'T': T, 'T_known': (T >= 0).astype(float),
        'round': d.ball_round.clip(upper=12), 'first': (d.ball_round == 1).astype(float),
        'backhand': d.backhand.astype(float),
    })
    for p in PREV:
        X['prev_' + p] = (d.prev == p).astype(float)
    return X

X, y, groups = features(df), df.type18.map(CLASSES.index).to_numpy(), df.match.to_numpy()

if len(sys.argv) > 1:  # export: final MLP on all strokes -> JSON for the browser (+ parity vectors for the JS test)
    out, (acc, top2) = sys.argv[1], (float(sys.argv[2]), float(sys.argv[3]))
    sc = StandardScaler().fit(X)
    m = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=400, early_stopping=True, random_state=0).fit(sc.transform(X), y)
    r = lambda a: np.round(np.asarray(a, float), 6).tolist()
    sample = X.sample(5, random_state=1)
    model = {
        'about': 'Shot-type MLP trained on ShuttleSet22 (CoachAI, MIT, github.com/wywyWang/CoachAI-Projects), '
                 f'{len(df)} strokes / {df.match.nunique()} pro matches. Match-wise 5-fold CV: accuracy {acc:.3f}, top-2 {top2:.3f}.',
        'classes': CLASSES, 'prev': PREV, 'features': list(X.columns),
        'mean': r(sc.mean_), 'scale': r(sc.scale_),
        'layers': [{'W': r(w), 'b': r(b)} for w, b in zip(m.coefs_, m.intercepts_)],
        'tests': [{'x': r(row), 'p': r(m.predict_proba(sc.transform(row.to_frame().T))[0])} for _, row in sample.iterrows()],
    }
    json.dump(model, open(out, 'w'), separators=(',', ':'))
    print('exported', out, os.path.getsize(out), 'bytes')
    sys.exit()
models = {
    'logreg': lambda: LogisticRegression(max_iter=2000, C=1.0),
    'hgb': lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=0),
    'mlp': lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=400, early_stopping=True, random_state=0),
}
res = {}
for name, mk in models.items():
    accs, top2, preds = [], [], np.zeros(len(y), int)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X.iloc[tr])
        m = mk().fit(sc.transform(X.iloc[tr]), y[tr])
        pr = m.predict_proba(sc.transform(X.iloc[te]))
        full = np.zeros((len(te), len(CLASSES))); full[:, m.classes_] = pr
        preds[te] = full.argmax(1)
        accs.append(accuracy_score(y[te], preds[te])); top2.append(top_k_accuracy_score(y[te], full, k=2, labels=range(len(CLASSES))))
    res[name] = preds
    print(f'{name:7s} accuracy {np.mean(accs):.3f} (+-{np.std(accs):.3f})  top-2 {np.mean(top2):.3f}', flush=True)

best = max(res, key=lambda k: accuracy_score(y, res[k]))
print('\nper-class report for', best)
print(classification_report(y, res[best], target_names=CLASSES, digits=2))
