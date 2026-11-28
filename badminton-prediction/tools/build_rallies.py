"""Next-stroke forecast (ShuttleNet) - how the model in models/forecast.* is rebuilt:
  1. git clone https://github.com/wywyWang/CoachAI-Projects  (MIT)
     ShuttleNet source: "CoachAI-Challenge-IJCAI2023/Track 2: Stroke Forecasting/src" (ShuttleNet/ folder);
     in ShuttleNet/ShuttleNet.py replace `merged_seq.cuda(seq_A.device)` by `merged_seq.to(seq_A.device)` (CPU).
     Raw strokes: CoachAI-Challenge-IJCAI2023/ShuttleSet22/set (+ index.tsv 'file<TAB>original path' if flattened).
  2. python build_rallies.py <set dir> rallies.npz                       (numpy, pandas)
  3. python train_forecast.py <src dir> rallies.npz shuttlenet.pt         (torch; ~45 epochs on a CPU, early stopping)
  4. python export_forecast.py <src dir> '' shuttlenet.pt rallies.npz ../models   (torch, onnx, onnxruntime)
"""
"""Rally sequences for next-stroke forecasting (ShuttleNet), from ShuttleSet22 (CoachAI, MIT).
Each rally: shot types (ShuttleSet 18 types, driven flight merged into drive), landing points in court metres
(absolute frame, turned 180 deg when the server stands on the far side, so the server is always on the near side),
and the hitter as a role token: 1 = server, 2 = receiver (player-agnostic, unlike the original 35 player ids).
usage: python build_rallies.py <set dir (index.tsv + flattened CSVs)> <out.npz>"""
import sys, ast
import numpy as np, pandas as pd

D, OUT = sys.argv[1], sys.argv[2]
ZH = {'放小球': 'net shot', '擋小球': 'return net', '殺球': 'smash', '點扣': 'wrist smash', '挑球': 'lob',
      '防守回挑': 'defensive return lob', '長球': 'clear', '平球': 'drive', '小平球': 'drive', '後場抽平球': 'back-court drive',
      '切球': 'drop', '過渡切球': 'passive drop', '過度切球': 'passive drop', '推球': 'push', '撲球': 'rush',
      '防守回抽': 'defensive return drive', '勾球': 'cross-court net shot', '發短球': 'short service', '發長球': 'long service'}
CLASSES = sorted(set(ZH.values()))
W, L, NET = 6.1, 13.4, 6.7

idx = pd.read_csv(f'{D}/index.tsv', sep='\t', header=None, names=['file', 'path'])
files = dict(idx.values)
homo = pd.read_csv(f"{D}/{[f for f, p in files.items() if p.endswith('/homography.csv')][0]}")

def to_metres(Hm, corners):
    top, bottom = sorted(corners, key=lambda c: c[1])[:2], sorted(corners, key=lambda c: c[1])[2:]
    (tl, tr), (bl, br) = sorted(top), sorted(bottom)
    tl, tr, bl, br = [Hm @ np.array([x, y, 1.0]) for x, y in (tl, tr, bl, br)]
    tl, tr, bl, br = [p[:2] / p[2] for p in (tl, tr, bl, br)]
    x0, x1, y0, y1 = (tl[0] + bl[0]) / 2, (tr[0] + br[0]) / 2, (tl[1] + tr[1]) / 2, (bl[1] + br[1]) / 2
    def f(x, y):
        p = Hm @ np.vstack([x, y, np.ones_like(x)])
        return (p[0] / p[2] - x0) / (x1 - x0) * W, (p[1] / p[2] - y0) / (y1 - y0) * L
    return f

rallies, matches = [], []
for f, p in files.items():
    if not p.split('/')[-1].startswith('set'):
        continue
    video = p.split('/')[-2]
    h = homo[homo.video == video].iloc[0]
    conv = to_metres(np.array(ast.literal_eval(h.homography_matrix)),
                     [(h.upleft_x, h.upleft_y), (h.upright_x, h.upright_y), (h.downleft_x, h.downleft_y), (h.downright_x, h.downright_y)])
    r = pd.read_csv(f'{D}/{f}')
    r['lx'], r['ly'] = conv(r.landing_x.to_numpy(float), r.landing_y.to_numpy(float))
    r['hx'], r['hy'] = conv(r.player_location_x.to_numpy(float), r.player_location_y.to_numpy(float))
    r['type18'] = r.type.map(ZH)
    for _, g in r.sort_values(['rally', 'ball_round']).groupby('rally'):
        if g.type18.isna().any() or g[['lx', 'ly', 'hy']].isna().any().any() or len(g) < 3:
            continue  # unknown type or missing position somewhere: skip the rally (sequences must be complete)
        server = g.player.iloc[0]
        lx, ly = g.lx.to_numpy(), g.ly.to_numpy()
        if g.hy.iloc[0] < NET:  # server on the far side: turn the court so the server is near
            lx, ly = W - lx, L - ly
        rallies.append((np.array([CLASSES.index(t) + 1 for t in g.type18]),   # 0 = padding
                        (lx - W / 2) / 1.5, (ly - NET) / 3.5,                  # normalised metres (~unit scale)
                        np.where(g.player.to_numpy() == server, 1, 2)))
        matches.append(video)

lens = np.array([len(r[0]) for r in rallies])
print(f'{len(rallies)} rallies, {len(set(matches))} matches, strokes per rally: median {np.median(lens):.0f}, 95% {np.percentile(lens, 95):.0f}, max {lens.max()}')
np.savez(OUT, types=np.array([r[0] for r in rallies], dtype=object), x=np.array([r[1] for r in rallies], dtype=object),
         y=np.array([r[2] for r in rallies], dtype=object), player=np.array([r[3] for r in rallies], dtype=object),
         match=np.array(matches), classes=np.array(CLASSES))
