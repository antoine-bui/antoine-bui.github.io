"""Build a stroke dataset from ShuttleSet22 (CoachAI, MIT): court metres, hitter on the near side, 18 shot types.
Raw per-set CSVs are in image pixels; each match's homography + the court corners in the image give metres exactly."""
import os, ast
import numpy as np, pandas as pd

# usage: python build_dataset.py <dir with ShuttleSet22 set/*.csv + homography.csv, flattened as listed in index.tsv> <out.csv>
# (from github.com/wywyWang/CoachAI-Projects, CoachAI-Challenge-IJCAI2023/ShuttleSet22/set; index.tsv: 'file<TAB>original path')
import sys
D = sys.argv[1]
ZH = {  # ShuttleSet translation table (README)
    '放小球': 'net shot', '擋小球': 'return net', '殺球': 'smash', '點扣': 'wrist smash', '挑球': 'lob',
    '防守回挑': 'defensive return lob', '長球': 'clear', '平球': 'drive', '小平球': 'driven flight',
    '後場抽平球': 'back-court drive', '切球': 'drop', '過渡切球': 'passive drop', '推球': 'push', '撲球': 'rush',
    '防守回抽': 'defensive return drive', '勾球': 'cross-court net shot', '發短球': 'short service', '發長球': 'long service',
    '過度切球': 'passive drop',  # variant spelling; '未知球種' (unknown) is dropped
}
W, L, NET, FPS = 6.1, 13.4, 6.7, 30.0

idx = pd.read_csv(f'{D}/index.tsv', sep='\t', header=None, names=['file', 'path'])
files = dict(idx.values)
homo = pd.read_csv(f"{D}/{[f for f, p in files.items() if p.endswith('/homography.csv')][0]}")

def to_metres(Hm, corners):
    """image px -> court metres (x: left->right 0..6.1, y: far->near 0..13.4) through the template the homography maps to.
    The corner columns' left/right labels are not reliable (downleft_x > downright_x in the file): sort by image position."""
    top, bottom = sorted(corners, key=lambda c: c[1])[:2], sorted(corners, key=lambda c: c[1])[2:]
    (tl, tr), (bl, br) = sorted(top), sorted(bottom)
    tl, tr, bl, br = [Hm @ np.array([x, y, 1.0]) for x, y in (tl, tr, bl, br)]
    tl, tr, bl, br = [p[:2] / p[2] for p in (tl, tr, bl, br)]
    x0, x1 = (tl[0] + bl[0]) / 2, (tr[0] + br[0]) / 2
    y0, y1 = (tl[1] + tr[1]) / 2, (bl[1] + br[1]) / 2
    def f(x, y):
        p = Hm @ np.vstack([x, y, np.ones_like(x)])
        return (p[0] / p[2] - x0) / (x1 - x0) * W, (p[1] / p[2] - y0) / (y1 - y0) * L
    return f

rows = []
for f, p in files.items():
    if not p.split('/')[-1].startswith('set'):
        continue
    video = p.split('/')[-2]
    h = homo[homo.video == video]
    if h.empty:
        print('no homography for', video); continue
    h = h.iloc[0]
    Hm = np.array(ast.literal_eval(h.homography_matrix))
    # corner columns: up = far baseline, down = near baseline (broadcast camera behind the near baseline)
    conv = to_metres(Hm, [(h.upleft_x, h.upleft_y), (h.upright_x, h.upright_y), (h.downleft_x, h.downleft_y), (h.downright_x, h.downright_y)])
    r = pd.read_csv(f'{D}/{f}')
    r['match'], r['set'] = video, int(p.split('/set')[-1].split('.')[0])
    for a, b, n in [('player_location_x', 'player_location_y', 'h'), ('opponent_location_x', 'opponent_location_y', 'o'), ('landing_x', 'landing_y', 'l')]:
        r[n + 'x'], r[n + 'y'] = conv(r[a].to_numpy(float), r[b].to_numpy(float))
    rows.append(r)
df = pd.concat(rows, ignore_index=True)
df['type18'] = df.type.map(ZH)
print('strokes:', len(df), '| unknown type labels:', df.type18.isna().sum(), df.loc[df.type18.isna(), 'type'].value_counts().head(5).to_dict())

df = df.sort_values(['match', 'set', 'rally', 'ball_round']).reset_index(drop=True)
g = df.groupby(['match', 'set', 'rally'])
df['T'] = (g.frame_num.shift(-1) - df.frame_num) / FPS
df['prev'] = g.type18.shift(1).fillna('none')
df['backhand'] = df.backhand.fillna(0).astype(int)
df['aroundhead'] = df.aroundhead.fillna(0).astype(int)
# hitter always on the near side (y > net): rotate the court 180 deg otherwise (keeps the player's own left/right)
far = df.hy < NET
for x, y in [('hx', 'hy'), ('ox', 'oy'), ('lx', 'ly')]:
    df.loc[far, x], df.loc[far, y] = W - df.loc[far, x], L - df.loc[far, y]
df = df.dropna(subset=['type18', 'hx', 'hy', 'ox', 'oy', 'lx', 'ly'])
out = sys.argv[2]
df.to_csv(out, index=False)
print('kept:', len(df), 'strokes,', df.match.nunique(), 'matches ->', out)
print(df.type18.value_counts().to_string())
for c in ['hx', 'hy', 'ox', 'oy', 'lx', 'ly', 'T']:
    print(c, df[c].quantile([.02, .5, .98]).round(2).tolist())
# sanity: serves should land past the short service line (1.98 m), long serves near the back
for t in ['short service', 'long service']:
    s = df[df.type18 == t]; print(t, 'landing y (m)', s.ly.quantile([.1, .5, .9]).round(2).tolist())
