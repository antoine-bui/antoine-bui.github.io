"""Train ShuttleNet (CoachAI, AAAI 2022, MIT) for next-stroke forecasting in Badminton Vision.
Changes vs the challenge baseline: player-agnostic roles (1 = server, 2 = receiver), 17 ShuttleSet types,
encode_length 3 (the encoder reads the serve and the return - it needs one stroke of each player - so forecasts start\nfrom the 4th stroke), court-metre coordinates,
early stopping on held-out matches. Reports one-step-ahead accuracy on unseen matches vs a frequency baseline.
usage: python train_forecast.py <ShuttleNet src dir> <rallies.npz> <out.pt>"""
import sys, json, random
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, sys.argv[1])
from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor

def Gaussian2D_loss(V_pred, V_trgt):  # copied from ShuttleNet_runner.py (avoids its tqdm/matplotlib imports)
    normx, normy = V_trgt[:, 0] - V_pred[:, 0], V_trgt[:, 1] - V_pred[:, 1]
    sx, sy, corr = torch.exp(V_pred[:, 2]), torch.exp(V_pred[:, 3]), torch.tanh(V_pred[:, 4])
    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2
    result = torch.exp(-z / (2 * negRho)) / (2 * np.pi * (sxsy * torch.sqrt(negRho)))
    return torch.sum(-torch.log(torch.clamp(result, min=1e-20)))

SRC, DATA, OUT = sys.argv[1:4]
torch.manual_seed(42); random.seed(42); np.random.seed(42)
d = np.load(DATA, allow_pickle=True)
classes = list(d['classes'])
MAXR = 34  # strokes kept per rally (95 % of rallies are shorter); decoder length 32
cfg = dict(encode_length=3, max_ball_round=MAXR, shot_num=len(classes) + 1, player_num=3, shot_dim=32, area_dim=32,
           player_dim=32, encode_dim=32, area_num=5)

def pad(rally):
    t, x, y, p = rally
    n = min(len(t), MAXR + 1) - 1  # input i -> target i+1
    a = [np.zeros(MAXR, dtype=np.int64) for _ in range(2)] + [np.zeros(MAXR, dtype=np.float32) for _ in range(4)] + [np.zeros(MAXR, dtype=np.int64) for _ in range(2)]
    a[0][:n], a[1][:n], a[2][:n], a[3][:n] = t[:n], p[:n], x[:n], y[:n]                     # inputs: shot, player, x, y
    a[4][:n], a[5][:n], a[6][:n], a[7][:n] = x[1:n + 1], y[1:n + 1], t[1:n + 1], p[1:n + 1]  # targets: x, y, shot, player
    return a
rallies = list(zip(d['types'], d['x'], d['y'], d['player']))
match = d['match']
ms = sorted(set(match)); random.shuffle(ms)
test_m, val_m = set(ms[:6]), set(ms[6:11])
split = lambda S: [pad(r) for r, m in zip(rallies, match) if m in S]
train = split(set(ms[11:])); val = split(val_m); test = split(test_m)
print(f'rallies train {len(train)} / val {len(val)} / test {len(test)} (matches {len(ms) - 11}/{len(val_m)}/{len(test_m)})')
stack = lambda rows: [torch.from_numpy(np.stack([r[k] for r in rows])) for k in range(8)]

enc, dec = ShotGenEncoder(cfg), ShotGenPredictor(cfg)
enc.area_embedding.weight = dec.shotgen_decoder.area_embedding.weight  # shared embeddings, as in train.py
enc.shot_embedding.weight = dec.shotgen_decoder.shot_embedding.weight
enc.player_embedding.weight = dec.shotgen_decoder.player_embedding.weight
dec.player_embedding.weight = dec.shotgen_decoder.player_embedding.weight
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
ce = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
E = cfg['encode_length'] - 1

def forward(b):
    shot, player, x, y, tx, ty, tshot, tplayer = b
    el, gA, gB = enc(shot[:, :E], x[:, :E], y[:, :E], player[:, :E])
    xy, logits = dec(shot[:, E:], x[:, E:], y[:, E:], player[:, E:], el, gA, gB, tplayer[:, E:])
    mask = shot[:, E:] != 0
    return xy[mask], logits[mask], tshot[:, E:][mask], torch.stack([tx[:, E:][mask], ty[:, E:][mask]], -1)

def run(rows, train_mode):
    enc.train(train_mode); dec.train(train_mode)
    if train_mode: random.shuffle(rows)
    tot, n = 0.0, 0
    for i in range(0, len(rows), 32):
        xy, logits, ts, txy = forward(stack(rows[i:i + 32]))
        loss = ce(logits, ts) + Gaussian2D_loss(xy, txy)
        if train_mode: opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); n += len(ts)
    return tot / n

best, bad = 1e9, 0
for ep in range(80):
    tr = run(train, True)
    with torch.no_grad(): va = run(val, False)
    print(f'epoch {ep:2d} train {tr:.3f} val {va:.3f}', flush=True)
    if va < best - 1e-3: best, bad, state = va, 0, ({k: v.clone() for k, v in enc.state_dict().items()}, {k: v.clone() for k, v in dec.state_dict().items()})
    else:
        bad += 1
        if bad >= 8: break
enc.load_state_dict(state[0]); dec.load_state_dict(state[1])

# ---- one-step-ahead evaluation on unseen matches (exactly what the app does live) ----
enc.eval(); dec.eval()
with torch.no_grad():
    xy, logits, ts, txy = forward(stack(test))
prob = logits[:, 1:].softmax(-1)  # drop padding class
top3 = prob.topk(3, -1).indices + 1
acc1 = (top3[:, 0] == ts).float().mean().item(); acc3 = (top3 == ts[:, None]).any(-1).float().mean().item()
metres = lambda v: torch.stack([v[:, 0] * 1.5, v[:, 1] * 3.5], -1)
err = (metres(xy[:, :2]) - metres(txy)).norm(dim=-1).mean().item()
# baseline: most frequent next type after the current type, mean landing of that pair (train counts)
def pairs(rows):
    out = []
    for r in rows:
        n = int((r[0] != 0).sum())
        for i in range(E, n): out.append((r[0][i], r[6][i], r[4][i], r[5][i]))
    return out
from collections import Counter, defaultdict
cnt, pos = defaultdict(Counter), defaultdict(list)
for a, b, x_, y_ in pairs(train): cnt[a][b] += 1; pos[a].append((x_, y_))
tp = pairs(test)
b1 = np.mean([cnt[a].most_common(1)[0][0] == b if cnt[a] else 0 for a, b, _, _ in tp])
b3 = np.mean([b in [k for k, _ in cnt[a].most_common(3)] for a, b, _, _ in tp])
berr = np.mean([np.hypot((np.mean([p[0] for p in pos[a]]) - x_) * 1.5, (np.mean([p[1] for p in pos[a]]) - y_) * 3.5) for a, _, x_, y_ in tp])
report = dict(test_strokes=len(ts), shuttlenet=dict(top1=round(acc1, 3), top3=round(acc3, 3), landing_error_m=round(err, 2)),
              baseline=dict(top1=round(float(b1), 3), top3=round(float(b3), 3), landing_error_m=round(float(berr), 2)))
print(json.dumps(report, indent=1))
torch.save(dict(cfg=cfg, classes=classes, encoder=enc.state_dict(), decoder=dec.state_dict(), report=report), OUT)
