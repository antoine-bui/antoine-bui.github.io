"""Export the retrained ShuttleNet (encoder + decoder, fixed length) to ONNX for the browser, check it against PyTorch.
usage: python export_forecast.py <ShuttleNet src dir> <onnx tools dir or ''> <shuttlenet.pt> <rallies.npz> <out dir>"""
import sys, os, json
SRC, LIB, PT, DATA, OUT = sys.argv[1:6]
if LIB: sys.path.insert(0, LIB)
sys.path.insert(0, SRC)
import numpy as np, torch, torch.nn as nn
import onnxruntime as ort
from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor

ck = torch.load(PT, map_location='cpu', weights_only=False)  # our own training output
cfg = ck['cfg']
enc, dec = ShotGenEncoder(cfg), ShotGenPredictor(cfg)
enc.load_state_dict(ck['encoder']); dec.load_state_dict(ck['decoder']); enc.eval(); dec.eval()
E, Lfull = cfg['encode_length'] - 1, cfg['max_ball_round']

class Forecast(nn.Module):
    """inputs [1, L]: shot (int), x, y (float), player (int), target = next hitter (int); outputs per decoder step:
    xy = (mean x, mean y, log sx, log sy, atanh corr), probs over [padding] + classes."""
    def __init__(self):
        super().__init__()
        self.enc, self.dec = enc, dec  # registered submodules: their weights become ONNX initializers
    def forward(self, shot, x, y, player, target):
        el, gA, gB = self.enc(shot[:, :E], x[:, :E], y[:, :E], player[:, :E])
        xy, logits = self.dec(shot[:, E:], x[:, E:], y[:, E:], player[:, E:], el, gA, gB, target[:, E:])
        return xy, logits.softmax(-1)

d = np.load(DATA, allow_pickle=True)
def example(i):  # a real rally, padded like the browser does
    t, x, y, p = d['types'][i], d['x'][i], d['y'][i], d['player'][i]
    n = min(len(t), Lfull)
    a = [np.zeros((1, Lfull), np.int64), np.zeros((1, Lfull), np.float32), np.zeros((1, Lfull), np.float32), np.zeros((1, Lfull), np.int64), np.zeros((1, Lfull), np.int64)]
    a[0][0, :n], a[1][0, :n], a[2][0, :n], a[3][0, :n] = t[:n], x[:n], y[:n], p[:n]
    a[4][0, :n] = np.append(p[1:n + 1], 3 - p[n - 1])[:n]
    return a, n
model = Forecast().eval()
args, _ = example(0)
os.makedirs(OUT, exist_ok=True)
onnx_path = os.path.join(OUT, 'forecast.onnx')
torch.onnx.export(model, tuple(torch.from_numpy(a) for a in args), onnx_path, input_names=['shot', 'x', 'y', 'player', 'target'],
                  output_names=['xy', 'probs'], opset_version=17, dynamo=False)
model.eval()  # export restores the wrapper's (train) mode, which would switch dropout back on in the reference below
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
worst = 0
for i in range(0, len(d['types']), 97):
    a, n = example(i)
    with torch.no_grad():
        ref = [v.numpy() for v in model(*[torch.from_numpy(v) for v in a])]
    got = sess.run(None, dict(zip(['shot', 'x', 'y', 'player', 'target'], a)))
    rows = slice(0, n - E)  # valid decoder steps
    worst = max(worst, max(float(np.abs(g[0, rows] - r[0, rows]).max()) for g, r in zip(got, ref)))
print(f'{onnx_path}: {os.path.getsize(onnx_path) / 1e3:.0f} kB, max |onnx - torch| on real rallies = {worst:.2e}')
json.dump({'about': 'ShuttleNet (CoachAI, AAAI 2022, MIT, github.com/wywyWang/CoachAI-Projects) retrained player-agnostic on '
           'ShuttleSet22 for one-step next-stroke forecasting. Test = 6 unseen matches.',
           'classes': ck['classes'], 'length': Lfull, 'encodeLength': E, 'report': ck['report']},
          open(os.path.join(OUT, 'forecast.json'), 'w'), indent=1)
