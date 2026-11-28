// Next-stroke forecast worker: ShuttleNet (CoachAI, AAAI 2022, MIT), retrained player-agnostic on ShuttleSet22
// (tools/train_forecast.py) and exported to ONNX (tools/export_forecast.py). Tiny model: runs on the CPU (WASM).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.bundle.min.mjs';
import { COURT, forecastInputs } from './logic.mjs';

let session = null, meta = null, running = false;
const queue = [];
onmessage = ({ data }) => { queue.push(data); run(); };

try {
  meta = await (await fetch(new URL('./models/forecast.json', import.meta.url))).json();
  session = await ort.InferenceSession.create(new URL('./models/forecast.onnx', import.meta.url).href, { executionProviders: ['wasm'] });
  postMessage({ type: 'ready', report: meta.report });
} catch (e) {
  postMessage({ type: 'error', msg: e.message });
}
run(); // forecasts asked while the model was loading

async function run() {
  if (running || !session) return;
  running = true;
  while (queue.length) {
    const { key, shots } = queue.shift();
    const L = meta.length, E = meta.encodeLength, inp = forecastInputs(shots, meta.classes, L);
    if (inp.n < E + 1 || shots.length > L) continue; // needs the serve and the return, and fits the model's length
    const i64 = (a) => new ort.Tensor('int64', BigInt64Array.from(a.map(BigInt)), [1, L]);
    const f32 = (a) => new ort.Tensor('float32', Float32Array.from(a), [1, L]);
    const out = await session.run({ shot: i64(inp.shot), x: f32(inp.x), y: f32(inp.y), player: i64(inp.player), target: i64(inp.target) });
    const row = inp.n - 1 - E; // decoder output at the last observed stroke = the next one
    const P = out.probs.data, K = meta.classes.length + 1, xy = out.xy.data.slice(row * 5, row * 5 + 5);
    const probs = meta.classes.map((c, k) => ({ type: c, p: P[row * K + k + 1] })).sort((a, b) => b.p - a.p);
    let mean = [xy[0] * 1.5 + COURT.w / 2, xy[1] * 3.5 + COURT.net];
    if (inp.flip) mean = [COURT.w - mean[0], COURT.l - mean[1]]; // back to the app's court (a 180 deg turn keeps the spread)
    postMessage({ type: 'forecast', key, top: probs.slice(0, 3), mean, sx: Math.exp(xy[2]) * 1.5, sy: Math.exp(xy[3]) * 3.5, corr: Math.tanh(xy[4]) });
  }
  running = false;
}
