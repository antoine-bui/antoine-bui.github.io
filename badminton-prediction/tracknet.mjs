// TrackNetV3 worker (MIT, github.com/qaz812345/TrackNetV3, see models/TrackNetV3-LICENSE.txt).
// Input: a background + 8 consecutive frames, 512x288 RGB in [0, 1]. Output: 8 heatmaps; the shuttle is the
// largest blob above 0.5. Runs on the GPU through WebGPU (~50 ms per 8 frames on an RTX 4070 Ti).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.bundle.min.mjs';
import { findBlobs } from './logic.mjs';

const W = 512, H = 288, N = W * H, SEQ = 8;
// The exported model takes 32 channels instead of 27 (5 zero channels with zero weights, same maths):
// onnxruntime-web's WebGPU conv returned wrong heatmaps with 27 input channels.
const CH = 32;

const canvas = new OffscreenCanvas(W, H), c2d = canvas.getContext('2d', { willReadFrequently: true });
let session = null, vw = 0, vh = 0, seq = [], queue = [], running = false, failures = 0;
const input = new Float32Array(CH * N); // reused for every batch (19 MB); channels 27-31 stay zero
let prev = null, base = 0, cutNext = false; // cut detection: frame change vs its recent normal level

onmessage = ({ data }) => {
  if (data.type === 'config') { ({ vw, vh } = data); seq = []; queue = []; prev = null; base = 0; }
  else if (data.type === 'reset') { seq = []; queue = []; prev = null; base = 0; }
  else if (data.type === 'frame') addFrame(data.bitmap, data.t);
};

const createSession = () => ort.InferenceSession.create(new URL('./models/tracknet.onnx', import.meta.url).href, {
  executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
});
try {
  session = await createSession();
  postMessage({ type: 'ready' });
} catch (e) {
  postMessage({ type: 'error', msg: e.message });
}

function addFrame(bitmap, t) {
  c2d.drawImage(bitmap, 0, 0, W, H);
  bitmap.close();
  const px = c2d.getImageData(0, 0, W, H).data, f = new Float32Array(3 * N);
  for (let i = 0, j = 0; i < N; i++, j += 4) { f[i] = px[j] / 255; f[N + i] = px[j + 1] / 255; f[2 * N + i] = px[j + 2] / 255; }
  // Cut in an edited video (often between two points): the image changes several times more than usual.
  // The sequence restarts (TrackNet must not mix two scenes) and the frame is flagged for the rally logic.
  if (prev) {
    let d = 0, n = 0;
    for (let i = 0; i < 3 * N; i += 61, n++) d += Math.abs(f[i] - prev[i]);
    d = d / n * 255;
    if (d > 6 && d > 4 * base) { seq = []; cutNext = true; } else base = base ? base * 0.9 + d * 0.1 : d;
  }
  prev = f;
  seq.push({ t, f, cut: cutNext });
  cutNext = false;
  if (seq.length < SEQ) return;
  queue.push(seq); seq = [];
  // ~2.7 s of backlog fits in the 2 s lead (GPU warm-up, catching up after a seek); beyond, the GPU is too slow for
  // real time: drop the oldest batch rather than lag forever
  if (queue.length > 10) queue.shift();
  run();
}

async function run() {
  if (running || !session) return;
  running = true;
  try {
    while (queue.length) {
      const batch = queue.shift();
      // Background = per-pixel median of the 8 frames themselves: the moving shuttle drops out, and unlike a
      // long-term median it stays right with a moving camera or after a cut (measured as good as a 15 s median).
      input.set(medianOf(batch.map((fr) => fr.f)), 0);
      batch.forEach((fr, k) => input.set(fr.f, 3 * N * (k + 1)));
      const out = await session.run({ frames: new ort.Tensor('float32', input, [1, CH, H, W]) });
      const heat = out.heatmaps.data;
      postMessage({ type: 'points', points: batch.map((fr, k) => ({ t: fr.t, p: locate(heat, k), cut: fr.cut })) });
      out.heatmaps.dispose();
    }
  } catch (e) {
    // WebGPU can fail mid-video (seen: "Invalid Buffer" after a while, e.g. GPU memory pressure or a lost device):
    // rebuild the session and carry on; after 3 failures give up so the page falls back to the simple detector.
    console.warn('TrackNet: inference failed, restarting the session', e);
    queue = []; seq = [];
    try { await session.release(); } catch { /* already broken */ }
    session = null;
    if (++failures <= 3) { try { session = await createSession(); } catch { /* handled below */ } }
    if (!session) postMessage({ type: 'error', msg: e.message });
  } finally {
    running = false;
  }
}

function medianOf(frames) {
  const k = frames.length, v = new Float32Array(k), m = new Float32Array(3 * N);
  for (let i = 0; i < 3 * N; i++) {
    for (let s = 0; s < k; s++) {
      let x = frames[s][i], j = s;
      while (j > 0 && v[j - 1] > x) { v[j] = v[j - 1]; j--; } // insertion sort of 8 values
      v[j] = x;
    }
    m[i] = v[k >> 1];
  }
  return m;
}

// Largest blob of the k-th heatmap above 0.5, in video pixels.
function locate(heat, k) {
  const mask = new Uint8Array(N);
  for (let i = 0, o = k * N; i < N; i++) mask[i] = heat[o + i] > 0.5 ? 1 : 0;
  const blobs = findBlobs(mask, W, H, 1, Infinity);
  if (!blobs.length) return null;
  const b = blobs.reduce((a, c) => (c.area > a.area ? c : a));
  return [b.x * vw / W, b.y * vh / H];
}
