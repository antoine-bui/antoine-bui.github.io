// Player worker: MediaPipe person detection + id tracking. Runs at its own pace (~10/s) so it never slows the shuttle tracker.
import { ObjectDetector, PoseLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { COURT, project } from './logic.mjs';

// MediaPipe loads its wasm glue with importScripts, which throws in module workers: replace it with sync XHR + global eval.
self.importScripts = (...urls) => urls.forEach((url) => {
  const x = new XMLHttpRequest();
  x.open('GET', String(url), false);
  x.send();
  (0, eval)(x.responseText);
});

let detector = null, pose = null, mpT = 0, H = null, vw = 0, perSide = 2, tracks = [], others = [];

// Handler first: messages sent while the model downloads must not be lost.
onmessage = ({ data }) => {
  if (data.type === 'config') { ({ H, vw, perSide } = data); tracks = []; others = []; }
  else if (data.type === 'pose') handVote(data);
  else if (data.type === 'frame') {
    try { if (H && detector) detect(data.bitmap); } finally { data.bitmap.close(); }
    postMessage({ type: 'players', players: tracks.filter((tr) => tr.miss < 5), others });
  }
};

try {
  const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
  detector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
      delegate: 'CPU', // GPU delegate fights the video decoder for the GPU (measured: 35% dropped frames vs 0% on CPU)
    },
    runningMode: 'VIDEO', scoreThreshold: 0.3, maxResults: 15, categoryAllowlist: ['person'],
  });
  postMessage({ type: 'ready' });
  // pose model for the racket-hand votes (optional: without it every player stays right-handed)
  pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task', delegate: 'CPU' },
    runningMode: 'IMAGE', numPoses: 1,
  }).catch(() => null);
} catch (e) {
  postMessage({ type: 'error', msg: e.message });
}

// Ids are tied to the side of the net: singles J1 = near side (camera), J2 = far side; doubles J1-J2 near, J3-J4 far.
// So a cut in the video or a missed detection can never turn J1 into J3, and extra people (umpire, coach) are dropped.
function detect(bitmap) {
  mpT = Math.max(mpT + 1, performance.now()); // MediaPipe wants strictly increasing timestamps
  const k = vw / bitmap.width;
  const all = detector.detectForVideo(bitmap, mpT).detections.map(({ boundingBox: b, categories }) => {
    const x = b.originX * k, y = b.originY * k, w = b.width * k, h = b.height * k, feet = [x + w / 2, y + h];
    return { x, y, w, h, feet, court: project(H, feet), score: categories[0].score };
  });
  // players lunge ~1 m past the side lines and step ~2 m behind the baseline
  const inPlay = all.filter(({ court: [cx, cy] }) => cx > -1 && cx < COURT.w + 1 && cy > -2 && cy < COURT.l + 2);
  const used = new Set();
  tracks.forEach((tr) => tr.miss++);
  for (const near of [true, false]) {
    const ids = Array.from({ length: perSide }, (_, i) => (near ? i : perSide + i));
    // players stand inside the lines, spectators/umpire outside: rank by distance outside the court, then by confidence
    const dets = inPlay.filter((d) => d.court[1] >= COURT.net === near)
      .sort((a, b) => outside(a.court) - outside(b.court) || b.score - a.score);
    // known players keep their id: nearest detection on their side (a just-seen player can't teleport, a lost one can)
    for (const tr of tracks.filter((t) => ids.includes(t.id)).sort((a, b) => a.miss - b.miss)) {
      let best = null, bd = tr.miss <= 3 ? vw * 0.2 : Infinity;
      for (const d of dets) if (!used.has(d)) { const dd = dist(tr.feet, d.feet); if (dd < bd) { bd = dd; best = d; } }
      // someone standing further inside the lines takes the slot back (a spectator grabbed it while the player was missed)
      const better = dets.find((d) => !used.has(d) && outside(d.court) < outside((best || tr).court));
      if (better) best = better;
      if (best) { used.add(best); Object.assign(tr, best, { miss: 0 }); }
    }
    // free slots of this side go to the most confident remaining detections
    for (const d of dets) {
      const id = ids.find((i) => !tracks.some((t) => t.id === i));
      if (id === undefined) break;
      if (!used.has(d)) { used.add(d); tracks.push({ ...d, id, miss: 0 }); }
    }
  }
  tracks = tracks.filter((tr) => tr.miss < 15);
  others = all.filter((d) => !used.has(d)).map(({ x, y, w, h }) => ({ x, y, w, h })); // spectators etc., masked for the shuttle search
}

// One racket-hand vote: full-res crop of the hitter at the contact frame, p = shuttle in crop pixels. Pose landmarks are
// anatomical (15 = left wrist, 16 = right wrist) whether the player faces the camera or not. The wrist nearest the
// shuttle holds the racket; no vote when both are at similar distances (blur, crossed arms).
function handVote({ bitmap, id, p }) {
  try {
    const lm = pose && pose.detect(bitmap).landmarks[0];
    if (!lm) return;
    const d = [15, 16].map((k) => Math.hypot(lm[k].x * bitmap.width - p[0], lm[k].y * bitmap.height - p[1]));
    if (Math.max(...d) < 1.5 * Math.min(...d)) return;
    postMessage({ type: 'hand', id, right: d[1] < d[0] });
  } finally { bitmap.close(); }
}

// function declarations (hoisted): frames are handled while the pose model is still loading further up
function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
// metres outside the court lines (0 inside), rounded to 0.5 m so close calls fall back to the detector's confidence
function outside([x, y]) { return Math.round(2 * (Math.max(0, -x, x - COURT.w) + Math.max(0, -y, y - COURT.l))) / 2; }
