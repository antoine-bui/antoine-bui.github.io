// Pure logic (no DOM) — tested by logic.test.mjs with `node badminton/logic.test.mjs`.

// Doubles court in metres. Origin = far-left corner (far from camera), y grows toward the camera.
export const COURT = { w: 6.1, l: 13.4, net: 6.7 };
export const COURT_CORNERS = [[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]];

// Court markings as segments [[x1,y1],[x2,y2]] in metres.
export const COURT_LINES = [
  [[0, 0], [6.1, 0]], [[6.1, 0], [6.1, 13.4]], [[6.1, 13.4], [0, 13.4]], [[0, 13.4], [0, 0]],
  [[0.46, 0], [0.46, 13.4]], [[5.64, 0], [5.64, 13.4]],             // singles sidelines
  [[0, 0.76], [6.1, 0.76]], [[0, 12.64], [6.1, 12.64]],             // doubles long service lines
  [[0, 4.72], [6.1, 4.72]], [[0, 8.68], [6.1, 8.68]],               // short service lines
  [[3.05, 0], [3.05, 4.72]], [[3.05, 8.68], [3.05, 13.4]],          // centre lines
];
export const NET_LINE = [[-0.3, 6.7], [6.4, 6.7]];

// Thresholds for shot classification (tune on your own footage).
export const T = { overhead: 0.15, smashKmh: 140, driveKmh: 80, killKmh: 60, frontZone: 2.2, shortEnd: 3.0 };

// 3x3 homography (row-major, h33 = 1) mapping 4 src points onto 4 dst points.
export function homography(src, dst) {
  const A = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  // Gaussian elimination with partial pivoting on the 8x9 augmented matrix.
  for (let c = 0; c < 8; c++) {
    let p = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < 8; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 9; k++) A[r][k] -= f * A[c][k];
    }
  }
  return [...A.map((row, i) => row[8] / row[i]), 1];
}

export function project(H, [x, y]) {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

export const inCourt = ([x, y], margin = 0) =>
  x >= -margin && x <= COURT.w + margin && y >= -margin && y <= COURT.l + margin;

// Connected components (4-neighbour) of a binary mask; keeps blobs with minA <= area <= maxA.
export function findBlobs(mask, w, h, minA, maxA) {
  const seen = new Uint8Array(mask.length), out = [], stack = [];
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || seen[i]) continue;
    let n = 0, sx = 0, sy = 0;
    seen[i] = 1; stack.push(i);
    while (stack.length) {
      const k = stack.pop(), x = k % w, y = (k / w) | 0;
      n++; sx += x; sy += y;
      const nb = [x > 0 && k - 1, x < w - 1 && k + 1, y > 0 && k - w, y < h - 1 && k + w];
      for (const q of nb) if (q !== false && mask[q] && !seen[q]) { seen[q] = 1; stack.push(q); }
    }
    if (n >= minA && n <= maxA) out.push({ x: sx / n, y: sy / n, area: n });
  }
  return out;
}

// A hit = the shuttle direction turns by more than ~60° or it suddenly accelerates.
export function isHit(v0, v1) {
  const n0 = Math.hypot(...v0), n1 = Math.hypot(...v1);
  if (n0 < 1e-6) return n1 > 0;
  if (n1 < 1e-6) return false;
  const cos = (v0[0] * v1[0] + v0[1] * v1[1]) / (n0 * n1);
  return cos < 0.5 || n1 > 1.8 * n0;
}

// Multi-candidate tracker. Every bright moving blob starts a candidate; a candidate grows when a blob shows up
// where its constant-velocity prediction says. `turn` flags a direction break (a hit) on an established path.
// blobs: [[x, y]], t: ms, gate: px. Mutates and returns the candidate list.
export function trackCandidates(cands, blobs, t, gate) {
  const used = new Set();
  cands.sort((a, b) => b.path.length - a.path.length); // established paths pick first
  for (const c of cands) {
    const last = c.path[c.path.length - 1], dt = t - last.t;
    if (dt <= 0) continue;
    const moving = c.path.length > 1;
    const pred = [last.p[0] + c.v[0] * dt, last.p[1] + c.v[1] * dt];
    // a lone blob has no velocity yet: search a radius that grows with the time gap (frames may be skipped)
    let bi = -1, bd = moving ? gate + Math.hypot(...c.v) * dt * 0.5 : gate * 3 * Math.max(1, dt / 33);
    blobs.forEach((b, i) => { if (!used.has(i)) { const d = Math.hypot(b[0] - pred[0], b[1] - pred[1]); if (d < bd) { bd = d; bi = i; } } });
    c.turn = false;
    if (bi < 0) continue;
    used.add(bi);
    const b = blobs[bi], v = [(b[0] - last.p[0]) / dt, (b[1] - last.p[1]) / dt];
    c.turn = moving && isHit(c.v, v);
    c.v = v;
    c.path.push({ p: b, t });
    if (c.path.length > 60) { c.path.shift(); c.seg = Math.max(0, c.seg - 1); }
    c.seen = t;
  }
  blobs.forEach((b, i) => { if (!used.has(i)) cands.push({ path: [{ p: b, t }], v: [0, 0], seen: t, turn: false, seg: 0, done: false }); });
  // singles die fast (noise), established paths survive short occlusions
  return cands.filter((c) => t - c.seen < (c.path.length >= 3 ? 300 : 70)).slice(0, 40);
}

// Shuttle = a path of >= 4 points, fast (px/ms), turning less than ~45° per step (a flight arc is smooth at 30 fps):
// flicker, slow limbs and random noise chains fail this.
export function isShuttle(c, minSpeed) {
  if (c.path.length < 4) return false;
  const q = c.path.slice(-4), d = q.slice(1).map((x, i) => [x.p[0] - q[i].p[0], x.p[1] - q[i].p[1]]);
  const smooth = (u, v) => u[0] * v[0] + u[1] * v[1] > 0.7 * Math.hypot(...u) * Math.hypot(...v);
  const speed = Math.hypot(q[3].p[0] - q[0].p[0], q[3].p[1] - q[0].p[1]) / (q[3].t - q[0].t);
  return smooth(d[0], d[1]) && smooth(d[1], d[2]) && speed >= minSpeed;
}

// Launch speed (km/h) from the court distance D (m) covered in T (s). A shuttle is a quadratic-drag projectile with a
// ~6.8 m/s terminal speed: v(t) = v0 / (1 + k v0 t)  =>  D = ln(1 + k v0 T) / k  =>  v0 = (e^(kD) - 1) / (kT).
// Gravity and height are ignored (flat-flight approximation), so lobs read a bit low.
export const DRAG_K = 9.81 / 6.8 ** 2;
export function launchKmh(D, T) {
  return Math.round((Math.exp(DRAG_K * D) - 1) / (DRAG_K * T) * 3.6);
}

// Share (0-1) of a D-metre flight lasting T seconds covered after t seconds, with the same drag model:
// x(t) = ln(1 + k v0 t) / k, i.e. fast right after the hit, slow at the end. Used to place the shuttle on the minimap.
export function flightFraction(D, T, t) {
  if (t <= 0) return 0;
  if (D < 0.5) return Math.min(1, t / T);
  const v0 = (Math.exp(DRAG_K * D) - 1) / (DRAG_K * T);
  return Math.min(1, Math.log(1 + DRAG_K * v0 * t) / (DRAG_K * D));
}

// Cross-court: the shuttle ends on the other half of the court (left/right of the centre line) and moved sideways
// at least 1.5 m. x0 = hitter, x1 = where it went (receiver or landing), in court metres.
export const isCross = (x0, x1) => (x0 - COURT.w / 2) * (x1 - COURT.w / 2) < 0 && Math.abs(x1 - x0) >= 1.5;

// offset: contact point beside the body, in body widths, positive = on the player's right. rel: contact height in the
// player box (0 = top of the head). A contact clearly on the non-racket side is around-the-head only for an overhead
// shot type hit above the head (ShuttleSet: 24-33 % of clears/smashes/drops are around-the-head, 1-9 % backhand),
// a backhand otherwise (45-60 % of net shots, lobs and pushes are backhands).
const AROUND_TYPES = new Set(['clear', 'smash', 'wrist smash', 'drop']);
export function handModifiers(type, offset, rightHanded, rel) {
  if ((rightHanded ? -offset : offset) <= 0.3) return { backhand: false, aroundhead: false };
  const around = AROUND_TYPES.has(type) && rel < T.overhead;
  return { backhand: !around, aroundhead: around };
}

// ShuttleSet's 18 shot types (driven flight merged into drive) -> display names. "rush" is the net kill.
export const SHOT_NAMES = {
  'net shot': 'Net shot', 'return net': 'Return net', 'cross-court net shot': 'Cross-court net shot', rush: 'Kill',
  push: 'Push', lob: 'Lob', 'defensive return lob': 'Defensive lob', clear: 'Clear', drive: 'Drive',
  'back-court drive': 'Back-court drive', 'defensive return drive': 'Defensive drive', drop: 'Drop',
  'passive drop': 'Passive drop', smash: 'Smash', 'wrist smash': 'Wrist smash', 'short service': 'Short serve',
  'long service': 'Long serve',
};
const NO_CROSS = new Set(['net shot', 'return net', 'cross-court net shot', 'short service', 'long service']);

// English name: [Backhand | Around-the-head] [cross] base, e.g. "Backhand cross smash", "Around-the-head clear".
// A crossed net shot is its own ShuttleSet type ("Cross-court net shot"); serves and net replies are never "cross".
export function shotName(type, { backhand = false, aroundhead = false, cross = false } = {}) {
  if (type === 'net shot' && cross) type = 'cross-court net shot';
  const base = (SHOT_NAMES[type] || type).toLowerCase();
  const words = [aroundhead ? 'around-the-head' : backhand && 'backhand', cross && !NO_CROSS.has(type) && 'cross', base]
    .filter(Boolean).join(' ');
  return words[0].toUpperCase() + words.slice(1);
}

// Shot-type model (models/shots.json, trained by tools/train_shots.py on ShuttleSet22): the input vector, built
// exactly like in training. Court metres, hitter brought to the near side (y > net) by a 180 deg turn, which keeps
// the player's own left/right. T = flight time to the next hit in s (null = the rally ended on this shot).
export function shotFeatures(model, { hitter, opponent, landing, T, round, prev, backhand }) {
  const far = hitter[1] < COURT.net, f = ([x, y]) => (far ? [COURT.w - x, COURT.l - y] : [x, y]);
  const [hx, hy] = f(hitter), [ox, oy] = f(opponent), [lx, ly] = f(landing), t = T == null ? -1 : T;
  const v = {
    hx, hy, ox, oy, lx, ly, hit_net: hy - COURT.net, land_net: COURT.net - ly, side: lx - hx,
    dist: Math.hypot(lx - hx, ly - hy), T: t, T_known: t >= 0 ? 1 : 0, round: Math.min(round, 12),
    first: round === 1 ? 1 : 0, backhand: backhand ? 1 : 0,
  };
  for (const p of model.prev) v['prev_' + p] = prev === p ? 1 : 0;
  return model.features.map((k) => v[k]);
}

// Forward pass of the scikit-learn MLP: standardise, ReLU hidden layers, softmax. Returns class probabilities.
export function shotProbs(model, x) {
  let h = x.map((v, i) => (v - model.mean[i]) / model.scale[i]);
  model.layers.forEach(({ W, b }, k) => {
    const out = b.slice();
    h.forEach((hi, i) => { if (hi) for (let j = 0; j < out.length; j++) out[j] += hi * W[i][j]; });
    h = k < model.layers.length - 1 ? out.map((v) => Math.max(0, v)) : out;
  });
  const m = Math.max(...h), e = h.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// ---------- Virtual umpire (idea from CoachAI's Segmentation.py, on court metres instead of image pixels) ----------
// Where the shuttle landed, for the player who hit it last. "net" = it never crossed the net (fault of the hitter).
// Lines count as in. Serves: past the short service line, before the back service line (doubles: 0.76 m inside the
// baseline), within the serve side lines, and in the diagonally opposite box (other half of the court).
export function lineCall(hitterCourt, landing, { doubles = false, serve = false } = {}) {
  const [x, y] = landing, near = hitterCourt[1] >= COURT.net;
  if ((y >= COURT.net) === near) return 'net';
  const depth = Math.abs(y - COURT.net); // metres from the net, on the receiver's side
  const [x0, x1] = doubles ? [0, COURT.w] : [0.46, COURT.w - 0.46];
  if (x < x0 || x > x1 || depth > COURT.l / 2) return 'out';
  if (serve) {
    if (depth < 1.98 || (doubles && depth > COURT.l / 2 - 0.76)) return 'out';
    if ((hitterCourt[0] - COURT.w / 2) * (x - COURT.w / 2) > 0) return 'out'; // same half = not diagonal
  }
  return 'in';
}

// Score from the rallies seen, in order: { t, winner: 'near' | 'far' }. Scores belong to teams (A = near side at the
// start), not sides: ends change after each set and at 11 in the deciding set. A set: 21 with a 2-point lead, cap 30.
// Returns the near/far view used on screen plus the per-rally history (for the timeline chart and the CSV export).
export function scoreBoard(rallies) {
  const pts = [0, 0], sets = [0, 0], history = [];
  let nearTeam = 0, set = 1, switchedAt11 = false;
  for (const r of rallies) {
    const team = r.winner === 'near' ? nearTeam : 1 - nearTeam;
    pts[team]++;
    history.push({ t: r.t, set, a: pts[0], b: pts[1], team });
    const [w, l] = [pts[team], pts[1 - team]];
    if ((w >= 21 && w - l >= 2) || w === 30) {
      sets[team]++; pts[0] = pts[1] = 0; set++; nearTeam = 1 - nearTeam; switchedAt11 = false;
    } else if (set === 3 && w === 11 && !switchedAt11) {
      nearTeam = 1 - nearTeam; switchedAt11 = true;
    }
  }
  return { near: { pts: pts[nearTeam], sets: sets[nearTeam] }, far: { pts: pts[1 - nearTeam], sets: sets[1 - nearTeam] }, set, nearTeam, history };
}

// ---------- Export: one row per shot, in ShuttleSet's column layout (CoachAI) ----------
// Positions like ShuttleSet: camera pixels (court -> image through Hinv, the inverse calibration). A = the side at the
// camera when the analysis started (ends change each set). Extra columns at the end: speed, metres, player J1-J4.
export const CSV_COLUMNS = ['rally', 'ball_round', 'time', 'frame_num', 'roundscore_A', 'roundscore_B', 'player', 'server', 'type',
  'aroundhead', 'backhand', 'hit_height', 'hit_area', 'hit_x', 'hit_y', 'landing_height', 'landing_area', 'landing_x', 'landing_y',
  'lose_reason', 'win_reason', 'getpoint_player', 'flaw', 'player_location_area', 'player_location_x', 'player_location_y',
  'opponent_location_area', 'opponent_location_x', 'opponent_location_y', 'db',
  'speed_kmh', 'player_x_m', 'player_y_m', 'opponent_x_m', 'opponent_y_m', 'landing_x_m', 'landing_y_m', 'player_id'];

export function shotsCsv(shots, rallies, Hinv, fps = 30) {
  const ordered = [...shots].sort((a, b) => a.t - b.t), decided = rallies.filter((r) => r.winner).sort((a, b) => a.t - b.t);
  const px = (p) => (p ? project(Hinv, p).map((v) => v.toFixed(1)) : ['', '']), m = (p) => (p ? p.map((v) => v.toFixed(2)) : ['', '']);
  const hms = (ms) => new Date(ms).toISOString().slice(11, 19);
  const lastOfRally = new Map(ordered.map((s) => [s.rally, s]));
  const rows = ordered.map((s) => {
    const before = scoreBoard(decided.filter((r) => r.t < s.t)), letter = (near) => ((near ? before.nearTeam : 1 - before.nearTeam) === 0 ? 'A' : 'B');
    const near = s.court[1] >= COURT.net, rally = decided.find((r) => r.rally === s.rally);
    const after = rally ? scoreBoard(decided.filter((r) => r.t <= rally.t)).history.at(-1) : null;
    const serveShot = ordered.find((x) => x.rally === s.rally && x.serve);
    const end = lastOfRally.get(s.rally) === s && rally;
    const lose = end && (rally.reason === 'out' ? 'out' : rally.reason === 'net' ? 'not over the net' : '');
    return [s.rally, s.round, hms(s.t), Math.round(s.t / 1000 * fps), after ? after.a : '', after ? after.b : '', letter(near),
      serveShot ? letter(serveShot.court[1] >= COURT.net) : '', s.type, s.aroundhead ? 1 : 0, s.backhand ? 1 : 0, '', '', ...px(s.court), '', '', ...px(s.to),
      lose || '', end && rally.reason === 'in' ? 'landed in' : '', end ? (rally.winner === 'near' ? letter(true) : letter(false)) : '', '',
      '', ...px(s.court), '', ...px(s.opp), '', s.kmh ?? '', ...m(s.court), ...m(s.opp), ...m(s.to), `J${s.id + 1}`];
  });
  return [CSV_COLUMNS, ...rows].map((r) => r.map((v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v)).join(',')).join('\n') + '\n';
}

// ---------- Next-stroke forecast (ShuttleNet, CoachAI AAAI 2022, retrained by tools/train_forecast.py) ----------
// Inputs for the model from the rally so far (shots in order, first = serve): shot type index (1-based, 0 = padding),
// landing point normalised like in training ((x - 3.05) / 1.5, (y - 6.7) / 3.5, court turned so the server is near),
// role token (1 = server, 2 = receiver) of each hitter and of the next hitter. Fixed length `len` (padding after).
export function forecastInputs(shots, classes, len) {
  const flip = shots[0].court[1] < COURT.net, serverNear = !flip;
  const role = (s) => ((s.court[1] >= COURT.net) === serverNear ? 1 : 2);
  const z = (a) => Array(len).fill(a);
  const inp = { shot: z(0), x: z(0), y: z(0), player: z(0), target: z(0) };
  shots.slice(0, len).forEach((s, i, arr) => {
    const [x, y] = flip ? [COURT.w - s.to[0], COURT.l - s.to[1]] : s.to;
    inp.shot[i] = classes.indexOf(s.type) + 1;
    inp.x[i] = (x - COURT.w / 2) / 1.5;
    inp.y[i] = (y - COURT.net) / 3.5;
    inp.player[i] = role(s);
    inp.target[i] = i + 1 < arr.length ? role(arr[i + 1]) : 3 - role(s); // next hitter: the other side
  });
  return { ...inp, flip, n: Math.min(shots.length, len) };
}

// Bivariate normal (metres) -> ellipse at one standard deviation: radii along the principal axes and their angle.
export function ellipse(sx, sy, corr) {
  const a = sx * sx, b = corr * sx * sy, c = sy * sy, h = (a - c) / 2, r = Math.sqrt(h * h + b * b);
  return { rx: Math.sqrt((a + c) / 2 + r), ry: Math.sqrt(Math.max(0, (a + c) / 2 - r)), angle: 0.5 * Math.atan2(2 * b, a - c) };
}

// Fallback rules when the model can't be loaded. rel: shuttle height at impact relative to the player box
// (0 = top of head, 1 = feet). hitterNetDist / endNetDist: metres from the net. rising: shuttle goes up after impact.
// Returns ShuttleSet type keys. (A serve is decided before this: the first hit of a rally.)
export function classifyShot({ rel, hitterNetDist, endNetDist, rising, kmh }) {
  // kill (rush): at the net, hit downward and fast (whatever the contact height)
  if (hitterNetDist < T.frontZone && !rising && kmh >= T.killKmh) return 'rush';
  if (rel < T.overhead) {
    if (!rising && kmh >= T.smashKmh) return 'smash';
    return endNetDist < T.shortEnd ? 'drop' : 'clear';
  }
  if (hitterNetDist < T.frontZone) {
    if (endNetDist < T.frontZone) return 'net shot';
    return rising ? 'lob' : 'push';
  }
  if (rising) return 'clear';
  return kmh >= T.driveKmh ? 'drive' : 'push';
}
