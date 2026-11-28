// Shuttle worker: hit/shot logic on the shuttle positions, one state per analysed frame (tagged with its media time).
// Positions come from TrackNet (tracknet.mjs, via the page) or, without WebGPU, from the hand-made detector below.
// Player boxes come from detector.mjs via the page.
import { COURT, T as LIMITS, project, inCourt, findBlobs, classifyShot, trackCandidates, isShuttle, launchKmh, isCross, handModifiers, shotName,
  shotFeatures, shotProbs, lineCall } from './logic.mjs';

const WORK_W = 480, DIFF_T = 18, BRIGHT = 130; // shuttle search resolution, min brightness jump over background, min luminance
const MOVE_T = 6; // min change vs the previous frame: rejects bright things that appeared and then stopped
const BG_RATE = 0.08; // background learning rate per analysed frame
const CUT_T = 22; // mean frame-to-frame change above this = a cut in the video
const MIN_SPEED = 0.00025; // shuttle min speed, in video widths per ms (slower = limbs, flicker)
const SERVE_GAP = 5000; // ms without any hit before a hit counts as a serve (pauses between points are longer)
const GHOST_MS = 250; // keep showing a briefly occluded shuttle at its predicted position for this long

let H = null, vw = 0, vh = 0, work = null, wctx = null;
let players = [], others = [], raw = null, prev = null, bg = null, cands = [], cur = null, lastSeen = null, lastHit = null, pending = null, shots = [], event = null;
let hands = []; // hands[id] === false for a detected left-hander (pose votes, sent by the page); right-handed otherwise
let rawHist = [], contact = null; // recent TrackNet points; contact of the last hit, reported to the page for the pose vote
let startT = null; // media time of the first analysed frame since the last reset
let rallyBreak = false; // a cut was seen: the next hit starts a new rally (serve)
let doubles = false, still = []; // still: last TrackNet detections, to spot a shuttle that stopped moving
// Rallies & score: round = stroke number in the rally, prevKey = previous shot type (model inputs), rallies = ended
// rallies { t, rally, winner: 'near' | 'far', reason: 'in' | 'out' | 'net' | '?', at }, open = the last rally still
// waiting for its winner (no landing seen: the next server tells, the rally winner always serves).
let round = 0, prevKey = 'none', rallyNo = 1, rallies = [], open = null, callEvt = null, dirty = true;
// Shot-type model trained on ShuttleSet22 (tools/train_shots.py); hand-written rules until it loads or if it fails.
let shotModel = null;
fetch(new URL('./models/shots.json', import.meta.url)).then((r) => r.json()).then((m) => { shotModel = m; }, () => {});

onmessage = handle;

// ---------- Messages ----------
function handle({ data }) {
  if (data.type === 'config') {
    ({ H, vw, vh } = data);
    doubles = data.perSide === 2;
    rallies = []; open = null; rallyNo = 1; dirty = true;
    if (H) {
      work = new OffscreenCanvas(WORK_W, Math.round(WORK_W * vh / vw));
      wctx = work.getContext('2d', { willReadFrequently: true });
    }
    shots = []; resetTracking();
  } else if (data.type === 'reset') {
    resetTracking();
  } else if (data.type === 'frame') { // fallback detector: one frame
    ({ players, others, hands } = data);
    try { if (H) { raw = null; const blobs = blobsOf(data.bitmap); if (blobs) analyze(blobs, data.t); } } finally { data.bitmap.close(); }
    post(data.t);
  } else if (data.type === 'points') { // TrackNet: a batch of frames, each with 0 or 1 shuttle position
    ({ players, others, hands } = data);
    for (let { t, p, cut } of data.points) {
      if (H && cut) newScene();
      if (H && p) p = stillCheck(p, t); else still = [];
      if (p) { rawHist.push({ t, p }); while (rawHist[0].t < t - 1000) rawHist.shift(); }
      if (H) { raw = p; analyze(p ? [p] : [], t); }
      post(t);
    }
  }
}

// shots/rallies only travel when they changed (a full match is ~1000 shots, states are 30/s)
function post(t) {
  postMessage({ type: 'state', t, ...shuttleView(t), event, contact, call: callEvt, ...(dirty ? { shots, rallies } : {}) });
  event = contact = callEvt = null; dirty = false;
}

// A detection that does not move for 5 frames in a row is not a shuttle in play (idea from CoachAI's Segmentation.py):
// it is dropped. If it is where the shuttle of the current shot was last seen, the shuttle has landed: rally over.
function stillCheck(p, t) {
  still.push({ t, p });
  if (still.length > 5) still.shift();
  const run = still.length === 5 && still.every((q, i) => !i || q.t - still[i - 1].t < 45)
    && still.every((q) => dist(q.p, still[0].p) < vw * 0.006);
  if (!run) return p;
  const s = still[0];
  // landed only if it rests on the floor near the court (a fixed object up on a wall projects far behind it)
  const at = project(H, s.p);
  if (pending && lastSeen && dist(s.p, lastSeen.p) < vw * 0.08 && s.t - lastSeen.t < 600 && inCourt(at, 2)) finalizeShot(at, s.t, 'landed');
  return null;
}

// t = media time in ms (video clock), so speeds stay right even when frames are skipped.
function analyze(blobs, t) {
  if (startT === null) { // first frame after a seek: forget what came after (it will be analysed again)
    startT = t;
    shots = shots.filter((s) => s.t < t); rallies = rallies.filter((r) => r.t < t); open = null; dirty = true;
  }
  trackBlobs(blobs, t);
  // shuttle lost for 0.8 s (out of frame, occluded...): the shot ends at its last seen position, the rally goes on
  if (pending && lastSeen && t - lastSeen.t > 800) finalizeShot(project(H, lastSeen.p), lastSeen.t, 'lost');
}

// Shuttle candidates = small blobs BRIGHTER than the learned background (a white shuttle over a darker floor/wall) that
// also changed since the previous frame, outside player boxes. A background model (not just the previous frames) keeps
// a slow shuttle at the top of its arc. Arms, shadows and dark clothes get darker or don't pass the brightness test.
// Only used without WebGPU: it loses the shuttle against bright backgrounds (lights, white walls), TrackNet does not.
function blobsOf(bitmap) {
  const W = work.width, Hh = work.height, s = W / vw;
  wctx.drawImage(bitmap, 0, 0, W, Hh);
  const px = wctx.getImageData(0, 0, W, Hh).data, g = new Uint8Array(W * Hh);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = (px[j] * 77 + px[j + 1] * 150 + px[j + 2] * 29) >> 8;

  let change = 0;
  if (prev) for (let i = 0; i < g.length; i += 16) change += Math.abs(g[i] - prev[i]);
  if (!prev || change / (g.length / 16) > CUT_T) { // first frame or cut in the video: restart from this image
    resetTracking();
    prev = g; bg = Float32Array.from(g);
    return null;
  }
  const mask = new Uint8Array(g.length);
  for (let i = 0; i < g.length; i++) {
    mask[i] = g[i] > BRIGHT && g[i] - bg[i] > DIFF_T && g[i] - prev[i] > MOVE_T ? 1 : 0;
    bg[i] += (g[i] - bg[i]) * BG_RATE;
  }
  prev = g;
  for (const p of players) {
    // box grown by 20% sideways / 30% down: white shoes, racket, swinging arms and their reflection on a glossy floor
    const mx = p.w * 0.2, my = p.h * 0.3;
    const x0 = Math.max(0, (p.x - mx) * s | 0), x1 = Math.min(W, (p.x + p.w + mx) * s | 0);
    const y0 = Math.max(0, p.y * s | 0), y1 = Math.min(Hh, (p.y + p.h + my) * s | 0);
    for (let y = y0; y < y1; y++) mask.fill(0, y * W + x0, y * W + x1);
  }
  for (const p of others) { // spectators, umpire: their moves are not the shuttle
    const x0 = Math.max(0, p.x * s | 0), x1 = Math.min(W, (p.x + p.w) * s | 0), y0 = Math.max(0, p.y * s | 0), y1 = Math.min(Hh, (p.y + p.h) * s | 0);
    for (let y = y0; y < y1; y++) mask.fill(0, y * W + x0, y * W + x1);
  }
  return findBlobs(mask, W, Hh, 1, 60).map((b) => [b.x / s, b.y / s]);
}

function trackBlobs(blobs, t) {
  cands = trackCandidates(cands, blobs, t, vw * 0.03);
  const minSpeed = vw * MIN_SPEED;
  for (const c of cands) if (c.turn) { c.seg = c.path.length - 2; c.done = false; } // direction break: new segment from the impact point
  pickShuttle(t, minSpeed);
  if (!cur || cur.seen !== t) return;
  lastSeen = cur.path[cur.path.length - 1];

  // A segment confirmed over 4 points that starts within a player's reach = a hit by that player.
  if (!cur.done && cur.path.length - cur.seg >= 4 && isShuttle(cur, minSpeed)) {
    cur.done = true;
    const impact = cur.path[cur.seg], hitter = nearestPlayer(impact.p);
    if (hitter) {
      const side = hitter.court[1] < COURT.net ? 'far' : 'near';
      const newRally = !lastHit || impact.t - lastHit.t > 2000;
      // serve = first hit after a cut in the video or a real pause (>= 5 s without any hit, counted from the start of the analysis after
      // a seek): one or two hits missed mid-rally leave gaps up to ~4 s, a pause between points is longer
      const serve = rallyBreak || impact.t - (lastHit ? lastHit.t : startT) >= SERVE_GAP;
      if (serve || newRally || (impact.t - lastHit.t > 300 && side !== lastHit.side)) registerHit(hitter, cur, side, serve);
    }
  }
}

// Once a flight is confirmed, keep following it even when it slows down at the top of its arc. When it vanishes, hand over
// to a new confirmed flight, or right away to a young fast path starting where it vanished (the shuttle after a hit).
function pickShuttle(t, minSpeed) {
  if (cur && cur.seen === t && cands.includes(cur)) return;
  const last = cur && cur.path[cur.path.length - 1];
  const next = cands.filter((c) => c !== cur && c.seen === t && (isShuttle(c, minSpeed) || (last && c.path.length >= 2
      && Math.hypot(...c.v) >= minSpeed && c.path[0].t >= last.t - 40 && dist(c.path[0].p, last.p) < vw * 0.1)))
    .sort((a, b) => b.path.length - a.path.length)[0];
  if (next) cur = next;
  else if (cur && (!cands.includes(cur) || t - cur.seen > GHOST_MS)) cur = null;
}

// Where to draw the shuttle: its last detection, or for a short occlusion its predicted position (frozen near a player,
// where it was probably just hit and the old direction is wrong).
function shuttleView(t) {
  if (raw) return { shuttle: raw, ghost: false }; // TrackNet saw it in this very frame: show exactly that
  if (!cur) return { shuttle: null, ghost: false };
  const L = cur.path[cur.path.length - 1], dt = t - L.t;
  if (dt <= 0) return { shuttle: L.p, ghost: false };
  if (dt > GHOST_MS) return { shuttle: null, ghost: false };
  const p = nearestPlayer(L.p) ? L.p : [L.p[0] + cur.v[0] * dt, L.p[1] + cur.v[1] * dt];
  return { shuttle: [Math.min(vw, Math.max(0, p[0])), Math.min(vh, Math.max(0, p[1]))], ghost: true };
}

function nearestPlayer(pt) {
  let best = null, bd = Infinity;
  for (const p of players) {
    // reach zone: racket arm + jump above the head
    const inside = pt[0] > p.x - 0.6 * p.w && pt[0] < p.x + 1.6 * p.w && pt[1] > p.y - 0.6 * p.h && pt[1] < p.y + p.h;
    const d = dist(pt, [p.x + p.w / 2, p.y + p.h / 2]);
    if (inside && d < bd) { bd = d; best = p; }
  }
  return best;
}

function registerHit(hitter, c, side, serve) {
  const impact = c.path[c.seg], after = c.path[Math.min(c.path.length - 1, c.seg + 3)];
  if (pending) finalizeShot(hitter.court, impact.t, 'hit');
  if (serve) newRally(side, impact.t);
  round = serve ? 1 : Math.max(2, round + 1);
  const near = hitter.court[1] >= COURT.net;
  pending = { id: hitter.id, t: impact.t, p0: impact.p, court: hitter.court, rel: (impact.p[1] - hitter.y) / hitter.h,
    rising: after.p[1] < impact.p[1] - 2, offset: sideOffset(hitter, impact.p), serve, round, prev: serve ? 'none' : prevKey,
    opps: players.filter((p) => (p.court[1] >= COURT.net) !== near).map((p) => p.court) };
  lastHit = { t: impact.t, side };
  rallyBreak = false;
  // Racket-hand vote, done by the page on the full-res frame: at the real contact (the shuttle point closest to the
  // hitter within 300 ms before) the wrist nearest the shuttle holds the racket. Not on serves: the free hand holds
  // the shuttle there.
  if (!serve) {
    const edge = ({ p }) => Math.hypot(p[0] - Math.min(Math.max(p[0], hitter.x), hitter.x + hitter.w), p[1] - Math.min(Math.max(p[1], hitter.y), hitter.y + hitter.h));
    const c = rawHist.filter((q) => q.t >= impact.t - 300).sort((a, b) => edge(a) - edge(b))[0];
    if (c) contact = { t: c.t, p: c.p, id: hitter.id, box: { x: hitter.x, y: hitter.y, w: hitter.w, h: hitter.h } };
  }
}

// End of a shot. ending: 'hit' (next hit), 'landed' (shuttle at rest on the floor), 'lost' (out of sight), 'cut'.
// Real court distance + flight time give the launch speed.
function finalizeShot(endCourt, endT, ending) {
  const D = Math.hypot(endCourt[0] - pending.court[0], endCourt[1] - pending.court[1]), T = (endT - pending.t) / 1000;
  // outside these bounds a hit was missed or a position is off (court diagonal = 14.7 m): show no speed rather than a wrong one
  const v = T > 0.15 && T < 2.5 && D > 0.5 && D < 15 ? launchKmh(D, T) : 0;
  const kmh = v >= 15 && v <= 420 ? v : null;
  const right = hands[pending.id] !== false;
  // model input: off-side contact below the head (the model then tells the type, the name modifiers follow)
  const lowBackhand = handModifiers('lob', pending.offset, right, pending.rel).backhand && pending.rel >= LIMITS.overhead;
  // opponent = the receiver: the player on the other side closest to where the shuttle went
  const opp = pending.opps.sort((a, b) => Math.hypot(a[0] - endCourt[0], a[1] - endCourt[1]) - Math.hypot(b[0] - endCourt[0], b[1] - endCourt[1]))[0]
    || [COURT.w - pending.court[0], COURT.l - pending.court[1]];
  const type = shotType(endCourt, ending === 'hit' ? T : null, kmh, opp, lowBackhand);
  const hand = handModifiers(type, pending.offset, right, pending.rel);
  const name = shotName(type, { ...hand, cross: isCross(pending.court[0], endCourt[0]) });
  prevKey = type;
  // from/to/t..t1 = the flight on the court (minimap); the rest feeds the stats panel and the CSV export
  shots.unshift({ t: pending.t, t1: endT, id: pending.id, kmh, type, name, court: pending.court, to: endCourt, opp,
    rally: rallyNo, round: pending.round, backhand: hand.backhand, aroundhead: hand.aroundhead, serve: pending.serve });
  dirty = true;
  // tagged with the impact time: the analysis runs ~2 s ahead, so the page can show it at the very hit
  event = { t: pending.t, text: kmh ? `${name} · ${kmh} km/h` : name, id: pending.id };
  const ended = pending;
  pending = null;
  if (ending === 'landed' || ending === 'cut') endRally(ended, ending === 'landed' ? endCourt : null, endT);
}

// ShuttleSet type of the shot: the trained model when loaded (serves can only be serves, and the other way round),
// the hand-written rules otherwise.
function shotType(endCourt, T, kmh, opp, backhand) {
  if (shotModel) {
    const probs = shotProbs(shotModel, shotFeatures(shotModel, { hitter: pending.court, opponent: opp, landing: endCourt, T,
      round: pending.round, prev: pending.prev, backhand }));
    let best = -1;
    shotModel.classes.forEach((c, i) => { if (c.endsWith('service') === !!pending.serve && (best < 0 || probs[i] > probs[best])) best = i; });
    return shotModel.classes[best];
  }
  if (pending.serve) return Math.abs(endCourt[1] - COURT.net) < 3.5 ? 'short service' : 'long service';
  return classifyShot({ rel: pending.rel, rising: pending.rising, kmh: kmh || 0,
    hitterNetDist: Math.abs(pending.court[1] - COURT.net), endNetDist: Math.abs(endCourt[1] - COURT.net) });
}

// ---------- Virtual umpire ----------
// Landing seen: line call, the point goes to the hitter if "in", to the other side otherwise. No landing (cut):
// the rally stays open until the next serve, the server being the rally winner.
function endRally(shot, landing, t) {
  const hitterSide = shot.court[1] >= COURT.net ? 'near' : 'far';
  if (landing) {
    const reason = lineCall(shot.court, landing, { doubles, serve: shot.serve });
    open = { t, rally: rallyNo, winner: reason === 'in' ? hitterSide : other(hitterSide), reason, at: landing };
    rallies.push(open);
    callEvt = { t, call: reason, at: landing };
  } else {
    open = { t, rally: rallyNo, winner: null, reason: '?', at: null };
  }
  dirty = true;
  rallyNo++; round = 0; prevKey = 'none';
}

// A serve starts a new rally. The server won the previous one: that settles an open rally, or corrects a line call
// that said otherwise (call kept as the reason only when it agrees).
function newRally(serverSide, t) {
  if (!open && shots.length && shots[0].rally === rallyNo) { // the previous rally ended unseen (shuttle lost, then a serve)
    open = { t: shots[0].t1, rally: rallyNo, winner: null, reason: '?', at: null };
    rallyNo++;
  }
  if (open) {
    if (open.winner === null) rallies.push(open);
    if (open.winner !== serverSide) { open.winner = serverSide; open.reason = '?'; }
    open = null; dirty = true;
  }
  round = 0; prevKey = 'none';
}
const other = (side) => (side === 'near' ? 'far' : 'near');

// Contact point beside the body in body widths, positive = on the PLAYER's right. Players face the net: the near-side
// one turns their back to the camera (their right = court +x), the far-side one faces it (their right = court -x).
// Image-left/right vs court x is read from the calibration at the player's feet.
function sideOffset(p, contact) {
  const imgRightIsPlusX = project(H, [p.feet[0] + 10, p.feet[1]])[0] > p.court[0];
  const near = p.court[1] >= COURT.net;
  return (contact[0] - (p.x + p.w / 2)) / p.w * (imgRightIsPlusX ? 1 : -1) * (near ? 1 : -1);
}


// Cut in the video: close the shot in progress where the shuttle was last seen, forget the old trajectories.
function newScene() {
  if (pending && lastSeen) finalizeShot(project(H, lastSeen.p), lastSeen.t, 'cut');
  pending = null; cands = []; cur = null; rallyBreak = true;
}

function resetTracking() { rawHist = []; contact = null; prev = null; bg = null; cands = []; cur = null; lastSeen = null; lastHit = null; pending = null; startT = null; }
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
