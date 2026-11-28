// UI thread: video player, calibration, drawing. Analysis runs in workers so playback never stutters:
// tracknet.mjs finds the shuttle (TrackNetV3 on the GPU), detector.mjs the players, worker.mjs turns that into hits/shots.
// For a video file the analysis runs on a hidden twin video ~2 s AHEAD of the visible one, and every result is tagged
// with its media time: the visible video then shows the result of the exact frame on screen, with no lag.
import { COURT, COURT_CORNERS, COURT_LINES, NET_LINE, homography, project, flightFraction, scoreBoard, shotsCsv, ellipse, SHOT_NAMES } from './logic.mjs';
import { dashboard } from './dashboard.mjs';

const COLORS = ['#ff4d4d', '#3da5ff', '#ffd23d', '#4dff88'];
const CORNER_HINTS = ['coin du fond à gauche', 'coin du fond à droite', 'coin avant à droite', 'coin avant à gauche'];
const ANALYSIS_W = 512, ANALYSIS_H = 288; // TrackNet's input size (16:9); the detector is fine with it too
const ANALYSIS_MS = 33; // analyse ~30 frames/s (TrackNet's training rate), even on 60 fps videos
const DETECT_MS = 100; // players move slowly: ~10 detections/s is enough
const LEAD = 2; // seconds the twin video runs ahead of the visible one

const $ = (id) => document.getElementById(id);
const video = $('video'), ahead = $('ahead'), view = $('view'), ctx = view.getContext('2d');
const mini = $('minimap'), mctx = mini.getContext('2d');

let H = null, Hinv = null, calib = [], perSide = 0, detReady = false, tnReady = false, lastT = -1;
let busy = false, lastSent = -Infinity, detBusy = false, lastDet = -Infinity, detT = 0;
let anaPlayers = [], anaOthers = []; // latest detection, fed to the shuttle worker
let handVotes = new Map(), contacts = [];
let allShots = [], allRallies = [], calls = [];
let forecasts = new Map(), fcAsked = new Set(); // next-stroke forecasts by the time of the shot they follow // every shot / ended rally of the analysis so far; line calls to show // racket-hand votes per player id; hit contacts waiting for their frame
let trackTL = [], playTL = [], events = []; // results by media time (ms)
const EMPTY = { shuttle: null, ghost: false, shots: [], pending: null };
let st = EMPTY, players = [], shown = null, vis = null, shownKey = '', shownShots = -1;
const BRIDGE_MS = 500; // longest TrackNet miss bridged between two detections
const disp = new Map(); // id -> smoothed box shown on screen

// ---------- Workers ----------
const workerUrl = (f) => new URL(f, import.meta.url);
const tracker = new Worker(workerUrl('./worker.mjs'), { type: 'module' });
const detector = new Worker(workerUrl('./detector.mjs'), { type: 'module' });
const tracknet = new Worker(workerUrl('./tracknet.mjs'), { type: 'module' });
const forecaster = new Worker(workerUrl('./forecast.mjs'), { type: 'module' });
forecaster.onmessage = ({ data }) => {
  if (data.type === 'forecast') forecasts.set(data.key, data);
  else if (data.type === 'error') console.warn('ShuttleNet indisponible :', data.msg);
};
tracker.onmessage = ({ data }) => {
  busy = false;
  trackTL.push(data);
  if (data.shots) { ({ shots: allShots, rallies: allRallies } = data); $('exportCsv').disabled = !allShots.length; requestForecasts(); }
  if (data.call) calls.push(data.call);
  if (data.event) events.push(data.event); // t = impact time
  if (data.contact) contacts.push(data.contact);
};
detector.onmessage = ({ data }) => {
  if (data.type === 'ready') { detReady = true; modelStatus(); }
  else if (data.type === 'error') $('model').textContent = 'Erreur de chargement du modèle : ' + data.msg;
  else if (data.type === 'hand') {
    const v = handVotes.get(data.id) || { R: 0, L: 0 };
    v[data.right ? 'R' : 'L']++;
    handVotes.set(data.id, v);
  } else if (data.type === 'players') {
    detBusy = false;
    ({ players: anaPlayers, others: anaOthers } = data);
    playTL.push({ t: detT, players: data.players });
  }
};
tracknet.onmessage = ({ data }) => {
  if (data.type === 'ready') { tnReady = true; tracker.postMessage({ type: 'reset' }); modelStatus(); }
  else if (data.type === 'error') { tnReady = false; console.warn('TrackNet indisponible, détecteur de secours :', data.msg); modelStatus(data.msg); }
  else if (data.type === 'points') tracker.postMessage({ type: 'points', points: data.points, players: anaPlayers, others: anaOthers, hands: hands() });
};
const onWorkerError = (e) => { $('model').textContent = 'Erreur du worker : ' + (e.message || 'voir la console'); };
tracker.onerror = detector.onerror = tracknet.onerror = onWorkerError;
function modelStatus(tnError) {
  $('model').textContent = !detReady ? 'Chargement du modèle…'
    : tnReady ? 'Modèles prêts · volant : TrackNetV3 (GPU)'
    : tnError ? 'Modèle prêt · volant : détecteur simple (WebGPU indisponible)' : 'Chargement de TrackNet…';
}

// ---------- Analysis (on the twin video for files, on the live video for the camera) ----------
const src = () => (video.srcObject ? video : ahead);
const grab = () => createImageBitmap(src(), { resizeWidth: ANALYSIS_W, resizeHeight: ANALYSIS_H });
function sendFrame(t) {
  if (!H || !video.videoWidth) return;
  const rewound = t < lastSent;
  if (rewound || t - lastSent >= ANALYSIS_MS) {
    if (tnReady) { // TrackNet needs every frame in order: no skipping, the worker queues them
      lastSent = t;
      grab().then((bitmap) => tracknet.postMessage({ type: 'frame', bitmap, t }, [bitmap]), () => {});
    } else if (!busy) { // fallback detector: one frame in flight, skip the rest
      lastSent = t; busy = true;
      grab().then((bitmap) => tracker.postMessage({ type: 'frame', bitmap, t, players: anaPlayers, others: anaOthers, hands: hands() }, [bitmap]), () => { busy = false; });
    }
  }
  if (detReady && !detBusy && (t < lastDet || t - lastDet >= DETECT_MS)) {
    lastDet = detT = t; detBusy = true;
    grab().then((bitmap) => detector.postMessage({ type: 'frame', bitmap }, [bitmap]), () => { detBusy = false; });
  }
}

// requestVideoFrameCallback fires once per decoded frame and gives the exact media time of that frame.
for (const el of [video, ahead]) {
  if ('requestVideoFrameCallback' in el) {
    const onFrame = (now, meta) => {
      if (src() === el) sendFrame(el.srcObject ? meta.presentationTime : meta.mediaTime * 1000);
      if (el === video && !video.srcObject) poseAtContact(meta.mediaTime * 1000);
      el.requestVideoFrameCallback(onFrame);
    };
    el.requestVideoFrameCallback(onFrame);
  } else if (el === video) {
    const poll = () => {
      const s = src();
      if (s.currentTime !== lastT) { lastT = s.currentTime; sendFrame(s.srcObject ? performance.now() : lastT * 1000); }
      requestAnimationFrame(poll);
    };
    requestAnimationFrame(poll);
  }
}

// Racket hand, detected on the video. The analysis finds each hit ~2 s before the visible video shows it; when the
// visible video reaches the contact frame, the hitter is cropped at full resolution and the pose model (detector.mjs)
// votes for the wrist nearest the shuttle. Sure after 3 votes with a 2-vote lead; right-handed meanwhile (~90 %).
function poseAtContact(t) {
  contacts = contacts.filter((c) => c.t > t - 50); // missed (seek, pause) or done
  const i = contacts.findIndex((c) => Math.abs(c.t - t) <= 17); // this very frame (60 fps video)
  if (i < 0) return;
  const { p, id, box: b } = contacts.splice(i, 1)[0];
  const sx = Math.max(0, b.x - b.w * 0.6), sy = Math.max(0, b.y - b.h * 0.5);
  const sw = Math.min(video.videoWidth - sx, b.w * 2.2), sh = Math.min(video.videoHeight - sy, b.h * 1.6);
  createImageBitmap(video, sx, sy, sw, sh)
    .then((bitmap) => detector.postMessage({ type: 'pose', bitmap, id, p: [p[0] - sx, p[1] - sy] }, [bitmap]), () => {});
}
function handOf(id) {
  const v = handVotes.get(id);
  if (!v || v.R + v.L < 3 || Math.abs(v.R - v.L) < 2) return { right: true, sure: false };
  return { right: v.R > v.L, sure: true };
}
const hands = () => [0, 1, 2, 3].map((id) => handOf(id).right);

// Keep the twin LEAD seconds ahead: catch up at 2x after a seek, wait when the visible video is paused.
setInterval(() => {
  if (video.srcObject || !ahead.src || !isFinite(video.duration) || ahead.seeking) return;
  const want = Math.min(video.duration, video.currentTime + LEAD), d = ahead.currentTime - want;
  if (d > 0.5 || d < -LEAD - 1) { ahead.currentTime = video.currentTime; return; } // seek: restart from the visible frame
  if (d < -0.05) { ahead.playbackRate = 2; ahead.play().catch(() => {}); }
  else if (d > 0.15 || video.paused) ahead.pause();
  else { ahead.playbackRate = 1; ahead.play().catch(() => {}); }
}, 150);
ahead.addEventListener('seeking', resetAnalysis);
function resetAnalysis() {
  tracker.postMessage({ type: 'reset' }); tracknet.postMessage({ type: 'reset' });
  trackTL = []; playTL = []; events = []; busy = false; lastSent = lastDet = -Infinity;
}

// Result for the frame on screen: the last one at or before the visible media time (camera: the latest one).
function at(tl, maxAge) {
  if (video.srcObject) return tl[tl.length - 1];
  const now = video.currentTime * 1000;
  let i = tl.length - 1;
  while (i >= 0 && tl[i].t > now) i--;
  while (tl.length > 300 && tl[0].t < now - 3000) { tl.shift(); i--; } // keep ~3 s of history
  return i >= 0 && now - tl[i].t <= maxAge ? tl[i] : null;
}

// ---------- Sources ----------
$('file').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  stopCamera();
  video.src = ahead.src = URL.createObjectURL(f);
});
for (const b of document.querySelectorAll('.cam-btn')) b.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } }, audio: false });
    video.removeAttribute('src'); ahead.removeAttribute('src'); ahead.load(); // live: analysed directly, no twin
    video.srcObject = stream;
    video.play();
  } catch (e) {
    setStatus('Caméra indisponible : ' + e.message);
  }
});
function stopCamera() {
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}
$('recal').addEventListener('click', () => { video.pause(); startCalibration(); });
video.addEventListener('error', () => { if (!video.srcObject && video.getAttribute('src')) setStatus('Impossible de lire cette vidéo : essayez un fichier MP4 (H.264).'); });
video.addEventListener('seeking', () => { if (video.srcObject) resetAnalysis(); });
video.addEventListener('loadeddata', () => {
  view.width = video.videoWidth; view.height = video.videoHeight;
  $('empty').hidden = true;
  $('stage').style.setProperty('--ar', video.videoWidth / video.videoHeight); // the stage takes the video's shape
  if (!video.srcObject) video.pause();
  startCalibration();
});
function configure() {
  const cfg = { type: 'config', H, vw: video.videoWidth, vh: video.videoHeight, perSide: perSide || 2 };
  tracker.postMessage(cfg); detector.postMessage(cfg); tracknet.postMessage(cfg);
  resetAnalysis(); anaPlayers = []; anaOthers = []; detBusy = false; handVotes = new Map(); contacts = [];
  allShots = []; allRallies = []; calls = []; shownShots = -1; forecasts = new Map(); fcAsked = new Set();
  disp.clear();
}

// ---------- Player ----------
const fmt = (s) => (isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}` : 'direct');
const togglePlay = () => (video.paused ? video.play() : video.pause());
const skip = (d) => { if (isFinite(video.duration)) video.currentTime = Math.min(video.duration, Math.max(0, video.currentTime + d)); };
video.volume = 0; // sound off by default (the element is also muted)
$('play').addEventListener('click', togglePlay);
$('back').addEventListener('click', () => skip(-10));
$('fwd').addEventListener('click', () => skip(10));
$('seek').addEventListener('input', (e) => { if (isFinite(video.duration)) video.currentTime = e.target.value / 1000 * video.duration; });
$('vol').addEventListener('input', (e) => { video.volume = +e.target.value; video.muted = video.volume === 0; });
$('mute').addEventListener('click', () => { if (video.muted && !video.volume) video.volume = 0.5; video.muted = !video.muted; });
video.addEventListener('volumechange', () => {
  $('mute').textContent = video.muted || !video.volume ? '🔇' : '🔊';
  $('vol').value = video.muted ? 0 : video.volume;
});
video.addEventListener('play', () => { $('play').textContent = '❚❚'; });
video.addEventListener('pause', () => { $('play').textContent = '▶'; });
video.addEventListener('timeupdate', () => {
  $('time').textContent = `${fmt(video.currentTime)} / ${fmt(video.duration)}`;
  if (isFinite(video.duration)) $('seek').value = video.currentTime / video.duration * 1000;
});
// Fullscreen = the whole app (video + minimap + stats), so nothing is lost.
const toggleFullscreen = () => {
  const app = $('app');
  if (document.fullscreenElement) return document.exitFullscreen();
  (app.requestFullscreen || app.webkitRequestFullscreen).call(app);
};
$('fs').addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', () => { $('fs').textContent = document.fullscreenElement ? '🗗' : '⛶'; });
// ---------- Match statistics (S) and CSV export ----------
let statsKey = '';
const toggleStats = () => { $('stats').hidden = !$('stats').hidden; statsKey = ''; renderDashboard(); };
$('statsBtn').addEventListener('click', toggleStats);
$('statsClose').addEventListener('click', toggleStats);
function renderDashboard() {
  if ($('stats').hidden) return;
  const shots = shotsNow(), rallies = ralliesNow(), key = `${shots.length}/${rallies.length}/${perSide}`;
  if (key === statsKey) return; // only when a shot or a point was added
  statsKey = key;
  $('statsBody').innerHTML = dashboard({ shots, rallies, perSide: perSide || 1 });
}
setInterval(renderDashboard, 1000);
// one tooltip for every chart mark (data-tip), following the pointer
$('statsBody').addEventListener('pointermove', (e) => {
  const m = e.target.closest('[data-tip]'), tip = $('dtip');
  tip.hidden = !m;
  if (!m) return;
  tip.textContent = m.dataset.tip;
  tip.style.left = `${Math.min(innerWidth - tip.offsetWidth - 8, e.clientX + 14)}px`;
  tip.style.top = `${e.clientY + 14}px`;
});
$('statsBody').addEventListener('pointerleave', () => { $('dtip').hidden = true; });
$('stats').addEventListener('scroll', () => { $('dtip').hidden = true; });
$('exportCsv').addEventListener('click', () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([shotsCsv(allShots, allRallies, Hinv)], { type: 'text/csv' }));
  a.download = 'badminton-vision-coups.csv';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// the control bar follows the video's on-screen width
new ResizeObserver(([e]) => { document.querySelector('.player').style.width = `${e.contentRect.width}px`; }).observe($('stage'));

document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, button')) return;
  if (e.code === 'KeyF') toggleFullscreen();
  if (e.code === 'KeyS') toggleStats();
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  if (e.code === 'ArrowLeft') skip(-10);
  if (e.code === 'ArrowRight') skip(10);
});

// ---------- Calibration ----------
// Step 1: 1 vs 1 or 2 vs 2 (how many players per side get an id). Step 2: click the 4 court corners.
// Each player's racket hand is learnt from the video (worker.mjs), no question asked.
function startCalibration() {
  H = null; calib = []; perSide = 0;
  configure();
  $('modePick').hidden = false;
  setStatus('Calibration : combien de joueurs sur le terrain ?');
}

// Mini court for the 1 vs 1 / 2 vs 2 cards: camera side at the bottom (like the minimap), players in their half.
function courtSvg(n) {
  const S = 10, lines = [...COURT_LINES, NET_LINE].map(([a, b]) => `<line x1="${a[0] * S}" y1="${a[1] * S}" x2="${b[0] * S}" y2="${b[1] * S}"/>`).join('');
  const spots = n === 1 ? [[30.5, 100], [30.5, 34]] : [[18, 100], [43, 100], [18, 34], [43, 34]];
  const dots = spots.map(([x, y], id) => `<circle cx="${x}" cy="${y}" r="7" fill="${COLORS[id]}"/><text x="${x}" y="${y + 3}">${id + 1}</text>`).join('');
  return `<svg viewBox="-4 -4 69 142" role="img" aria-label="${n === 1 ? 'Un joueur' : 'Deux joueurs'} de chaque côté du filet">
    <rect x="0" y="0" width="61" height="134" fill="#1f6f4a"/><g stroke="#fff" stroke-width="1">${lines}</g>${dots}</svg>`;
}
for (const el of document.querySelectorAll('.mode-card .court')) el.innerHTML = courtSvg(+el.dataset.players);
$('modePick').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-per-side]');
  if (!b) return;
  perSide = +b.dataset.perSide;
  $('mode').textContent = perSide === 1 ? 'Simple' : 'Double';
  $('modePick').hidden = true;
  setStatus(calibHint());
});
const calibHint = () => `Calibration : cliquez le ${CORNER_HINTS[calib.length]} du terrain (lignes extérieures de double) — ${calib.length + 1}/4`;
view.addEventListener('click', (e) => {
  if (H || !perSide || !view.width) return;
  const r = view.getBoundingClientRect();
  calib.push([(e.clientX - r.left) * view.width / r.width, (e.clientY - r.top) * view.height / r.height]);
  if (calib.length < 4) return setStatus(calibHint());
  H = homography(calib, COURT_CORNERS);
  Hinv = homography(COURT_CORNERS, calib);
  configure();
  if (!video.srcObject) ahead.currentTime = video.currentTime; // analyse from the frame on screen
  setStatus('Analyse en cours…');
  video.play();
});

// ---------- Drawing (every display frame, independent of the worker's pace) ----------
function draw() {
  if (view.width) {
    st = (H && at(trackTL, 150)) || EMPTY;
    players = (H && at(playTL, 500)?.players) || [];
    vis = H ? shuttleAt() : null;
    const key = vis ? (vis.est ? 'g' : 'o') : '-';
    if (st !== shown || key !== shownKey) { shown = st; shownKey = key; renderStats(); }
    ctx.clearRect(0, 0, view.width, view.height);
    smoothPlayers();
    drawOverlay();
    drawMinimap();
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

// Players are detected ~12 times/s: ease the displayed boxes toward the last detection instead of jumping.
function smoothPlayers() {
  const ids = new Set(players.map((p) => p.id));
  for (const id of disp.keys()) if (!ids.has(id)) disp.delete(id);
  for (const p of players) {
    const d = disp.get(p.id);
    if (!d) { disp.set(p.id, { ...p, court: [...p.court] }); continue; }
    for (const k of ['x', 'y', 'w', 'h']) d[k] += (p[k] - d[k]) * 0.35;
    d.court[0] += (p.court[0] - d.court[0]) * 0.35;
    d.court[1] += (p.court[1] - d.court[1]) * 0.35;
  }
}

function drawCourt(c, map, color, width) {
  c.strokeStyle = color; c.lineWidth = width;
  c.beginPath();
  for (const [a, b] of [...COURT_LINES, NET_LINE]) { c.moveTo(...map(a)); c.lineTo(...map(b)); }
  c.stroke();
}

function drawOverlay() {
  const lw = Math.max(2, view.width / 500);
  if (!H) {
    ctx.fillStyle = '#ffd23d'; ctx.font = `600 ${Math.round(view.width / 60)}px Inter, sans-serif`;
    calib.forEach(([x, y], i) => { ctx.beginPath(); ctx.arc(x, y, lw * 3, 0, 7); ctx.fill(); ctx.fillText(i + 1, x + 8, y - 8); });
    return;
  }
  drawCourt(ctx, (p) => project(Hinv, p), 'rgba(255,255,255,0.35)', lw / 2);
  ctx.font = `600 ${Math.round(view.width / 60)}px Inter, sans-serif`;
  for (const p of disp.values()) {
    ctx.strokeStyle = ctx.fillStyle = COLORS[p.id]; ctx.lineWidth = lw;
    ctx.strokeRect(p.x, p.y, p.w, p.h);
    // hit / shot info rides on the player's own label so it follows the box
    const f = labelFor(p.id);
    ctx.fillText(`J${p.id + 1}${f}`, p.x, p.y - 6);
  }
  const c = callNow();
  if (c) { // line call at the landing spot, same plain text style as the player labels
    ctx.fillStyle = CALL[c.call].color;
    ctx.fillText(CALL[c.call].text, ...project(Hinv, c.at).map((v, i) => v + (i ? -10 : 10)));
  }
  if (vis) {
    ctx.globalAlpha = vis.est ? 0.6 : 1; // slightly faded = estimated position (TrackNet missed this frame)
    ctx.fillStyle = '#ffd23d'; ctx.strokeStyle = '#000'; ctx.lineWidth = lw / 2;
    ctx.beginPath(); ctx.arc(...vis.p, lw * 2.5, 0, 7); ctx.fill(); ctx.stroke();
    ctx.globalAlpha = 1;
  }
}

// Virtual umpire: the line call of a landing, shown for 1.5 s of video time from the landing.
const CALL = { in: { text: 'IN', color: '#4dff88' }, out: { text: 'OUT', color: '#ff4d4d' }, net: { text: 'FILET', color: '#ffd23d' } };
function callNow() {
  const now = video.srcObject ? (trackTL[trackTL.length - 1]?.t ?? 0) : video.currentTime * 1000;
  return calls.findLast((c) => c.t <= now && now - c.t < 1500 && CALL[c.call]);
}

// Latest hit/shot event of that player shown for 1.5 s of video time ("J1 frappe", "J1 · Smash · 210 km/h").
function labelFor(id) {
  const now = video.srcObject ? (trackTL[trackTL.length - 1]?.t ?? 0) : video.currentTime * 1000;
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e.t > now || e.id !== id) continue;
    return now - e.t < 1500 ? ` ${e.text}` : '';
  }
  return '';
}

const M = 30, SC = Math.min((mini.width - 2 * M) / COURT.w, (mini.height - 2 * M) / COURT.l);
const mm = ([x, y]) => [mini.width / 2 + (x - COURT.w / 2) * SC, mini.height / 2 + (y - COURT.l / 2) * SC];
const clampCourt = ([x, y]) => [Math.min(COURT.w + 0.5, Math.max(-0.5, x)), Math.min(COURT.l + 0.5, Math.max(-0.5, y))];

function drawMinimap() {
  mctx.fillStyle = '#1f6f4a'; mctx.fillRect(0, 0, mini.width, mini.height);
  drawCourt(mctx, mm, '#fff', 1.5);
  mctx.fillStyle = '#fff'; mctx.font = '600 10px Inter, sans-serif'; mctx.textAlign = 'center';
  mctx.fillText('FOND (loin de la caméra)', mini.width / 2, 12);
  mctx.fillText('AVANT (côté caméra)', mini.width / 2, mini.height - 5);
  mctx.textAlign = 'start';
  if (!H) return drawCalibGuide();
  for (const s of shotsNow().slice(0, 8)) {
    mctx.fillStyle = COLORS[s.id] + '88';
    mctx.beginPath(); mctx.arc(...mm(s.court), 3, 0, 7); mctx.fill();
  }
  const c = callNow();
  if (c) {
    mctx.strokeStyle = CALL[c.call].color; mctx.lineWidth = 2;
    mctx.beginPath(); mctx.arc(...mm(clampCourt(c.at)), 6, 0, 7); mctx.stroke();
  }
  const f = forecastNow();
  if (f) { // likely landing of the next stroke: 1-sigma ellipse of ShuttleNet's prediction, dashed, colour of the next hitter
    const e = ellipse(f.sx, f.sy, f.corr), [cx, cy] = mm(clampCourt(f.mean));
    mctx.save(); mctx.setLineDash([4, 3]); mctx.strokeStyle = nextColor(f.shot); mctx.lineWidth = 2;
    mctx.beginPath(); mctx.ellipse(cx, cy, Math.max(4, e.rx * SC), Math.max(4, e.ry * SC), e.angle, 0, 2 * Math.PI); mctx.stroke();
    mctx.restore();
  }
  const sp = shuttleOnCourt();
  if (sp) {
    mctx.fillStyle = '#ffd23d'; mctx.strokeStyle = '#000'; mctx.lineWidth = 1;
    mctx.beginPath(); mctx.arc(...mm(clampCourt(sp)), 4.5, 0, 7); mctx.fill(); mctx.stroke();
  }
  mctx.font = '600 9px Inter, sans-serif'; mctx.textAlign = 'center'; mctx.textBaseline = 'middle';
  for (const p of disp.values()) {
    const [x, y] = mm(clampCourt(p.court));
    mctx.fillStyle = COLORS[p.id]; mctx.beginPath(); mctx.arc(x, y, 7, 0, 7); mctx.fill();
    mctx.fillStyle = '#000'; mctx.fillText(p.id + 1, x, y + 0.5);
  }
  mctx.textAlign = 'start'; mctx.textBaseline = 'alphabetic';
}

// Shuttle on the court plan. Projecting its image position onto the floor is wrong as soon as it flies (the higher,
// the deeper it looks, hence the jumps), so for a known shot it is placed on the hitter -> landing path instead,
// moving like a real shuttle: fast after the hit, slow at the end. The analysis runs ahead, so the shot is known.
function shuttleOnCourt() {
  if (video.srcObject) return st.shuttle && project(H, st.shuttle); // live: shots are only known afterwards
  const now = video.currentTime * 1000;
  const s = allShots.find((x) => x.t <= now && now <= x.t1);
  if (!s) return null;
  const D = Math.hypot(s.to[0] - s.court[0], s.to[1] - s.court[1]);
  const f = flightFraction(D, (s.t1 - s.t) / 1000, (now - s.t) / 1000);
  return [s.court[0] + (s.to[0] - s.court[0]) * f, s.court[1] + (s.to[1] - s.court[1]) * f];
}

// Shuttle on the video. The analysis runs ahead, so around the frame on screen both the previous AND the next
// TrackNet detections are known: the shuttle is drawn between them. That makes it move smoothly at the screen's
// 60 fps (analysis is 30 fps) and bridges gaps of up to 0.5 s where TrackNet lost it (motion blur, behind a player),
// like TrackNetV3's own InpaintNet step.
function shuttleAt() {
  if (video.srcObject) return st.shuttle ? { p: st.shuttle, est: st.ghost } : null; // live: no future frames
  const tl = trackTL, now = video.currentTime * 1000, seen = (s) => s && s.shuttle && !s.ghost;
  let i = tl.length - 1;
  while (i >= 0 && tl[i].t > now) i--;
  let a = i, b = i + 1;
  while (a >= 0 && !seen(tl[a])) a--;
  while (b < tl.length && !seen(tl[b])) b++;
  const A = tl[a], B = tl[b];
  if (A && B && B.t - A.t <= BRIDGE_MS
      && Math.hypot(B.shuttle[0] - A.shuttle[0], B.shuttle[1] - A.shuttle[1]) < view.width * 0.25) { // same flight
    const f = (now - A.t) / (B.t - A.t);
    return { p: [A.shuttle[0] + (B.shuttle[0] - A.shuttle[0]) * f, A.shuttle[1] + (B.shuttle[1] - A.shuttle[1]) * f], est: B.t - A.t > 70 };
  }
  if (A && now - A.t <= 50) return { p: A.shuttle, est: false };
  return st.shuttle ? { p: st.shuttle, est: true } : null; // the worker's short extrapolation
}

// Numbered corners: green = done, pulsing yellow = the one to click now.
function drawCalibGuide() {
  const pulse = 9 + 2 * Math.sin(performance.now() / 150);
  mctx.font = '700 11px Inter, sans-serif'; mctx.textAlign = 'center'; mctx.textBaseline = 'middle';
  COURT_CORNERS.forEach((c, i) => {
    const [x, y] = mm(c), next = i === calib.length;
    mctx.fillStyle = i < calib.length ? '#4dff88' : next ? '#ffd23d' : 'rgba(255,255,255,0.35)';
    mctx.beginPath(); mctx.arc(x, y, next ? pulse : 8, 0, 7); mctx.fill();
    mctx.fillStyle = '#000'; mctx.fillText(i + 1, x, y + 0.5);
  });
  mctx.textAlign = 'start'; mctx.textBaseline = 'alphabetic';
}

function renderStats() {
  $('count').textContent = players.length;
  $('track').textContent = !H ? '—' : !vis ? 'Volant non visible' : vis.est ? 'Volant (position estimée)' : 'Volant suivi';
  $('track').className = !vis ? '' : vis.est ? 'ghost' : 'ok';
  // racket hand learnt from the video; "?" until enough overhead shots were seen (right-handed assumed meanwhile)
  const hand = (id) => { const h = handOf(id); return h.sure ? ` ${h.right ? 'droitier' : 'gaucher'}` : ' ?'; };
  $('chips').innerHTML = players.map((p) => `<span class="chip" style="--c:${COLORS[p.id]}" title="Main détectée sur la vidéo">J${p.id + 1}${hand(p.id)}</span>`).join('');
  const shots = shotsNow(), last = shots[0];
  $('speed').textContent = last && last.kmh ? `${last.kmh} km/h` : '—';
  $('shot').textContent = last ? `J${last.id + 1} · ${last.name}` : '—';
  if (shots.length !== shownShots) { // rebuild the list only when a shot is added (30 last shots)
    shownShots = shots.length;
    $('history').innerHTML = shots.slice(0, 30).map((s) =>
      `<li><span class="dot" style="background:${COLORS[s.id]}"></span>J${s.id + 1} · ${s.name} <b>${s.kmh ? s.kmh + ' km/h' : '—'}</b></li>`).join('');
  }
  renderScore();
  renderForecast();
}

// ---------- Next stroke (ShuttleNet) ----------
// Each finished shot sends its rally so far to the forecaster (from the serve on: the model reads the serve and the
// return first). The analysis runs ~2 s ahead, so the forecast is ready when the shot is hit on screen.
function requestForecasts() {
  for (const s of allShots) {
    if (fcAsked.has(s.t)) continue;
    fcAsked.add(s.t);
    const rally = allShots.filter((x) => x.rally === s.rally && x.round <= s.round).sort((a, b) => a.round - b.round);
    if (!rally[0].serve) continue; // the start of the rally was not seen
    forecaster.postMessage({ key: s.t, shots: rally.map(({ court, to, type }) => ({ court, to, type })) });
  }
}
const nextShot = (s) => allShots.find((x) => x.rally === s.rally && x.round === s.round + 1);
// The forecast for the shot in the air: from the last hit on screen until the reply is played (or 3 s).
function forecastNow() {
  const now = nowMs(), s = shotsNow()[0];
  const f = s && forecasts.get(s.t), n = s && nextShot(s);
  return f && !(n && n.t <= now) && now - s.t < 3000 ? { ...f, shot: s } : null;
}
// Who plays next: the side opposite the hitter (singles: that player; doubles: their team).
const nextLabel = (s) => (perSide === 2 ? (s.court[1] >= COURT.net ? 'J3/J4' : 'J1/J2') : `J${2 - s.id}`);
const nextColor = (s) => COLORS[perSide === 2 ? (s.court[1] >= COURT.net ? 2 : 0) : 1 - s.id];
function renderForecast() {
  const f = forecastNow();
  const html = f ? `<b>${nextLabel(f.shot)}</b> va jouer : ${f.top.map((c) => `${SHOT_NAMES[c.type] || c.type} <b>${Math.round(c.p * 100)} %</b>`).join(' · ')}`
    : 'Prédiction à partir du 4ᵉ coup d\'un échange vu depuis le service.';
  if ($('fcNext').innerHTML !== html) $('fcNext').innerHTML = html;
  // checked against what was really played, once the reply is on screen
  let n = 0, top1 = 0, top3 = 0;
  for (const [t, f2] of forecasts) {
    const s = allShots.find((x) => x.t === t), real = s && nextShot(s);
    if (!real || real.t > nowMs()) continue;
    n++; top1 += f2.top[0].type === real.type; top3 += f2.top.some((c) => c.type === real.type);
  }
  $('fcAcc').textContent = n ? `Prédictions justes : ${Math.round(top1 / n * 100)} % (dans le top 3 : ${Math.round(top3 / n * 100)} %) sur ${n} coups` : '—';
}

// Score: rallies whose end is on screen or past. J1 (and J2) = camera side, the score follows the players when ends change.
function renderScore() {
  const sb = scoreBoard(ralliesNow());
  const team = (side) => (perSide === 2 ? (side === 'near' ? [0, 1] : [2, 3]) : [side === 'near' ? 0 : 1])
    .map((id) => `<span class="chip" style="--c:${COLORS[id]}">J${id + 1}</span>`).join('');
  const html = `${team('near')} <b>${sb.near.pts} – ${sb.far.pts}</b> ${team('far')}`;
  if ($('score').innerHTML !== html) $('score').innerHTML = html;
  $('sets').textContent = `Set ${sb.set} · sets ${sb.near.sets}–${sb.far.sets}`;
}

// Shots / rallies whose time is on screen or past: the analysis already knows the next ones, don't spoil them.
const nowMs = () => (video.srcObject ? Infinity : video.currentTime * 1000);
const shotsNow = () => allShots.filter((s) => s.t <= nowMs());
const ralliesNow = () => allRallies.filter((r) => r.t <= nowMs() && r.winner);

function setStatus(msg) { $('status').textContent = msg; }
