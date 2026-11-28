import assert from 'node:assert/strict';
import fs from 'node:fs';
import { homography, project, findBlobs, isHit, classifyShot, trackCandidates, isShuttle, launchKmh, flightFraction, isCross, handModifiers, shotName, shotFeatures, shotProbs, lineCall, scoreBoard, shotsCsv, CSV_COLUMNS, COURT_CORNERS } from './logic.mjs';

// Homography: the 4 clicked corners must land exactly on the court corners.
const img = [[420, 210], [860, 215], [1150, 690], [130, 680]];
const H = homography(img, COURT_CORNERS);
img.forEach((p, i) => project(H, p).forEach((v, k) => assert.ok(Math.abs(v - COURT_CORNERS[i][k]) < 1e-6)));
const back = project(homography(COURT_CORNERS, img), project(H, [600, 400]));
assert.ok(Math.abs(back[0] - 600) < 1e-6 && Math.abs(back[1] - 400) < 1e-6);

// Blobs: two separate blobs, one too big is dropped.
const w = 10, h = 10, m = new Uint8Array(w * h);
m[11] = m[12] = 1;                                   // area 2
for (let y = 5; y < 9; y++) for (let x = 5; x < 9; x++) m[y * w + x] = 1; // area 16
const blobs = findBlobs(m, w, h, 1, 10);
assert.equal(blobs.length, 1);
assert.deepEqual([blobs[0].x, blobs[0].y], [1.5, 1]);

// Hits
assert.ok(isHit([1, 0], [-1, 0]));
assert.ok(!isHit([1, 0], [0.9, 0.1]));
assert.ok(isHit([1, 0], [2, 0]));
assert.ok(isHit([0, 0], [1, 1]));

// Shots
const base = { rel: 0.5, hitterNetDist: 5, endNetDist: 5, rising: false, kmh: 30 };
assert.equal(classifyShot({ ...base, rel: 0, kmh: 200 }), 'smash');
assert.equal(classifyShot({ ...base, rel: 0, rising: true }), 'clear');
assert.equal(classifyShot({ ...base, rel: 0, endNetDist: 1 }), 'drop');
assert.equal(classifyShot({ ...base, hitterNetDist: 1, endNetDist: 1 }), 'net shot');
assert.equal(classifyShot({ ...base, hitterNetDist: 1 }), 'push');
assert.equal(classifyShot({ ...base, kmh: 90 }), 'drive');
assert.equal(classifyShot(base), 'push');
assert.equal(classifyShot({ ...base, hitterNetDist: 1, kmh: 110 }), 'rush');
assert.equal(classifyShot({ ...base, hitterNetDist: 1, kmh: 110, rising: true }), 'lob'); // lifted from the net, not killed

// Tracker: a shuttle flying at constant speed among random flickering noise is found, noise is not.
let cands = [], rnd = 1;
const rand = () => ((rnd = (rnd * 16807) % 2147483647) / 2147483647);
for (let f = 0; f < 10; f++) {
  const t = f * 33, blobs = [[100 + f * 30, 200 - f * 10]];
  for (let k = 0; k < 4; k++) blobs.push([rand() * 1000, rand() * 600]);
  cands = trackCandidates(cands, blobs, t, 20);
}
const shuttles = cands.filter((c) => isShuttle(c, 0.3));
assert.equal(shuttles.length, 1);
assert.deepEqual(shuttles[0].path.at(-1).p, [370, 110]);
// A sharp 90° turn inside the gate is flagged (a full reversal leaves the gate and starts a new candidate instead).
const c = [{ path: [{ p: [0, 0], t: 0 }, { p: [30, 0], t: 33 }], v: [30 / 33, 0], seen: 33, seg: 0 }];
trackCandidates(c, [[30, 30]], 66, 40);
assert.ok(c[0].turn);

// Launch speed: a smash covering 8 m in 0.5 s left the racket at ~150 km/h; slow drift ≈ D/T.
assert.ok(Math.abs(launchKmh(8, 0.5) - 152) <= 2);
assert.ok(Math.abs(launchKmh(0.1, 1) - 0.36) < 1);

// Cross / backhand / names
assert.ok(isCross(1, 5));            // left side -> right side
assert.ok(!isCross(1, 2.5));         // stays on the left half
assert.ok(!isCross(2.8, 3.4));       // crosses the centre line but barely moves
assert.deepEqual(handModifiers('lob', -0.5, true, 0.5), { backhand: true, aroundhead: false });    // right-hander, left side
assert.deepEqual(handModifiers('lob', 0.5, true, 0.5), { backhand: false, aroundhead: false });    // forehand
assert.deepEqual(handModifiers('clear', -0.5, true, 0), { backhand: false, aroundhead: true });    // above the head, left
assert.deepEqual(handModifiers('smash', -0.5, true, 0.3), { backhand: true, aroundhead: false });  // side contact: backhand smash
assert.deepEqual(handModifiers('push', -0.5, true, 0), { backhand: true, aroundhead: false });     // never around-the-head
assert.deepEqual(handModifiers('lob', 0.5, false, 0.5), { backhand: true, aroundhead: false });    // left-hander, right side
assert.equal(shotName('smash'), 'Smash');
assert.equal(shotName('smash', { cross: true }), 'Cross smash');
assert.equal(shotName('net shot', { cross: true }), 'Cross-court net shot');
assert.equal(shotName('clear', { aroundhead: true }), 'Around-the-head clear');
assert.equal(shotName('short service', { cross: true }), 'Short serve');
assert.equal(shotName('rush'), 'Kill');
assert.equal(shotName('defensive return lob', { backhand: true }), 'Backhand defensive lob');
assert.equal(shotName('drop', { backhand: true }), 'Backhand drop');
assert.equal(shotName('drop', { backhand: true, cross: true }), 'Backhand cross drop');
assert.equal(shotName('smash', { backhand: true, cross: true }), 'Backhand cross smash');

// Flight: starts at 0, ends at 1, and covers more than half the distance in the first half of the time (drag).
assert.equal(flightFraction(10, 1, 0), 0);
assert.ok(Math.abs(flightFraction(10, 1, 1) - 1) < 1e-9);
assert.ok(flightFraction(10, 1, 0.5) > 0.6);
assert.equal(shotName('long service', { backhand: true }), 'Backhand long serve');
assert.equal(shotName('rush', { cross: true }), 'Cross kill');

// Shot model: the JS forward pass gives the same probabilities as scikit-learn (vectors exported with the model)
const model = JSON.parse(fs.readFileSync(new URL('./models/shots.json', import.meta.url)));
for (const { x, p } of model.tests) shotProbs(model, x).forEach((v, i) => assert.ok(Math.abs(v - p[i]) < 1e-4, `class ${i}: ${v} vs ${p[i]}`));
// features: a far-side hitter is turned to the near side, a clear from the back to the back lands deep
const fx = shotFeatures(model, { hitter: [3, 2], opponent: [3, 11], landing: [3, 12.5], T: 1.4, round: 4, prev: 'lob', backhand: false });
assert.equal(fx[model.features.indexOf('hy')], 11.4);
assert.equal(model.classes[shotProbs(model, fx).reduce((b, v, i, a) => (v > a[b] ? i : b), 0)], 'clear');

// Umpire: near-side hitter (y > 6.7); the shuttle lands on the far side (y < 6.7)
const nearHitter = [4, 11];
assert.equal(lineCall(nearHitter, [3, 2]), 'in');
assert.equal(lineCall(nearHitter, [3, 8]), 'net');                         // stayed on the hitter's side
assert.equal(lineCall(nearHitter, [0.2, 2]), 'out');                       // singles side line (0.46 m)
assert.equal(lineCall(nearHitter, [0.2, 2], { doubles: true }), 'in');     // inside the doubles alley
assert.equal(lineCall(nearHitter, [3, -0.3]), 'out');                      // behind the baseline
assert.equal(lineCall(nearHitter, [2, 5.5], { serve: true }), 'out');      // short of the short service line
assert.equal(lineCall(nearHitter, [2, 3], { serve: true }), 'in');         // diagonal box (server on the right half)
assert.equal(lineCall(nearHitter, [4.5, 3], { serve: true }), 'out');      // straight, not diagonal
assert.equal(lineCall(nearHitter, [2, 0.3], { serve: true, doubles: true }), 'out'); // past the doubles long service line
// Score: 21-19 wins the set, then ends change; 30 caps; 11 switch in the deciding set
const rallies = (seq) => seq.split('').map((c, i) => ({ t: i, winner: c === 'n' ? 'near' : 'far' }));
let sb = scoreBoard(rallies('n'.repeat(20) + 'f'.repeat(19) + 'n'));      // team A 21-19
assert.deepEqual([sb.set, sb.near.sets, sb.far.sets, sb.near.pts], [2, 0, 1, 0]); // A won the set and is now far
sb = scoreBoard(rallies('n'.repeat(20) + 'f'.repeat(20) + 'nf'.repeat(9) + 'n')); // 29-29 then 30
assert.equal(sb.far.sets, 1); // A (now far) took the set at 30-29
sb = scoreBoard(rallies('n'.repeat(21) + 'n'.repeat(21) + 'f'.repeat(10)));       // B wins set 2 (B is near again? no)
assert.deepEqual([sb.near.sets + sb.far.sets, sb.set], [2, 3]);

// CSV export: ShuttleSet columns, one row per shot, score and point winner on the rally's last shot
const I = homography(COURT_CORNERS, COURT_CORNERS); // identity calibration: pixels = metres
const csvShots = [
  { t: 1000, t1: 1900, id: 0, kmh: 120, type: 'short service', name: 'Short serve', court: [4, 11], to: [2, 4], opp: [2, 3], rally: 1, round: 1, backhand: true, aroundhead: false, serve: true },
  { t: 1900, t1: 2600, id: 1, kmh: null, type: 'lob', name: 'Lob', court: [2, 4], to: [3, 12], opp: [4, 11], rally: 1, round: 2, backhand: false, aroundhead: false, serve: false },
];
const csv = shotsCsv(csvShots, [{ t: 2600, rally: 1, winner: 'near', reason: 'out', at: [3, 14] }], I).trim().split('\n').map((l) => l.split(','));
const col = (row, name) => csv[row][CSV_COLUMNS.indexOf(name)];
assert.equal(csv.length, 3);
assert.deepEqual([col(1, 'player'), col(1, 'server'), col(1, 'type'), col(1, 'backhand'), col(1, 'frame_num')], ['A', 'A', 'short service', '1', '30']);
assert.deepEqual([col(2, 'player'), col(2, 'lose_reason'), col(2, 'getpoint_player'), col(2, 'roundscore_A'), col(2, 'roundscore_B')], ['B', 'out', 'A', '1', '0']);
assert.deepEqual([col(2, 'landing_x'), col(2, 'landing_y')], ['3.0', '12.0']);

console.log('logic ok');
