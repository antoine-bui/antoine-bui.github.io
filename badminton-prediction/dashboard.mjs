// Match dashboard (idea from CoachAI's visualization platform): built from the shots and rallies already shown on screen.
// Plain HTML + inline SVG. Player colours come from CSS variables --p1..--p4 (chart steps of the overlay hues,
// validated for light and dark surfaces); every mark also carries its J1-J4 label, so colour is never the only cue.
import { COURT, SHOT_NAMES, scoreBoard } from './logic.mjs';

const pc = (id) => `var(--p${id + 1})`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const tile = (label, value, sub = '') => `<div class="dtile"><div class="label">${label}</div><div class="dval">${value}</div><div class="dsub">${sub}</div></div>`;

export function dashboard({ shots, rallies, perSide }) {
  const ids = perSide === 2 ? [0, 1, 2, 3] : [0, 1];
  const byRally = new Map();
  for (const s of shots) byRally.set(s.rally, (byRally.get(s.rally) || 0) + 1);
  const lengths = [...byRally.values()];
  const fast = shots.filter((s) => s.kmh).sort((a, b) => b.kmh - a.kmh)[0];
  const tiles = [
    tile('Échanges terminés', rallies.length),
    tile('Coups par échange', lengths.length ? (lengths.reduce((a, b) => a + b, 0) / lengths.length).toFixed(1) : '—', 'en moyenne'),
    tile('Échange le plus long', lengths.length ? `${Math.max(...lengths)} coups` : '—'),
    tile('Frappe la plus rapide', fast ? `${fast.kmh} km/h` : '—', fast ? `J${fast.id + 1} · ${esc(fast.name)}` : ''),
  ].join('');
  if (!shots.length) return `<div class="dtiles">${tiles}</div><p class="dempty">Les statistiques se remplissent pendant la lecture, au fil des coups analysés.</p>`;
  return `<div class="dtiles">${tiles}</div>
    <section class="dcard dwide"><h3>Évolution du score</h3>${scoreChart(rallies, perSide)}</section>
    <section class="dcard"><h3>Types de coups</h3><div class="dsmall">${ids.map((id) => typesChart(shots, id)).join('')}</div></section>
    <section class="dcard"><h3>Zones de chute</h3><p class="dnote">Où tombent les coups de chaque joueur, chez l'adversaire (filet en bas).</p><div class="dsmall">${ids.map((id) => zonesChart(shots, id)).join('')}</div></section>
    <section class="dcard dwide"><h3>Origine des points</h3>${pointsTable(rallies, perSide)}</section>`;
}

// Score of the current set: one step line per side (J1 = camera side), a tooltip per rally.
function scoreChart(rallies, perSide) {
  const { history, set, nearTeam } = scoreBoard(rallies.filter((r) => r.winner));
  const pts = history.filter((h) => h.set === set);
  if (!pts.length) return '<p class="dempty">Pas encore de point dans ce set.</p>';
  // wide viewBox so a full-width card keeps the text near its real size; score = steps from 0-0
  const W = 1200, Hh = 190, L = 34, R = 60, T = 10, B = 26, n = pts.length;
  const max = Math.max(5, ...pts.map((h) => Math.max(h.a, h.b)));
  const x = (i) => L + (i / n) * (W - L - R), y = (v) => T + (1 - v / max) * (Hh - T - B);
  const near = (h) => (nearTeam === 0 ? h.a : h.b), far = (h) => (nearTeam === 0 ? h.b : h.a);
  const line = (f) => `M${x(0)},${y(0)}` + pts.map((h, i) => `H${x(i + 1).toFixed(1)}V${y(f(h)).toFixed(1)}`).join('');
  const nearId = 0, farId = perSide === 2 ? 2 : 1, last = pts[n - 1];
  const grid = [0, Math.round(max / 2), max].map((v) => `<line class="dgrid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="dtick" x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join('');
  const hits = pts.map((h, i) => `<rect class="dhit" x="${x(i + 0.5)}" y="${T}" width="${(W - L - R) / n}" height="${Hh - T - B}" data-tip="Point ${i + 1} · J${nearId + 1} ${near(h)} – ${far(h)} J${farId + 1}"/>`).join('');
  return `<svg class="dchart" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Score du set ${set}">${grid}
    <path d="${line(near)}" fill="none" stroke="${pc(nearId)}" stroke-width="2"/>
    <path d="${line(far)}" fill="none" stroke="${pc(farId)}" stroke-width="2"/>
    <text class="dlab" x="${x(n) + 6}" y="${y(near(last)) + 4}">J${nearId + 1} ${near(last)}</text>
    <text class="dlab" x="${x(n) + 6}" y="${y(far(last)) + (near(last) === far(last) ? 16 : 4)}">J${farId + 1} ${far(last)}</text>
    <text class="dtick" x="${(W + L - R) / 2}" y="${Hh - 4}" text-anchor="middle">points du set ${set}</text>${hits}</svg>
    <div class="dlegend"><span><i style="background:${pc(nearId)}"></i>J${nearId + 1}${perSide === 2 ? '–J2' : ''} (côté caméra)</span><span><i style="background:${pc(farId)}"></i>J${farId + 1}${perSide === 2 ? '–J4' : ''} (côté opposé)</span></div>`;
}

// One small horizontal bar chart per player: their 7 most played types + "Autres", count labels at the bar end.
function typesChart(shots, id) {
  const mine = shots.filter((s) => s.id === id), counts = new Map();
  for (const s of mine) { const k = SHOT_NAMES[s.type] || s.type; counts.set(k, (counts.get(k) || 0) + 1); }
  let rows = [...counts].sort((a, b) => b[1] - a[1]);
  if (rows.length > 8) rows = [...rows.slice(0, 7), ['Autres', rows.slice(7).reduce((a, r) => a + r[1], 0)]];
  const max = Math.max(1, ...rows.map((r) => r[1])), W = 260, rowH = 18, L = 104;
  const bars = rows.map(([k, v], i) => {
    const w = Math.max(2, (v / max) * (W - L - 30)), yy = i * rowH;
    return `<g data-tip="J${id + 1} · ${esc(k)} : ${v} coup${v > 1 ? 's' : ''} (${Math.round(v / mine.length * 100)} %)">
      <text class="dtick" x="${L - 6}" y="${yy + 12}" text-anchor="end">${esc(k)}</text>
      <rect x="${L}" y="${yy + 3}" width="${w}" height="${rowH - 6}" rx="2" fill="${pc(id)}"/>
      <text class="dval2" x="${L + w + 4}" y="${yy + 12}">${v}</text><rect class="dhit" x="0" y="${yy}" width="${W}" height="${rowH}"/></g>`;
  }).join('');
  return `<figure><figcaption><i style="background:${pc(id)}"></i>J${id + 1} · ${mine.length} coup${mine.length > 1 ? 's' : ''}</figcaption>
    ${rows.length ? `<svg class="dchart" viewBox="0 0 ${W} ${rows.length * rowH}" role="img" aria-label="Types de coups de J${id + 1}">${bars}</svg>` : '<p class="dempty">—</p>'}</figure>`;
}

// Landing zones on the receiving half, seen from the hitter (net at the bottom): 3 depths x 3 lanes, count per cell,
// fill strength = share of the player's shots (one hue: the player's).
function zonesChart(shots, id) {
  const mine = shots.filter((s) => s.id === id && s.to), cells = Array(9).fill(0);
  for (const s of mine) {
    const far = s.court[1] < COURT.net; // turn the court so the hitter is on the near side (keeps their left/right)
    const [x, y] = far ? [COURT.w - s.to[0], COURT.l - s.to[1]] : s.to;
    const depth = COURT.net - y; // metres past the net on the receiving side
    if (depth < 0) continue; // did not cross the net
    const row = depth < 2.2 ? 2 : depth < 4.5 ? 1 : 0, col = Math.min(2, Math.max(0, Math.floor(x / (COURT.w / 3))));
    cells[row * 3 + col]++;
  }
  const max = Math.max(1, ...cells), cw = 52, ch = 38, names = ['fond', 'milieu', 'filet'], lanes = ['gauche', 'centre', 'droite'];
  const rects = cells.map((v, i) => {
    const r = Math.floor(i / 3), c = i % 3, a = v ? 0.15 + 0.75 * (v / max) : 0;
    return `<g data-tip="J${id + 1} · ${names[r]} ${lanes[c]} : ${v} coup${v > 1 ? 's' : ''}">
      <rect x="${c * cw + 1}" y="${r * ch + 1}" width="${cw - 2}" height="${ch - 2}" rx="3" fill="${pc(id)}" fill-opacity="${a.toFixed(2)}" class="dzone"/>
      <text class="dval2" x="${c * cw + cw / 2}" y="${r * ch + ch / 2 + 4}" text-anchor="middle">${v || ''}</text></g>`;
  }).join('');
  return `<figure><figcaption><i style="background:${pc(id)}"></i>J${id + 1}</figcaption>
    <svg class="dchart dcourt" viewBox="0 0 ${cw * 3} ${ch * 3 + 12}" role="img" aria-label="Zones de chute des coups de J${id + 1}">${rects}
    <line class="dnet" x1="0" x2="${cw * 3}" y1="${ch * 3 + 4}" y2="${ch * 3 + 4}"/><text class="dtick" x="${cw * 1.5}" y="${ch * 3 + 12}" text-anchor="middle">filet</text></svg></figure>`;
}

// Points won per side, split by how: own winner (shuttle IN), opponent fault (OUT / net), not seen (decided by the next serve).
function pointsTable(rallies, perSide) {
  const sides = [['near', perSide === 2 ? 'J1–J2 (côté caméra)' : 'J1 (côté caméra)'], ['far', perSide === 2 ? 'J3–J4 (côté opposé)' : 'J2 (côté opposé)']];
  const row = ([side, label]) => {
    const won = rallies.filter((r) => r.winner === side), c = (f) => won.filter(f).length;
    return `<tr><th scope="row">${label}</th><td>${won.length}</td><td>${c((r) => r.reason === 'in')}</td><td>${c((r) => r.reason === 'out')}</td><td>${c((r) => r.reason === 'net')}</td><td>${c((r) => r.reason === '?')}</td></tr>`;
  };
  return `<table class="dtable"><thead><tr><th></th><th>Points gagnés</th><th>Coup gagnant (IN)</th><th>Faute adverse : dehors</th><th>Faute adverse : filet</th><th>Non vu*</th></tr></thead>
    <tbody>${sides.map(row).join('')}</tbody></table>
    <p class="dnote">* Fin d'échange hors champ ou coupée au montage : le point est attribué au joueur qui sert ensuite (au badminton, le gagnant sert).</p>`;
}
