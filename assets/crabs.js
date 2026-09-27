// Release the crabs. Click the crab in the header; crabs burst out, fall, and wander.
// Once they're out: send a beach wave, drop a little nuke, or clear them.
(() => {
  const btn = document.querySelector('.crab-btn');
  if (!btn) return;
  const base = btn.dataset.crabs;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const clamp01 = (t) => (t < 0 ? 0 : t > 1 ? 1 : t);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (t) => { t = clamp01(t); return t * t * (3 - 2 * t); };
  const easeOut = (t) => 1 - (1 - clamp01(t)) ** 2;
  const rnd = (a, b) => a + Math.random() * (b - a);

  // [file, display px, visible top, visible bottom (fractions of the box), spawn weight]. Just the computer crab.
  const SPRITES = [
    ['code.gif',          62, 0.388, 1.000, 34],
  ];
  const total = SPRITES.reduce((s, p) => s + p[4], 0);
  const pick = () => { let r = Math.random() * total; for (const p of SPRITES) { r -= p[4]; if (r <= 0) return p; } return SPRITES[0]; };

  const G = 0.55;
  // fewer, smaller crabs on narrow screens so the pile doesn't bury the page
  const budget = () => { const w = innerWidth; return { batch: reduced ? 6 : Math.round(Math.min(16, Math.max(6, w / 80))), max: Math.round(Math.max(30, Math.min(140, w / 9))), scale: w < 600 ? 0.72 : 1 }; };

  const layer = document.createElement('div'); layer.className = 'crab-layer'; document.body.appendChild(layer);

  // effects canvas sits behind the crabs so they ride on the water.
  // the falling nuke is drawn on a low-res buffer and scaled up, so it matches the pixel crabs.
  const PS = 3;
  const fx = document.createElement('canvas'); fx.className = 'crab-fx'; fx.setAttribute('aria-hidden', 'true'); layer.appendChild(fx);
  const pix = document.createElement('canvas');
  let fctx, pctx, FW = 0, FH = 0;
  const fitFx = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2); FW = innerWidth; FH = innerHeight;
    fx.width = Math.round(FW * dpr); fx.height = Math.round(FH * dpr);
    fctx = fx.getContext('2d'); fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    pix.width = Math.ceil(FW / PS); pix.height = Math.ceil(FH / PS); pctx = pix.getContext('2d');
  };
  fitFx(); addEventListener('resize', fitFx);

  // tool stack, bottom right, only while crabs are out
  const tools = document.createElement('div'); tools.className = 'crab-tools'; tools.hidden = true;
  const tool = (label, title) => { const b = document.createElement('button'); b.type = 'button'; b.className = 'crab-tool'; b.textContent = label; b.title = title; tools.appendChild(b); return b; };
  const waveBtn = tool('wave', 'send a wave');
  const boomBtn = tool('kaboom', 'drop a little nuke');
  const clear = tool('clear crabs', 'remove all crabs');
  document.body.appendChild(tools);

  const crabs = [];
  const waves = [], nukes = [], booms = [], bits = [];
  let running = false, waveId = 0;
  const start = () => { if (!running) { running = true; requestAnimationFrame(tick); } };

  const spawn = () => {
    const r = btn.getBoundingClientRect();
    const ox = r.left + r.width / 2, oy = r.top + r.height / 2;
    const { batch, max, scale } = budget();
    for (let i = 0; i < batch; i++) {
      const [file, px, top, bot] = pick();
      const s = px * scale * (0.75 + Math.random() * 0.5);
      const el = new Image(); el.src = base + file; el.alt = ''; el.draggable = false;
      el.style.width = el.style.height = s + 'px';
      layer.appendChild(el);
      const c = {
        el, s, top, bot,
        x: ox - s / 2, y: oy - s * (top + bot) / 2,
        vx: (Math.random() - 0.5) * (reduced ? 3 : 14),
        vy: reduced ? 0 : -(6 + Math.random() * 9),
        dir: Math.random() < 0.5 ? -1 : 1,
        speed: 0.3 + Math.random() * 1.1,
        grounded: false, held: false, idle: 0, hit: -1, rot: 0, vr: 0,
      };
      grab(c);
      crabs.push(c);
    }
    while (crabs.length > max) { const old = crabs.shift(); old.el.remove(); if (old.bubble) old.bubble.remove(); }
    update();
    start();
  };

  // drag and fling
  const grab = (c) => {
    let lastX = 0, lastY = 0, offX = 0, offY = 0, lastT = 0;
    c.el.addEventListener('pointerdown', (e) => {
      e.preventDefault(); c.held = true; c.el.classList.add('held'); c.el.setPointerCapture(e.pointerId);
      offX = e.clientX - c.x; offY = e.clientY - c.y; lastX = e.clientX; lastY = e.clientY; c.vx = c.vy = 0;
    });
    c.el.addEventListener('pointermove', (e) => {
      if (!c.held) return;
      c.vx = (e.clientX - lastX) * 0.9; c.vy = (e.clientY - lastY) * 0.9; lastX = e.clientX; lastY = e.clientY; lastT = performance.now();
      c.x = e.clientX - offX; c.y = e.clientY - offY;
    });
    const drop = () => {
      if (!c.held) return; c.held = false; c.grounded = false; c.el.classList.remove('held');
      if (performance.now() - lastT > 80) c.vx = c.vy = 0;   // paused before letting go: a drop, not a throw
      if (Math.hypot(c.vx, c.vy) > 7) { yippee(c); c.vr = c.vx * 0.012; }   // a real throw, not a gentle drop
    };
    c.el.addEventListener('pointerup', drop); c.el.addEventListener('pointercancel', drop);
  };

  // speech bubble that rides along with a thrown crab
  const yippee = (c) => {
    if (c.bubble) c.bubble.remove();
    const b = document.createElement('div'); b.className = 'crab-bubble'; b.textContent = 'yippeeee!';
    layer.appendChild(b); c.bubble = b; c.bubbleUntil = 0;   // 0 = still airborne; timer starts on landing
    requestAnimationFrame(() => b.classList.add('on'));
  };
  const dropBubble = (c) => {
    const b = c.bubble; if (!b) return; c.bubble = null;
    b.classList.remove('on'); setTimeout(() => b.remove(), 250);
  };
  // lots of crabs get launched at once; only some of them shout, or it's bubble soup
  const maybeYippee = (c, odds) => { if (Math.random() < odds) yippee(c); };
  const isDark = () => {
    const t = document.documentElement.dataset.theme;
    return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  };
  const spray = (x, y, n, vx0, vx1, vy0, vy1) => {
    for (let i = 0; i < n; i++) bits.push({ x: x + rnd(-6, 6), y: y + rnd(-4, 4), vx: rnd(vx0, vx1), vy: rnd(vy0, vy1), g: 0.28, s: rnd(1.5, 3.5), t0: performance.now(), life: rnd(450, 900), c: Math.random() < 0.8 ? '#ffffff' : '#cfeaf2' });
  };

  /* ================= the beach wave =================
     Four acts, measured along the wave's direction of travel (u, from its origin edge):
     swell (rises and steepens) -> break (the lip curls over and crashes)
     -> bore (frothy whitewater surges up the beach) -> backwash (a thin sheet drains back). */
  const sendWave = () => {
    waves.push({ dir: waveId % 2 ? -1 : 1, id: ++waveId, t0: performance.now(), dur: reduced ? 5600 : 4400, A: Math.min(240, FH * 0.36), maxF: 0, splashed: false, seed: Math.random() * 1000 });
    start();
  };
  const waveState = (w, p) => {
    const L = FW, A = w.A;
    if (p < 0.42) {
      const q = p / 0.42;
      const c = lerp(-0.25 * L, 0.44 * L, smooth(q) * 0.55 + q * 0.45);
      return { mode: 'swell', q, c, lead: c, H: A * (0.38 + 0.62 * q), wB: 0.24 * L, wF: lerp(0.2, 0.055, q) * L, T: 0.13 * A, curl: 0 };
    }
    if (p < 0.56) {
      const q = (p - 0.42) / 0.14;
      const c = lerp(0.44 * L, 0.56 * L, q);
      return { mode: 'break', q, c, lead: c, H: A * (1 - 0.18 * q), wB: 0.24 * L, wF: 0.05 * L, T: 0.13 * A, curl: q };
    }
    if (p < 0.82) {
      const r = (p - 0.56) / 0.26;
      const F = lerp(0.61 * L, 1.04 * L, easeOut(r));
      return { mode: 'bore', r, F, lead: F, Hb: A * (0.72 * (1 - r) ** 1.6 + 0.12), T: 0.13 * A * (1 - 0.3 * r) };
    }
    const s = (p - 0.82) / 0.18;
    const F = lerp(1.04 * L, 0.66 * L, smooth(s));
    return { mode: 'back', s, F, lead: F, T: A * 0.09 * (1 - s) + 2, fade: 1 - smooth(s) };
  };
  // water height above the floor at u
  const hAt = (w, st, u, now) => {
    const L = FW;
    let h;
    if (st.mode === 'swell' || st.mode === 'break') {
      const d = u - st.c;
      const g = d < 0 ? Math.exp(-((d / st.wB) ** 2)) : Math.exp(-((d / st.wF) ** 2));
      const sheet = st.T * clamp01((st.c + st.wF * 2 - u) / (0.1 * L));
      h = Math.max(st.H * g, sheet);
    } else if (st.mode === 'bore') {
      if (u > st.F) return 0;
      const back = st.F - u;
      h = Math.sqrt(clamp01(back / (0.045 * L))) * (st.T + (st.Hb - st.T) * Math.exp(-((back / (0.28 * L)) ** 2)));
    } else {
      if (u > st.F) return 0;
      h = st.T * Math.sqrt(clamp01((st.F - u) / (0.04 * L))) * (1 + 0.15 * Math.sin(u / 23 + w.seed));
    }
    if (h > 3) h += 1.6 * Math.sin(u / 17 - now / 140 + w.seed) + 1.1 * Math.sin(u / 7.3 + now / 90);   // ripples
    return Math.max(0, h);
  };
  // how fast the water moves at u, in px per frame along the travel direction
  const flowAt = (w, st, v, u) => {
    if (st.mode === 'swell') return v * 0.55 * Math.exp(-(((u - st.c) / (st.wB * 0.8)) ** 2));
    if (st.mode === 'break') return v * 1.1 * Math.exp(-(((u - st.c) / (st.wB * 0.6)) ** 2));
    if (st.mode === 'bore') return u <= st.F ? v * (0.35 + 0.65 * Math.exp(-(((st.F - u) / (0.22 * FW)) ** 2))) : 0;
    return u <= st.F ? v * 0.9 : 0;
  };
  const lipOf = (st) => {
    const tipU = st.c + st.wF * (0.6 + 1.8 * st.curl);
    const tipH = st.H * (0.95 - 0.8 * st.curl * st.curl);   // the lip throws forward and falls
    return { tipU, tipH };
  };

  const stepWave = (w, now, H) => {
    const p = (now - w.t0) / w.dur;
    const st = waveState(w, p);
    const v = waveState(w, Math.min(1, p + 16 / w.dur)).lead - st.lead;
    const X = (u) => (w.dir > 0 ? u : FW - u);
    if (st.mode === 'bore') w.maxF = Math.max(w.maxF, st.F);
    // spray off the lip, a big splash when it lands, and spit off the whitewater front
    if (st.mode === 'break') {
      const { tipU, tipH } = lipOf(st);
      if (!reduced) spray(X(tipU), H - tipH, 2, w.dir * 1, w.dir * 4, -6, -2);
      if (st.curl > 0.92 && !w.splashed) { w.splashed = true; spray(X(tipU), H - st.T, reduced ? 12 : 46, w.dir * -2, w.dir * 9, -13, -4); }
    }
    if (st.mode === 'bore' && st.r < 0.8 && Math.random() < 0.6) spray(X(st.F - 10), H - st.Hb * 0.6, 1, w.dir * 2, w.dir * 5, -5, -1);

    for (const c of crabs) {
      if (c.held) continue;
      const cx = c.x + c.s / 2, u = w.dir > 0 ? cx : FW - cx;
      const h = hAt(w, st, u, now), surface = H - h, feet = c.y + c.s * c.bot;
      // the curling lip smashes into crabs and launches them, tumbling
      if (st.mode === 'break' && st.curl > 0.5 && c.hit !== w.id) {
        const { tipU, tipH } = lipOf(st);
        if (Math.abs(u - tipU) < 55 && feet > H - tipH - 30) {
          c.hit = w.id; c.grounded = false; c.idle = 0;
          c.vy = -rnd(9, 15); c.vx = w.dir * rnd(8, 14); c.vr = w.dir * rnd(0.2, 0.45);
          maybeYippee(c, 0.35); continue;
        }
      }
      // in the water: float toward the surface and get carried along
      if (h > 4 && feet > surface + 2) {
        const depth = Math.min(1, (feet - surface) / (c.s * 0.5));
        c.grounded = false; c.idle = 0;
        c.vy += -G * (1 + 0.9 * depth); c.vy *= 0.88;
        c.vx += (w.dir * flowAt(w, st, v, u) - c.vx) * 0.12;
        if (st.mode === 'bore' && Math.random() < 0.03) c.vr += w.dir * rnd(0.05, 0.2);   // tumbled by whitewater
      }
    }
    return p < 1;
  };

  const drawWave = (w, now) => {
    const p = (now - w.t0) / w.dur, st = waveState(w, p), dark = isDark();
    const X = (u) => (w.dir > 0 ? u : FW - u);
    const U = (x) => (w.dir > 0 ? x : FW - x);
    fctx.save(); fctx.globalAlpha = st.fade ?? 1;
    // body of water
    const step = 6;
    fctx.beginPath(); fctx.moveTo(0, FH);
    for (let x = 0; x <= FW + step; x += step) fctx.lineTo(x, FH - hAt(w, st, U(x), now));
    fctx.lineTo(FW + step, FH); fctx.closePath();
    const g = fctx.createLinearGradient(0, FH - w.A * 1.05, 0, FH);
    g.addColorStop(0, dark ? 'rgba(120,205,215,0.62)' : 'rgba(95,190,200,0.55)');
    g.addColorStop(0.45, dark ? 'rgba(55,140,185,0.66)' : 'rgba(45,130,175,0.6)');
    g.addColorStop(1, dark ? 'rgba(25,80,140,0.74)' : 'rgba(20,75,130,0.68)');
    fctx.fillStyle = g; fctx.fill();
    // glassy highlight along the surface
    fctx.strokeStyle = 'rgba(225,248,250,0.45)'; fctx.lineWidth = 1.5; fctx.stroke();

    if (st.mode === 'swell' || st.mode === 'break') {
      // the face goes white where it gets steep
      fctx.lineCap = 'round';
      for (let u = st.c; u < st.c + st.wF * 2.2; u += 5) {
        const h0 = hAt(w, st, u, now), h1 = hAt(w, st, u + 5, now), steep = clamp01((h0 - h1) / 5 - 0.6);
        if (steep <= 0 || h0 < 6) continue;
        fctx.strokeStyle = `rgba(255,255,255,${0.7 * steep})`; fctx.lineWidth = 2.5;
        fctx.beginPath(); fctx.moveTo(X(u), FH - h0); fctx.lineTo(X(u + 5), FH - h1); fctx.stroke();
      }
    }
    if ((st.mode === 'swell' && st.q > 0.8) || st.mode === 'break') {
      // the lip: a sheet thrown forward off the crest, with the dark barrel underneath
      const curl = st.mode === 'break' ? st.curl : 0;
      const top = FH - st.H;
      const { tipU, tipH } = st.mode === 'break' ? lipOf(st) : { tipU: st.c + st.wF * 0.6, tipH: st.H * 0.95 };
      fctx.beginPath();
      fctx.moveTo(X(st.c - st.wF * 0.4), top + 3);
      fctx.quadraticCurveTo(X(st.c + st.wF * (0.8 + 0.6 * curl)), top - st.H * 0.07 * (1 - curl), X(tipU), FH - tipH);
      fctx.quadraticCurveTo(X(st.c + st.wF * 0.7), top + st.H * 0.4, X(st.c + st.wF * 0.1), top + st.H * 0.6);
      fctx.closePath();
      fctx.fillStyle = dark ? 'rgba(110,200,212,0.8)' : 'rgba(85,180,195,0.78)'; fctx.fill();
      if (curl > 0.25) {   // the barrel: shade the inside of the curl, clipped to the lip
        fctx.save(); fctx.clip();
        const bg = fctx.createRadialGradient(X(st.c + st.wF * 0.8), top + st.H * 0.5, 2, X(st.c + st.wF * 0.8), top + st.H * 0.5, st.wF * 0.9);
        bg.addColorStop(0, `rgba(10,50,90,${0.45 * curl})`); bg.addColorStop(1, 'rgba(10,50,90,0)');
        fctx.fillStyle = bg; fctx.fillRect(0, top - st.H * 0.2, FW, st.H * 1.2);
        fctx.restore();
      }
      fctx.strokeStyle = 'rgba(255,255,255,0.9)'; fctx.lineWidth = 3; fctx.lineCap = 'round';
      fctx.beginPath(); fctx.moveTo(X(st.c - st.wF * 0.2), top + 2);
      fctx.quadraticCurveTo(X(st.c + st.wF * (0.8 + 0.6 * curl)), top - st.H * 0.07 * (1 - curl), X(tipU), FH - tipH); fctx.stroke();
    }
    if (st.mode === 'bore' || (st.mode === 'break' && st.curl > 0.85)) {
      // whitewater: churning foam, thickest at the front
      const F = st.mode === 'bore' ? st.F : lipOf(st).tipU + 20;
      const reach = FW * (st.mode === 'bore' ? 0.4 : 0.12), tick6 = Math.floor(now / 90);
      let i = 0;
      for (let u = F - reach; u <= F; u += 7, i++) {
        const dens = Math.exp(-(((F - u) / (FW * 0.16)) ** 2));
        const h = hAt(w, st, u, now); if (h < 3) continue;
        const n = 1 + Math.round(dens * 3);
        for (let k = 0; k < n; k++) {
          const hsh = Math.sin((i * 12.9898 + k * 78.233 + tick6 * 3.17 + w.seed) * 43758.5453);
          const nz = hsh - Math.floor(hsh);
          fctx.fillStyle = `rgba(255,255,255,${0.45 + 0.45 * dens})`;
          fctx.beginPath(); fctx.arc(X(u + nz * 7), FH - h + 3 + nz * 9 * (1 - dens * 0.5), 2 + dens * 7 * (0.5 + nz), 0, Math.PI * 2); fctx.fill();
        }
      }
    }
    if (st.mode === 'back') {
      // lacy foam lines on the draining sheet, and a wet sheen where the water reached
      fctx.strokeStyle = `rgba(255,255,255,0.6)`; fctx.lineWidth = 1.4;
      for (let k = 0; k < 3; k++) {
        const u0 = st.F - FW * (0.04 + 0.1 * k);
        fctx.beginPath();
        for (let u = u0 - FW * 0.18; u <= u0; u += 6) {
          const y = FH - st.T * 0.55 + 2.2 * Math.sin(u / 19 + k * 2 + w.seed) + 1.2 * Math.sin(u / 7 + k);
          u === u0 - FW * 0.18 ? fctx.moveTo(X(u), y) : fctx.lineTo(X(u), y);
        }
        fctx.stroke();
      }
      const a = X(st.F), b = X(w.maxF);
      fctx.fillStyle = dark ? 'rgba(140,190,215,0.22)' : 'rgba(80,140,175,0.18)';
      fctx.fillRect(Math.min(a, b), FH - 4, Math.abs(b - a), 4);
    }
    fctx.restore();
  };

  /* ================= kaboom: a little nuke ================= */
  const NUKE = [   // nose points down; drawn on the low-res buffer
    'kk..k..kk',
    'kfk.k.kfk',
    'kffkkkffk',
    '.kkbbbkk.',
    '..kbbbk..',
    '.kbbhbbk.',
    'kbbhbbbbk',
    'kyyyyyyyk',
    'kbbkykbbk',
    'kbhbbbbbk',
    'kbbbbbbbk',
    '.kbbbbbk.',
    '..kbbbk..',
    '...kkk...',
  ];
  const NUKE_COL = { k: '#2a2622', f: '#4a5535', b: '#6b7a4a', h: '#95a56b', y: '#f2c230' };
  const dropNuke = () => {
    let x = FW / 2;
    if (crabs.length) { const c = crabs[(Math.random() * crabs.length) | 0]; x = c.x + c.s / 2; }
    x = Math.max(60, Math.min(FW - 60, x + rnd(-60, 60)));
    nukes.push({ x, y: -50, vy: 1, t0: performance.now() });
    start();
  };
  // the explosion: the old bomb's flash, ring, and debris, at double scale
  const BOOM_MS = 600;
  const detonate = (x) => {
    const now = performance.now(), y = FH - 10;
    booms.push({ x, y, t0: now });
    const palette = [css('--accent'), css('--ink'), '#f2b544', '#f7e2a0'];
    for (let i = 0; i < 96; i++) {
      const a = Math.random() * Math.PI * 2, sp = rnd(4, 16);
      bits.push({ x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 6, g: 0.35, s: rnd(3, 8), t0: now, life: rnd(700, 1400), c: palette[i % palette.length] });
    }
    const R = Math.max(340, FW * 0.4);
    for (const c of crabs) {
      if (c.held) continue;
      const cx = c.x + c.s / 2, cy = c.y + c.s * (c.top + c.bot) / 2;
      const dx = cx - x, dy = cy - y, d = Math.hypot(dx, dy) || 1;
      if (d > R) continue;
      const f = 1 - d / R;
      c.vx += (dx / d) * 28 * f + rnd(-2, 2);
      c.vy += (dy / d) * 18 * f - 16 * f - 2;
      c.vr += rnd(-0.5, 0.5) * f;
      c.grounded = false; c.idle = 0;
      if (f > 0.3) maybeYippee(c, 0.35);
    }
    if (!reduced) document.body.animate(
      [{ transform: 'translate(0,0)' }, { transform: 'translate(-8px,5px)' }, { transform: 'translate(7px,-6px)' }, { transform: 'translate(-5px,-3px)' }, { transform: 'translate(4px,2px)' }, { transform: 'translate(0,0)' }],
      { duration: 400, easing: 'ease-out' });
  };
  const drawBoom = (k, now) => {
    const p = Math.min(1, Math.max(0, (now - k.t0) / BOOM_MS));
    fctx.fillStyle = `rgba(255,214,130,${0.55 * (1 - p)})`; fctx.beginPath(); fctx.arc(k.x, k.y, 48 + 340 * p, 0, Math.PI * 2); fctx.fill();
    fctx.strokeStyle = css('--accent'); fctx.globalAlpha = 1 - p; fctx.lineWidth = 4;
    fctx.beginPath(); fctx.arc(k.x, k.y, 60 + 560 * p, 0, Math.PI * 2); fctx.stroke(); fctx.globalAlpha = 1;
  };
  const drawNuke = (n, now) => {
    const wob = Math.sin((now - n.t0) / 90) * 0.6;
    const ox = Math.round(n.x / PS - 4.5 + wob), oy = Math.round(n.y / PS - 7);
    NUKE.forEach((row, j) => { for (let i = 0; i < row.length; i++) { const ch = row[i]; if (ch === '.') continue; pctx.fillStyle = NUKE_COL[ch]; pctx.fillRect(ox + i, oy + j, 1, 1); } });
  };

  /* ---------- drawing all effects ---------- */
  const drawFx = (now) => {
    fctx.clearRect(0, 0, FW, FH);
    for (const w of waves) drawWave(w, now);
    for (const k of booms) drawBoom(k, now);
    if (nukes.length) {
      pctx.clearRect(0, 0, pix.width, pix.height);
      for (const n of nukes) drawNuke(n, now);
      fctx.imageSmoothingEnabled = false;
      fctx.drawImage(pix, 0, 0, pix.width * PS, pix.height * PS);
    }
    for (const q of bits) {
      fctx.globalAlpha = Math.max(0, 1 - (now - q.t0) / q.life); fctx.fillStyle = q.c;
      fctx.fillRect(q.x - q.s / 2, q.y - q.s / 2, q.s, q.s);
    }
    fctx.globalAlpha = 1;
  };
  const fxBusy = () => waves.length || nukes.length || booms.length || bits.length;

  const tick = () => {
    if (!crabs.length && !fxBusy()) { fctx.clearRect(0, 0, FW, FH); running = false; return; }
    if (document.hidden) { requestAnimationFrame(tick); return; }
    const W = innerWidth, H = innerHeight, now = performance.now();

    for (let i = waves.length - 1; i >= 0; i--) if (!stepWave(waves[i], now, H)) waves.splice(i, 1);
    for (let i = nukes.length - 1; i >= 0; i--) {
      const n = nukes[i]; n.vy += G * 0.7; n.y += n.vy;
      if (n.y + 20 >= H - 2) { nukes.splice(i, 1); detonate(n.x); }
    }
    for (let i = booms.length - 1; i >= 0; i--) if (now - booms[i].t0 > BOOM_MS) booms.splice(i, 1);
    for (let i = bits.length - 1; i >= 0; i--) {
      const q = bits[i]; q.vy += q.g; q.x += q.vx; q.y += q.vy; q.vx *= 0.99;
      if (now - q.t0 > q.life || q.y > H + 20) bits.splice(i, 1);
    }

    for (const c of crabs) {
      if (c.held) { c.rot *= 0.8; continue; }
      const floor = H - c.s * c.bot;           // y at which the crab's feet touch the bottom
      if (c.grounded) {
        // wander: sometimes idle, sometimes turn, sometimes hop
        if (c.idle > 0) { c.idle--; c.vx *= 0.8; }
        else {
          c.vx += (c.dir * c.speed - c.vx) * 0.08;
          if (Math.random() < 0.006) c.dir *= -1;
          if (Math.random() < 0.004) c.idle = 40 + (Math.random() * 120) | 0;
          if (!reduced && Math.random() < 0.0035) { c.vy = -(4 + Math.random() * 6); c.grounded = false; }
        }
        // settle back upright after a tumble
        c.vr = 0; c.rot = Math.atan2(Math.sin(c.rot), Math.cos(c.rot)) * 0.75;
      } else {
        c.vy += G; c.vx *= 0.995;
        c.rot += c.vr; c.vr *= 0.985;
      }
      c.x += c.vx; c.y += c.vy;
      if (c.y > floor) {
        c.y = floor;
        if (c.vy > 2.5) { c.vy *= -0.32; c.vx *= 0.7; } else { c.vy = 0; c.grounded = true; }
      }
      if (c.y < -c.s * c.top - 220) c.vy = Math.abs(c.vy) * 0.5;
      if (c.x < -c.s * 0.1) { c.x = -c.s * 0.1; c.vx = Math.abs(c.vx) * 0.6; c.dir = 1; }
      if (c.x > W - c.s * 0.9) { c.x = W - c.s * 0.9; c.vx = -Math.abs(c.vx) * 0.6; c.dir = -1; }
    }
    // crabs shove each other so they pile up instead of overlapping perfectly
    for (let i = 0; i < crabs.length; i++) for (let j = i + 1; j < crabs.length; j++) {
      const a = crabs[i], b = crabs[j];
      const ar = a.s * 0.3, br = b.s * 0.3;
      const ax = a.x + a.s / 2, ay = a.y + a.s * (a.top + a.bot) / 2, bx = b.x + b.s / 2, by = b.y + b.s * (b.top + b.bot) / 2;
      const dx = bx - ax, dy = by - ay, d = Math.hypot(dx, dy) || 0.01, min = ar + br;
      if (d < min) {
        const push = (min - d) * 0.25, nx = dx / d, ny = dy / d;
        if (!a.held) { a.x -= nx * push; a.y -= ny * push; }
        if (!b.held) { b.x += nx * push; b.y += ny * push; }
        // the one on top rests on the one below
        if (ny > 0.5 && !a.held) { a.vy = Math.min(a.vy, 0); a.grounded = true; }
        if (ny < -0.5 && !b.held) { b.vy = Math.min(b.vy, 0); b.grounded = true; }
      }
    }
    // a crab resting on nothing starts falling again
    for (const c of crabs) if (c.grounded && c.y < H - c.s * c.bot - 1 && Math.random() < 0.15) c.grounded = false;
    for (const c of crabs) {
      const face = (Math.abs(c.vx) > 0.15 ? Math.sign(c.vx) : c.dir) < 0 ? -1 : 1;
      c.el.style.transform = `translate3d(${c.x}px, ${c.y}px, 0) rotate(${c.rot}rad) scaleX(${face})`;
      if (c.bubble) {
        if (c.grounded && !c.bubbleUntil) c.bubbleUntil = now + 900;   // linger a moment after landing
        if (c.bubbleUntil && now > c.bubbleUntil) dropBubble(c);
        else {
          const bw = c.bubble.offsetWidth, bh = c.bubble.offsetHeight;
          const bx = Math.max(4, Math.min(W - bw - 4, c.x + c.s * 0.55 - bw * 0.25));
          const by = Math.max(4, c.y + c.s * c.top - bh - 6);
          c.bubble.style.transform = `translate3d(${bx}px, ${by}px, 0)`;
        }
      }
    }
    drawFx(now);
    requestAnimationFrame(tick);
  };

  const update = () => { tools.hidden = !crabs.length; clear.textContent = `clear crabs · ${crabs.length}`; };
  const reset = () => { crabs.splice(0).forEach(c => { c.el.remove(); if (c.bubble) c.bubble.remove(); }); update(); };
  btn.addEventListener('click', spawn);
  waveBtn.addEventListener('click', sendWave);
  boomBtn.addEventListener('click', dropNuke);
  clear.addEventListener('click', reset);
  addEventListener('keydown', (e) => { if (e.key === 'Escape' && crabs.length) reset(); });
  // a link ending in #crabs arrives with crabs already out
  if (location.hash === '#crabs') { spawn(); setTimeout(spawn, 350); setTimeout(spawn, 700); }
})();
