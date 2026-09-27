// Patterns for ryanlagasse.github.io. Everything here is a cellular automaton.
(() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const rgb = (name, a) => `rgba(${css(name)},${a})`;

  // Deterministic PRNG so a post's pattern is the same every visit.
  const prng = (seed) => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const hash = (s) => { let h = 2166136261; for (const ch of s) h = Math.imul(h ^ ch.charCodeAt(0), 16777619); return h >>> 0; };

  const fit = (c) => {
    const dpr = Math.min(devicePixelRatio || 1, 2), w = c.clientWidth, h = c.clientHeight;
    c.width = Math.max(1, Math.round(w * dpr)); c.height = Math.max(1, Math.round(h * dpr));
    const x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); return [x, w, h];
  };

  /* ---------- theme toggle ---------- */
  const redrawers = [];
  const btn = document.querySelector('.theme');
  if (btn) btn.addEventListener('click', () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
    redrawers.forEach(f => f());
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => redrawers.forEach(f => f()));

  /* ---------- elementary (1D) cellular automaton ----------
     <canvas data-ca="rule-or-auto" data-seed="text" data-cell="px" data-start="single|random" data-reveal> */
  const drawCA = (c) => {
    const seedText = c.dataset.seed || 'seed';
    const r = prng(hash(seedText));
    const nice = [30, 45, 73, 90, 105, 110, 150, 169, 225];
    const rule = c.dataset.ca === 'auto' ? nice[hash(seedText) % nice.length] : Number(c.dataset.ca);
    const cell = Number(c.dataset.cell || 6);
    const [x, W, H] = fit(c);
    const cols = Math.ceil(W / cell), rows = Math.ceil(H / cell);
    let row = new Uint8Array(cols);
    if (c.dataset.start === 'single') row[cols >> 1] = 1; else for (let i = 0; i < cols; i++) row[i] = r() < 0.5 ? 1 : 0;
    const grid = [row];
    for (let y = 1; y < rows; y++) {
      const next = new Uint8Array(cols);
      for (let i = 0; i < cols; i++) {
        const l = row[(i - 1 + cols) % cols], m = row[i], rr = row[(i + 1) % cols];
        next[i] = (rule >> ((l << 2) | (m << 1) | rr)) & 1;
      }
      grid.push(next); row = next;
    }
    const paint = (upto) => {
      x.clearRect(0, 0, W, H);
      for (let y = 0; y < upto; y++) {
        // fade from accent at the top to ink further down
        const t = y / Math.max(1, rows - 1);
        x.fillStyle = t < 0.18 ? rgb('--trail', 0.85) : rgb('--cell', 0.14 + 0.5 * (1 - t));
        for (let i = 0; i < cols; i++) if (grid[y][i]) x.fillRect(i * cell + 0.5, y * cell + 0.5, cell - 1, cell - 1);
      }
      c.setAttribute('aria-label', `Rule ${rule} cellular automaton`);
      const cap = c.parentElement && c.parentElement.querySelector('[data-ca-caption]');
      if (cap) cap.textContent = `rule ${rule}`;
    };
    if ('reveal' in c.dataset && !reduced && !c._revealed) {
      c._revealed = true; let y = 0;
      const step = () => { y = Math.min(rows, y + 2); paint(y); if (y < rows) requestAnimationFrame(step); };
      step();
    } else paint(rows);
  };
  document.querySelectorAll('canvas[data-ca]').forEach(c => {
    drawCA(c);
    const again = () => { c._revealed = true; drawCA(c); };
    redrawers.push(again); addEventListener('resize', again);
  });

  /* ---------- site mark: rule 90 from a single cell, a tiny Sierpinski ---------- */
  document.querySelectorAll('.mark canvas').forEach(c => {
    const draw = () => {
      const [x, W] = fit(c); const n = 9, s = W / n; x.clearRect(0, 0, W, W);
      let row = new Uint8Array(n); row[n >> 1] = 1;
      for (let y = 0; y < n; y++) {
        x.fillStyle = y < 2 ? rgb('--trail', 1) : rgb('--cell', 0.85);
        row.forEach((v, i) => { if (v) x.fillRect(i * s, y * s, s * 0.86, s * 0.86); });
        const next = new Uint8Array(n);
        for (let i = 0; i < n; i++) next[i] = (row[i - 1] || 0) ^ (row[i + 1] || 0);
        row = next;
      }
    };
    draw(); redrawers.push(draw);
  });

  /* ---------- Conway's Game of Life, with fading trails, drawable ---------- */
  const life = document.getElementById('life');
  if (life) {
    const cell = 9;
    let ctx, W, H, cols, rows, a, b, age, running = true, visible = true, last = 0;
    const genEl = document.getElementById('life-gen');
    let gen = 0;
    const seed = () => {
      for (let i = 0; i < a.length; i++) { a[i] = Math.random() < 0.22 ? 1 : 0; age[i] = a[i] ? 0 : 99; }
      gen = 0;
      // settle the soup so the first frame already has structure, not noise
      for (let k = 0; k < 36; k++) step(true);
    };
    const size = () => {
      [ctx, W, H] = fit(life);
      cols = Math.ceil(W / cell); rows = Math.ceil(H / cell);
      a = new Uint8Array(cols * rows); b = new Uint8Array(cols * rows); age = new Uint8Array(cols * rows);
      seed(); draw();
    };
    const step = (warm) => {
      let alive = 0;
      for (let y = 0; y < rows; y++) {
        const up = ((y - 1 + rows) % rows) * cols, mid = y * cols, dn = ((y + 1) % rows) * cols;
        for (let x = 0; x < cols; x++) {
          const l = (x - 1 + cols) % cols, r = (x + 1) % cols;
          const n = a[up + l] + a[up + x] + a[up + r] + a[mid + l] + a[mid + r] + a[dn + l] + a[dn + x] + a[dn + r];
          const i = mid + x, v = a[i] ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
          b[i] = v; age[i] = v ? 0 : Math.min(99, age[i] + 1); alive += v;
        }
      }
      [a, b] = [b, a]; gen++;
      if (!warm && alive < cols * rows * 0.035) seed();
    };
    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      const cellC = css('--cell'), trailC = css('--trail');
      for (let i = 0; i < a.length; i++) {
        const x = (i % cols) * cell, y = ((i / cols) | 0) * cell;
        if (a[i]) { ctx.fillStyle = `rgba(${cellC},0.78)`; ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2); }
        else if (age[i] < 14) { ctx.fillStyle = `rgba(${trailC},${0.42 * (1 - age[i] / 14)})`; ctx.fillRect(x + 2, y + 2, cell - 4, cell - 4); }
      }
      if (genEl) genEl.textContent = String(gen).padStart(4, '0');
    };
    const loop = (ts) => {
      if (running && visible && ts - last > (reduced ? 900 : 110)) { step(); draw(); last = ts; }
      requestAnimationFrame(loop);
    };
    // drawing with the pointer: paint a small plus of live cells
    let down = false;
    const paint = (e) => {
      const r = life.getBoundingClientRect();
      const cx = Math.floor((e.clientX - r.left) / cell), cy = Math.floor((e.clientY - r.top) / cell);
      [[0,0],[1,0],[-1,0],[0,1],[0,-1]].forEach(([dx, dy]) => {
        const x = (cx + dx + cols) % cols, y = (cy + dy + rows) % rows, i = y * cols + x; a[i] = 1; age[i] = 0;
      });
      draw();
    };
    life.addEventListener('pointerdown', (e) => { down = true; life.setPointerCapture(e.pointerId); paint(e); });
    life.addEventListener('pointermove', (e) => { if (down) paint(e); });
    life.addEventListener('pointerup', () => { down = false; });
    const pauseBtn = document.getElementById('life-pause');
    if (pauseBtn) pauseBtn.addEventListener('click', () => { running = !running; pauseBtn.textContent = running ? 'pause' : 'play'; });
    const resetBtn = document.getElementById('life-reset');
    if (resetBtn) resetBtn.addEventListener('click', () => { seed(); draw(); });
    new IntersectionObserver(([en]) => { visible = en.isIntersecting; }).observe(life);
    addEventListener('resize', size);
    redrawers.push(draw);
    size(); requestAnimationFrame(loop);
  }
})();
