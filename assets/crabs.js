// Release the crabs. Click the crab in the header; crabs burst out, fall, and wander.
(() => {
  const btn = document.querySelector('.crab-btn');
  if (!btn) return;
  const base = btn.dataset.crabs;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

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
  const clear = document.createElement('button');
  clear.className = 'crab-clear'; clear.type = 'button'; clear.hidden = true; document.body.appendChild(clear);
  const crabs = [];
  let running = false;

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
        grounded: false, held: false, idle: 0,
      };
      grab(c);
      crabs.push(c);
    }
    while (crabs.length > max) { const old = crabs.shift(); old.el.remove(); if (old.bubble) old.bubble.remove(); }
    update();
    if (!running) { running = true; requestAnimationFrame(tick); }
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
      if (Math.hypot(c.vx, c.vy) > 7) yippee(c);   // a real throw, not a gentle drop
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

  const tick = () => {
    if (!crabs.length) { running = false; return; }
    if (document.hidden) { requestAnimationFrame(tick); return; }
    const W = innerWidth, H = innerHeight;
    for (const c of crabs) {
      if (c.held) continue;
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
      } else {
        c.vy += G; c.vx *= 0.995;
      }
      c.x += c.vx; c.y += c.vy;
      if (c.y > floor) {
        c.y = floor;
        if (c.vy > 2.5) { c.vy *= -0.32; c.vx *= 0.7; } else { c.vy = 0; c.grounded = true; }
      }
      if (c.y < -c.s * c.top - 200) c.vy = Math.abs(c.vy) * 0.5;
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
      c.el.style.transform = `translate3d(${c.x}px, ${c.y}px, 0) scaleX(${face})`;
      if (c.bubble) {
        const now = performance.now();
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
    requestAnimationFrame(tick);
  };

  const update = () => { clear.hidden = !crabs.length; clear.textContent = `clear crabs · ${crabs.length}`; };
  const reset = () => { crabs.splice(0).forEach(c => { c.el.remove(); if (c.bubble) c.bubble.remove(); }); update(); };
  btn.addEventListener('click', spawn);
  clear.addEventListener('click', reset);
  addEventListener('keydown', (e) => { if (e.key === 'Escape' && crabs.length) reset(); });
  // a link ending in #crabs arrives with crabs already out
  if (location.hash === '#crabs') { spawn(); setTimeout(spawn, 350); setTimeout(spawn, 700); }
})();
