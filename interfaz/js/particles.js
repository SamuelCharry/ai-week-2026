const TOP_K = 10;
const LINK_DIST = 150;
const DENSITY = 10500;
const MAX_NODES = 120;
const QUERY_PERIOD = 4200;
const QUERY_FADE = 1700;

function token(name, fallback) {
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return v || fallback;
}

export function initParticles(canvas) {
  if (!canvas) return;

  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const COLOR_NODE = token("--sc-deep", "#056072");
  const COLOR_QUERY = token("--sc-orange-ink", "#d6720f");

  let nodes = [];
  let w = 0;
  let h = 0;
  let dpr = 1;
  let raf = null;
  let lastQueryAt = 0;
  let query = null;
  let inViewport = false;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      w = 0;
      h = 0;
      return;
    }

    const nextDpr = Math.min(window.devicePixelRatio || 1, 2);
    if (w === rect.width && h === rect.height && dpr === nextDpr) return;

    dpr = nextDpr;
    w = rect.width;
    h = rect.height;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    build();
    query = null;
    draw(performance.now());
  }

  function build() {
    const target = Math.min(MAX_NODES, Math.round((w * h) / DENSITY));
    nodes = Array.from({ length: target }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.16,
      vy: (Math.random() - 0.5) * 0.16,
      r: 1.8 + Math.random() * 2.2,
    }));
  }

  function emitQuery() {
    const left = Math.random() < 0.5;
    const qx = left ? w * (0.06 + Math.random() * 0.2) : w * (0.74 + Math.random() * 0.2);
    const qy = h * (0.15 + Math.random() * 0.7);

    const ranked = nodes
      .map((n, i) => ({ i, d: (n.x - qx) ** 2 + (n.y - qy) ** 2 }))
      .sort((a, b) => a.d - b.d)
      .slice(0, TOP_K)
      .map((entry) => entry.i);

    query = { x: qx, y: qy, at: performance.now(), hits: ranked };
  }

  function draw(now) {
    ctx.clearRect(0, 0, w, h);

    const hits = query ? new Set(query.hits) : null;

    let progress = 0;
    if (query) {
      progress = (now - query.at) / QUERY_FADE;
      if (progress >= 1) {
        query = null;
        progress = 0;
      }
    }

    const pulse = query ? Math.sin(Math.min(progress, 1) * Math.PI) : 0;

    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > LINK_DIST * LINK_DIST) continue;

        const closeness = 1 - Math.sqrt(d2) / LINK_DIST;
        ctx.strokeStyle = COLOR_NODE;
        ctx.globalAlpha = closeness * 0.55;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    if (query && pulse > 0.01) {
      ctx.strokeStyle = COLOR_QUERY;
      ctx.lineWidth = 1.6;
      for (const idx of query.hits) {
        const n = nodes[idx];
        if (!n) continue;
        ctx.globalAlpha = pulse * 0.75;
        ctx.beginPath();
        ctx.moveTo(query.x, query.y);
        ctx.lineTo(n.x, n.y);
        ctx.stroke();
      }

      ctx.globalAlpha = pulse;
      ctx.fillStyle = COLOR_QUERY;
      ctx.beginPath();
      ctx.arc(query.x, query.y, 4.4, 0, Math.PI * 2);
      ctx.fill();

      ctx.globalAlpha = (1 - Math.min(progress, 1)) * 0.45;
      ctx.strokeStyle = COLOR_QUERY;
      ctx.beginPath();
      ctx.arc(query.x, query.y, 10 + progress * 90, 0, Math.PI * 2);
      ctx.stroke();
    }

    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const isHit = hits ? hits.has(i) : false;
      ctx.globalAlpha = isHit ? 0.6 + pulse * 0.4 : 0.9;
      ctx.fillStyle = isHit ? COLOR_QUERY : COLOR_NODE;
      ctx.beginPath();
      ctx.arc(n.x, n.y, isHit ? n.r + pulse * 2.6 : n.r, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1;
  }

  function step(now) {
    raf = null;
    if (!canAnimate()) return;

    for (const n of nodes) {
      n.x += n.vx;
      n.y += n.vy;

      if (n.x <= 0 || n.x >= w) n.vx *= -1;
      if (n.y <= 0 || n.y >= h) n.vy *= -1;
      n.x = Math.max(0, Math.min(w, n.x));
      n.y = Math.max(0, Math.min(h, n.y));
    }

    if (now - lastQueryAt > QUERY_PERIOD) {
      emitQuery();
      lastQueryAt = now;
    }

    draw(now);
    raf = requestAnimationFrame(step);
  }

  function start() {
    if (raf !== null || !canAnimate()) return;
    lastQueryAt = performance.now();
    raf = requestAnimationFrame(step);
  }

  function canAnimate() {
    return !document.hidden && inViewport && !reduceMotion.matches && w > 0 && h > 0;
  }

  function stop() {
    if (raf === null) return;
    cancelAnimationFrame(raf);
    raf = null;
  }

  function renderStatic() {
    stop();
    query = null;
    draw(performance.now());
  }

  function apply() {
    if (document.hidden || !inViewport || !w || !h) stop();
    else if (reduceMotion.matches) renderStatic();
    else start();
  }

  resize();

  if ("IntersectionObserver" in window) {
    new IntersectionObserver(
      ([entry]) => {
        inViewport = entry.isIntersecting;
        apply();
      },
      { threshold: 0 },
    ).observe(canvas);
  } else {
    const updateViewport = () => {
      const rect = canvas.getBoundingClientRect();
      inViewport = rect.bottom > 0 && rect.top < window.innerHeight
        && rect.right > 0 && rect.left < window.innerWidth;
      apply();
    };
    window.addEventListener("scroll", updateViewport, { passive: true });
    window.addEventListener("resize", updateViewport);
    updateViewport();
  }

  document.addEventListener("visibilitychange", apply);

  reduceMotion.addEventListener("change", apply);

  if ("ResizeObserver" in window) {
    new ResizeObserver(() => {
      resize();
      apply();
    }).observe(canvas);
  }

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      resize();
      apply();
    }, 150);
  });
}
