// Fireworks when an exercise or a "想一想" question is answered correctly.
//   window.courseFireworks(x, y, big)    x, y: viewport pixels of the button; big: the last exercise of its kind
// A small rocket rises from the button and bursts, with glowing trails and a two-colour palette; a
// smaller one follows. The last exercise of a lesson gets a short show across the screen. Drawn on one
// fixed, click-through canvas that is removed when the last spark fades. Nothing is drawn for readers
// who ask for reduced motion.
(function () {
  'use strict';
  // two colours per burst, so each one looks deliberate rather than like confetti
  var PALETTES = [
    ['#ff4f8b', '#ffb3cf'], ['#ff8a00', '#ffd166'], ['#2f9bff', '#9fd8ff'],
    ['#12b8a6', '#9ff0dc'], ['#8b5cf6', '#d9c6ff'], ['#ef4444', '#ffc857']
  ];
  var GOLD = ['#e8a317', '#ffe08a'];
  var MAX_SPARKS = 1200;
  var canvas = null, ctx = null, W = 0, H = 0, dark = false;
  var rockets = [], sparks = [], pending = 0, running = false, last = 0;
  var sprites = {};

  function reduced() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }
  function isDark() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t) return t === 'dark';
    return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  }
  function rgba(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }
  // a soft glowing dot, drawn once per colour and theme
  function sprite(color) {
    var key = color + (dark ? 'd' : 'l');
    if (sprites[key]) return sprites[key];
    var r = 16, c = document.createElement('canvas');
    c.width = c.height = r * 2;
    var g = c.getContext('2d'), grad = g.createRadialGradient(r, r, 0, r, r, r);
    grad.addColorStop(0, dark ? '#ffffff' : color);
    grad.addColorStop(0.22, color);
    grad.addColorStop(0.5, rgba(color, dark ? 0.4 : 0.3));
    grad.addColorStop(1, rgba(color, 0));
    g.fillStyle = grad;
    g.fillRect(0, 0, r * 2, r * 2);
    sprites[key] = c;
    return c;
  }

  function ensureCanvas() {
    if (canvas) return;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas = document.createElement('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:60';
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    document.body.appendChild(canvas);
    ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    dark = isDark();
  }

  function add(x, y, vx, vy, color, opt) {
    if (sparks.length >= MAX_SPARKS) return;
    sparks.push({
      x: x, y: y, vx: vx, vy: vy, color: color, life: 1,
      decay: opt.decay, size: opt.size, g: opt.g == null ? 0.05 : opt.g, drag: opt.drag || 0.972,
      tl: opt.trail == null ? 5 : opt.trail, trail: [], crackle: !!opt.crackle, flash: !!opt.flash
    });
  }

  function explode(x, y, palette, kind, scale) {
    var i, a, s, n;
    add(x, y, 0, 0, palette[1], { decay: 0.11, size: 13 * scale, g: 0, trail: 0, flash: true });   // the flash
    if (kind === 'ring') {
      n = Math.round(40 * scale);
      for (i = 0; i < n; i++) {
        a = (i / n) * Math.PI * 2;
        s = 3.4 * scale;
        add(x, y, Math.cos(a) * s, Math.sin(a) * s, palette[i % 2], { decay: 0.017, size: 2.1, trail: 4 });
      }
      for (i = 0; i < n / 2; i++) {                                  // an inner, slower ring
        a = (i / (n / 2)) * Math.PI * 2 + 0.2;
        s = 1.7 * scale;
        add(x, y, Math.cos(a) * s, Math.sin(a) * s, palette[1], { decay: 0.02, size: 1.7, trail: 3 });
      }
    } else if (kind === 'willow') {
      n = Math.round(56 * scale);
      for (i = 0; i < n; i++) {
        a = Math.random() * Math.PI * 2;
        s = (1.4 + Math.random() * 2.4) * scale;
        add(x, y, Math.cos(a) * s, Math.sin(a) * s - 0.6, GOLD[i % 2], { decay: 0.0075 + Math.random() * 0.003, size: 1.7, g: 0.035, drag: 0.982, trail: 9 });
      }
    } else {                                                          // peony, or crackle: twinkles and pops at the end
      n = Math.round(70 * scale);
      for (i = 0; i < n; i++) {
        a = Math.random() * Math.PI * 2;
        s = (1.2 + Math.pow(Math.random(), 0.5) * 3.3) * scale;
        add(x, y, Math.cos(a) * s, Math.sin(a) * s, palette[Math.random() < 0.72 ? 0 : 1],
          { decay: 0.012 + Math.random() * 0.007, size: 2.3, trail: 5, crackle: kind === 'crackle' });
      }
    }
  }

  function launch(x0, y0, x1, y1, palette, kind, scale) {
    rockets.push({ x: x0, y: y0, x0: x0, y0: y0, x1: x1, y1: y1, t: 0, T: 0.5 + Math.random() * 0.15,
      palette: palette, kind: kind, scale: scale || 1, trail: [], wob: Math.random() * 6 });
  }

  function drawTrail(points, x, y, color, alpha, width) {
    if (!points.length) return;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (var i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  function frame(now) {
    var dt = Math.min(2.5, Math.max(0.2, (now - last) / 16.667));   // in 60 Hz frames, so 120 Hz screens run at the same speed
    last = now;
    ctx.clearRect(0, 0, W, H);
    ctx.lineCap = 'round';

    rockets = rockets.filter(function (r) {
      r.t += dt / 60;
      var k = Math.min(1, r.t / r.T), e = 1 - Math.pow(1 - k, 2.4);
      r.trail.push([r.x, r.y]);
      if (r.trail.length > 10) r.trail.shift();
      r.x = r.x0 + (r.x1 - r.x0) * e + Math.sin(r.t * 28 + r.wob) * 0.8;
      r.y = r.y0 + (r.y1 - r.y0) * e;
      drawTrail(r.trail, r.x, r.y, r.palette[1], 0.55, 1.6);
      ctx.globalAlpha = 1;
      ctx.drawImage(sprite(r.palette[0]), r.x - 6, r.y - 6, 12, 12);
      if (k < 1) return true;
      explode(r.x, r.y, r.palette, r.kind, r.scale);
      return false;
    });

    var born = [];
    sparks = sparks.filter(function (s) {
      if (s.tl) {
        s.trail.push([s.x, s.y]);
        if (s.trail.length > s.tl) s.trail.shift();
      }
      var drag = Math.pow(s.drag, dt);
      s.vx *= drag;
      s.vy = s.vy * drag + s.g * dt;
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      s.life -= s.decay * dt;
      if (s.life <= 0) {
        if (s.crackle) for (var j = 0; j < 3; j++) born.push([s.x, s.y, (Math.random() - 0.5) * 2.2, (Math.random() - 0.5) * 2.2]);
        return false;
      }
      var alpha = Math.min(1, s.life * 1.5);
      if (s.crackle && s.life < 0.4 && Math.random() < 0.45) alpha *= 0.15;   // twinkle before it pops
      if (s.flash) {
        ctx.globalAlpha = alpha * 0.55;
        var f = s.size * (2.2 - s.life);
        ctx.drawImage(sprite(s.color), s.x - f, s.y - f, f * 2, f * 2);
        return true;
      }
      drawTrail(s.trail, s.x, s.y, s.color, alpha * 0.5, s.size * 0.85);
      ctx.globalAlpha = alpha;
      var d = s.size * 3.6;
      ctx.drawImage(sprite(s.color), s.x - d / 2, s.y - d / 2, d, d);
      return true;
    });
    born.forEach(function (b) {
      add(b[0], b[1], b[2], b[3], dark ? '#ffffff' : '#ffb000', { decay: 0.06 + Math.random() * 0.04, size: 1.3, g: 0.03, trail: 2 });
    });
    ctx.globalAlpha = 1;

    if (rockets.length || sparks.length || pending) {
      requestAnimationFrame(frame);
      return;
    }
    running = false;
    canvas.remove();
    canvas = null;
    ctx = null;
  }

  function start() {
    if (running) return;
    running = true;
    last = performance.now();
    requestAnimationFrame(frame);
  }
  function later(ms, fn) {
    pending++;
    setTimeout(function () {
      pending--;
      if (canvas) { fn(); start(); }
    }, ms);
  }
  function pick(list) { return list[(Math.random() * list.length) | 0]; }

  // one rocket from (x, y) up to about 200 px above; if there is no room above, a burst right there
  function single(x, y, palette, kind, scale, rise) {
    var ty = y - rise;
    if (ty < 50) { explode(x, Math.max(y, 40), palette, kind, scale); return; }
    launch(x, y, x + (Math.random() - 0.5) * 70, ty, palette, kind, scale);
  }

  window.courseFireworks = function (x, y, big) {
    if (reduced()) return;
    ensureCanvas();
    var first = (Math.random() * PALETTES.length) | 0;
    if (!big) {
      single(x, y, PALETTES[first], pick(['peony', 'crackle', 'ring']), 1, 170 + Math.random() * 60);
      later(300, function () {
        single(x + (Math.random() < 0.5 ? -1 : 1) * (60 + Math.random() * 50), y, PALETTES[(first + 2) % PALETTES.length], pick(['ring', 'peony']), 0.7, 120 + Math.random() * 50);
      });
    } else {
      var kinds = ['peony', 'ring', 'willow', 'crackle', 'peony', 'ring', 'willow', 'crackle'];
      kinds.forEach(function (kind, i) {
        later(i * 280, function () {
          var tx = W * (0.12 + 0.76 * Math.random()), ty = H * (0.12 + 0.28 * Math.random());
          launch(x + (Math.random() - 0.5) * 40, y, tx, ty, PALETTES[(first + i) % PALETTES.length], kind, 1.15);
        });
      });
      later(kinds.length * 280 + 250, function () {                   // the finale: two at once
        launch(x, y, W * 0.3, H * 0.2, PALETTES[(first + 1) % PALETTES.length], 'peony', 1.35);
        launch(x, y, W * 0.7, H * 0.22, PALETTES[(first + 3) % PALETTES.length], 'crackle', 1.35);
      });
    }
    start();
  };
})();
