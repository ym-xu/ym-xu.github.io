// A short burst of fireworks when an exercise or a "想一想" question is answered correctly.
//   window.courseFireworks(x, y, big)    x, y in viewport pixels; big adds two more bursts
// Drawn on one fixed, click-through canvas that is removed when the last spark fades.
// Nothing is drawn for readers who ask for reduced motion.
(function () {
  'use strict';
  var COLORS = ['#f59e0b', '#ef4444', '#10b981', '#3b82f6', '#a855f7', '#ec4899', '#facc15'];
  var canvas = null, ctx = null, sparks = [], running = false;

  function ensureCanvas() {
    if (canvas) return;
    var dpr = window.devicePixelRatio || 1;
    canvas = document.createElement('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:60';
    canvas.width = Math.round(window.innerWidth * dpr);
    canvas.height = Math.round(window.innerHeight * dpr);
    document.body.appendChild(canvas);
    ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
  }
  function burst(x, y, count, speed) {
    for (var i = 0; i < count; i++) {
      var angle = Math.random() * Math.PI * 2, v = speed * (0.35 + 0.65 * Math.random());
      sparks.push({
        x: x, y: y, vx: Math.cos(angle) * v, vy: Math.sin(angle) * v - speed * 0.25,
        life: 1, fade: 0.012 + Math.random() * 0.014, size: 1.6 + Math.random() * 1.8,
        color: COLORS[(Math.random() * COLORS.length) | 0]
      });
    }
  }
  function frame() {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    sparks = sparks.filter(function (s) {
      s.vx *= 0.985; s.vy = s.vy * 0.985 + 0.07;          // drag and gravity
      s.x += s.vx; s.y += s.vy; s.life -= s.fade;
      if (s.life <= 0) return false;
      ctx.globalAlpha = Math.max(0, s.life);
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2); ctx.fill();
      return true;
    });
    if (sparks.length) { requestAnimationFrame(frame); return; }
    running = false;
    if (canvas) { canvas.remove(); canvas = null; ctx = null; }
  }
  window.courseFireworks = function (x, y, big) {
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    ensureCanvas();
    burst(x, y, big ? 70 : 34, big ? 6.5 : 4.6);
    if (big) {
      setTimeout(function () { if (canvas) burst(x - 90, y - 50, 50, 5.5); }, 220);
      setTimeout(function () { if (canvas) burst(x + 90, y - 80, 50, 5.5); }, 440);
    }
    if (!running) { running = true; requestAnimationFrame(frame); }
  };
})();
