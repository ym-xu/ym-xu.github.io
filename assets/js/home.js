// Home page: a translucent highlighter that snaps (never slides) between key
// phrases of the bio, on a musical beat — a steady phrase, then a fast
// "glitch" burst, then a rest. The text itself never moves.
// Hovering a phrase pins the highlight there. Respects reduced motion, and
// stops while the hero is off-screen or the tab is hidden.
(function () {
  // Email: assembled on click so the address is never one string in the HTML.
  var mail = document.getElementById('home-email');
  if (mail) {
    mail.addEventListener('click', function (e) {
      e.preventDefault();
      window.location.href = 'mailto:' + this.dataset.u + String.fromCharCode(64) + this.dataset.d;
    });
  }

  var hero = document.querySelector('[data-beat]');
  if (!hero) return;
  var hls = Array.prototype.slice.call(hero.querySelectorAll('.hl'));
  if (!hls.length) return;

  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var bpm = Math.min(200, Math.max(60, parseFloat(hero.getAttribute('data-beat')) || 128));
  var quarter = 60000 / bpm / 4;          // one tick = a sixteenth note

  // ---- the score: [phrase index, length in ticks, glitch?] ----
  var n = hls.length, seq = [], last = -1, seed = 7;
  var lens = [8, 4, 4, 6, 4, 8, 6, 4, 8, 12];
  for (var i = 0; i < n; i++) { seq.push([i, lens[i % lens.length], 0]); last = i; }
  function rand(k) { seed = (seed * 16807) % 2147483647; return seed % k; }
  function pick() { var j = rand(n); if (j === last) j = (j + 1) % n; last = j; return j; }
  for (var g = 0; g < 14; g++) seq.push([pick(), 1, 1]);        // the burst
  [2, 2, 2, 4].forEach(function (l) { seq.push([pick(), l, 1]); });
  seq.push([-1, 4, 0]);                                         // a rest
  var timeline = [];
  seq.forEach(function (e) { for (var t = 0; t < e[1]; t++) timeline.push(e); });

  var tick = 0, cur = -1, hover = -1, timer = null, onScreen = true;

  function show(k) {
    if (k === cur) return;
    if (cur >= 0) hls[cur].classList.remove('on');
    if (k >= 0) hls[k].classList.add('on');
    cur = k;
  }
  function step() {
    if (hover >= 0) return;
    var e = timeline[tick];
    show(e[0]);
    hero.classList.toggle('glitch', !!e[2]);
    tick = (tick + 1) % timeline.length;
  }
  function start() { if (!timer && onScreen && !document.hidden) timer = setInterval(step, quarter); }
  function stop() { if (timer) { clearInterval(timer); timer = null; } }

  hls.forEach(function (el, k) {
    el.addEventListener('mouseenter', function () { hover = k; hero.classList.remove('glitch'); show(k); });
    el.addEventListener('mouseleave', function () { hover = -1; });
  });

  if (reduce) { show(0); return; }

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      onScreen = entries[0].isIntersecting;
      if (onScreen) start(); else stop();
    }).observe(hero);
  }
  document.addEventListener('visibilitychange', function () { if (document.hidden) stop(); else start(); });
  step();
  start();
})();
