// Home page: a translucent highlighter that snaps (never slides) between key
// phrases of the bio, on a musical beat — a steady phrase, then a fast
// "glitch" burst, then a rest. The text itself never moves.
// Hovering a phrase pins the highlight there. Respects reduced motion, and
// stops while the hero is off-screen or the tab is hidden.
(function () {
  // Email: the page only carries an encoded string (reversed, XOR 0x2a, hex);
  // it is decoded on a real click, never at load, so scrapers find nothing.
  var mail = document.getElementById('home-email');
  if (mail) {
    mail.addEventListener('click', function () {
      var c = mail.getAttribute('data-c') || '', s = '';
      for (var i = 0; i < c.length; i += 2) s += String.fromCharCode(parseInt(c.substr(i, 2), 16) ^ 0x2a);
      var addr = s.split('').reverse().join('');
      if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(addr)) return;
      if (navigator.clipboard) {
        navigator.clipboard.writeText(addr).then(function () {
          var old = mail.textContent;
          mail.textContent = 'address copied';
          setTimeout(function () { mail.textContent = old; }, 1800);
        }, function () {});
      }
      window.location.href = 'mailto:' + addr;
    });
  }

  // (Scroll reveals and the reading-progress bar live in site.js, site-wide.)
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var hero = document.querySelector('[data-beat]');
  if (!hero) return;
  var hls = Array.prototype.slice.call(hero.querySelectorAll('.hl'));
  if (!hls.length) return;

  var reduce = reduceMotion;
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
  // let the name finish inking in before the highlighter starts playing
  setTimeout(function () { step(); start(); }, 900);
})();
