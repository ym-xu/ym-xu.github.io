// Home page: a marker that goes through the key phrases of the bio in reading
// order, one phrase per bar of the beat, then rests for a bar. Each phrase is
// drawn left to right at a hand's pace (the look is in site.css). The text
// itself never moves. Hovering a phrase pins the marker there, and it carries
// on from that phrase. Respects reduced motion, and stops while the hero is
// off-screen or the tab is hidden.
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

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var hero = document.querySelector('[data-beat]');
  if (!hero) return;
  var hls = Array.prototype.slice.call(hero.querySelectorAll('.hl'));
  if (!hls.length) return;

  var bpm = Math.min(200, Math.max(60, parseFloat(hero.getAttribute('data-beat')) || 128));
  var bar = 4 * 60000 / bpm;              // one tick = a bar of four beats

  // ---- the score: phrase 0, 1, … n-1, then a bar's rest (-1) ----
  var n = hls.length, tick = 0, cur = -1, hover = -1, timer = null, onScreen = true;

  // the stroke takes as long as a hand would: about 1.1 px per ms, within limits
  function pace(el) {
    var w = 0, r = el.getClientRects();
    for (var i = 0; i < r.length; i++) w += r[i].width;
    el.style.setProperty('--sweep', Math.round(Math.min(520, Math.max(220, w * 0.9))) + 'ms');
  }
  function show(k) {
    if (k === cur) return;
    if (cur >= 0) hls[cur].classList.remove('on');
    if (k >= 0) { pace(hls[k]); hls[k].classList.add('on'); }
    cur = k;
  }
  function step() {
    if (hover >= 0) return;
    show(tick < n ? tick : -1);
    tick = (tick + 1) % (n + 1);
  }
  function start() { if (!timer && onScreen && !document.hidden) timer = setInterval(step, bar); }
  function stop() { if (timer) { clearInterval(timer); timer = null; } }

  hls.forEach(function (el, k) {
    el.addEventListener('mouseenter', function () { hover = k; show(k); });
    el.addEventListener('mouseleave', function () { hover = -1; tick = k + 1; });
  });

  if (reduceMotion) { show(0); return; }

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      onScreen = entries[0].isIntersecting;
      if (onScreen) start(); else stop();
    }).observe(hero);
  }
  document.addEventListener('visibilitychange', function () { if (document.hidden) stop(); else start(); });
  // let the name finish inking in before the highlighter starts playing; the
  // bar is counted from here, so the first phrase gets a whole bar too
  setTimeout(function () { stop(); step(); start(); }, 900);
})();
