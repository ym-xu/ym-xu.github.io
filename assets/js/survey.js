// Survey page: filter the paper list by subfield (or a pair "a+b"), by flags
// (benchmarks, people I follow) and by free text, 30 at a time. Figure 1 rows,
// subfield cards, subfield pairs and rising terms all drive the same filters.
// State lives in the URL hash (#tag=tables&q=ocr&bench=1), so views can be shared.
(function () {
  var list = document.getElementById('survey-list');
  if (!list) return;
  var cards = Array.prototype.slice.call(list.querySelectorAll('.paper'));
  var q = document.getElementById('survey-q');
  var more = document.getElementById('survey-more');
  var count = document.getElementById('survey-count');
  var empty = document.getElementById('survey-empty');
  var chips = Array.prototype.slice.call(document.querySelectorAll('.chips .chip[data-filter]'));
  var flags = Array.prototype.slice.call(document.querySelectorAll('.chips .chip[data-flag]'));
  var papersSection = document.getElementById('papers');
  var STEP = 30;
  var state = { tag: '', text: '', bench: false, watch: false, shown: STEP };
  var texts = cards.map(function (c) { return c.textContent.toLowerCase().replace(/\s+/g, ' '); });
  var tagsOf = cards.map(function (c) { return (c.getAttribute('data-tags') || '').split(' '); });

  function readHash() {
    var h = new URLSearchParams(location.hash.slice(1));
    state.tag = h.get('tag') || '';
    state.text = (h.get('q') || '').toLowerCase();
    state.bench = h.get('bench') === '1';
    state.watch = h.get('watch') === '1';
    q.value = h.get('q') || '';
  }
  function writeHash() {
    var h = new URLSearchParams();
    if (state.tag) h.set('tag', state.tag);
    if (state.text) h.set('q', state.text);
    if (state.bench) h.set('bench', '1');
    if (state.watch) h.set('watch', '1');
    var s = h.toString();
    history.replaceState(null, '', s ? '#' + s : location.pathname + location.search);
  }

  function render() {
    var need = state.tag ? state.tag.split('+') : [];
    var words = state.text.split(/\s+/).filter(Boolean);
    var match = 0, visible = 0;
    cards.forEach(function (c, i) {
      var ok = need.every(function (t) { return tagsOf[i].indexOf(t) >= 0; })
        && (!state.bench || c.getAttribute('data-bench') === 'true')
        && (!state.watch || c.getAttribute('data-watch') === 'true')
        && words.every(function (w) { return texts[i].indexOf(w) >= 0; });
      if (ok) match++;
      var show = ok && match <= state.shown;
      c.hidden = !show;
      if (show) visible++;
    });
    count.textContent = visible < match ? visible + ' of ' + match : match + (match === 1 ? ' paper' : ' papers');
    more.hidden = match <= state.shown;
    empty.hidden = match > 0;
    chips.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-filter') === state.tag)); });
    flags.forEach(function (b) { b.setAttribute('aria-pressed', String(!!state[b.getAttribute('data-flag')])); });
  }
  function update(jump) {
    state.shown = STEP; writeHash(); render();
    if (jump) papersSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  document.addEventListener('click', function (e) {
    var t = e.target.closest ? e.target : null;
    if (!t) return;
    var f = t.closest('[data-filter]');
    if (f) { state.tag = f.getAttribute('data-filter'); update(!f.classList.contains('chip')); return; }
    var fl = t.closest('[data-flag]');
    if (fl) { var k = fl.getAttribute('data-flag'); state[k] = !state[k]; update(false); return; }
    var term = t.closest('[data-term]');
    if (term) { q.value = term.getAttribute('data-term'); state.text = q.value.toLowerCase(); state.tag = ''; update(true); }
  });
  more.addEventListener('click', function () { state.shown += STEP; render(); });
  var timer = 0;
  q.addEventListener('input', function () {
    clearTimeout(timer);
    timer = setTimeout(function () { state.text = q.value.trim().toLowerCase(); update(false); }, 150);
  });
  window.addEventListener('hashchange', function () { readHash(); state.shown = STEP; render(); });

  readHash();
  render();
  if (location.hash.length > 1 && (state.tag || state.text || state.bench || state.watch)) {
    papersSection.classList.add('in');
    setTimeout(function () { papersSection.scrollIntoView({ block: 'start' }); }, 0);
  }
})();
