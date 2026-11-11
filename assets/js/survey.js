// Survey page: filter the paper list by subfield / benchmark / free text,
// 30 at a time with "show more". Bars in Figure 1 and the subfield cards
// jump to the list with their filter applied.
(function () {
  var list = document.getElementById('survey-list');
  if (!list) return;
  var cards = Array.prototype.slice.call(list.querySelectorAll('.paper'));
  var q = document.getElementById('survey-q');
  var more = document.getElementById('survey-more');
  var count = document.getElementById('survey-count');
  var empty = document.getElementById('survey-empty');
  var chips = Array.prototype.slice.call(document.querySelectorAll('.chips .chip[data-filter]'));
  var benchChip = document.querySelector('.chips .chip[data-bench]');
  var STEP = 30;
  var state = { tag: '', bench: false, text: '', shown: STEP };
  var texts = cards.map(function (c) { return c.textContent.toLowerCase().replace(/\s+/g, ' '); });

  function render() {
    var words = state.text.split(/\s+/).filter(Boolean);
    var match = 0, visible = 0;
    cards.forEach(function (c, i) {
      var ok = (!state.tag || (' ' + c.getAttribute('data-tags') + ' ').indexOf(' ' + state.tag + ' ') >= 0)
        && (!state.bench || c.getAttribute('data-bench') === 'true')
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
    benchChip.setAttribute('aria-pressed', String(state.bench));
  }

  function setTag(tag, jump) {
    state.tag = tag; state.shown = STEP; render();
    if (jump) document.getElementById('papers').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('[data-filter]');
    if (b) { setTag(b.getAttribute('data-filter'), !b.classList.contains('chip')); return; }
    if (e.target.closest && e.target.closest('[data-bench]') === benchChip) { state.bench = !state.bench; state.shown = STEP; render(); }
  });
  more.addEventListener('click', function () { state.shown += STEP; render(); });
  var t = 0;
  q.addEventListener('input', function () {
    clearTimeout(t);
    t = setTimeout(function () { state.text = q.value.trim().toLowerCase(); state.shown = STEP; render(); }, 120);
  });
  render();
})();
