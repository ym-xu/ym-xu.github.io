// One task per screen for the themed practice (a page with <div class="steps">, each task a
// <div class="step"> inside it): a bar with "第 k / N 步" and one dot per step, 上一步 / 下一步,
// and a switch to show all steps at once. The position is remembered in this browser.
// A step counts as done when its exercise has passed (class "passed", set by pyodide-run.js),
// when its question was answered right (quiz.js), or, for steps without a check, once it was opened.
// Adding the class "steps-all" to <html> (tools/check_course.py does) shows every step.
(function () {
  'use strict';
  function make(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function setup(box) {
    var steps = Array.prototype.filter.call(box.children, function (el) { return el.classList.contains('step'); });
    if (!steps.length) return;
    var key = 'course-step:v1:' + location.pathname.replace(/index\.html$/, '');
    var cur = 0, seen = {}, showAll = false;
    try { cur = Math.max(0, Math.min(steps.length - 1, parseInt(localStorage.getItem(key) || '0', 10) || 0)); } catch (e) { /* no storage */ }

    var bar = make('div', 'steps-bar'), label = make('span', 'steps-label'), dots = make('span', 'steps-dots');
    var all = make('button', 'steps-all-btn');
    all.type = 'button';
    var dot = steps.map(function (_, i) {
      var d = make('button', 'steps-dot', String(i + 1));
      d.type = 'button';
      d.setAttribute('aria-label', '第 ' + (i + 1) + ' 步');
      d.addEventListener('click', function () { show(i, false); });
      dots.appendChild(d);
      return d;
    });
    bar.appendChild(label); bar.appendChild(dots); bar.appendChild(all);
    box.insertBefore(bar, steps[0]);

    var nav = make('div', 'steps-nav'), prev = make('button', 'steps-prev', '← 上一步'), next = make('button', 'steps-next');
    prev.type = next.type = 'button';
    nav.appendChild(prev); nav.appendChild(next);
    box.appendChild(nav);

    function isDone(i) {
      var s = steps[i];
      if (s.querySelector('pre.py-check')) return s.classList.contains('passed');
      if (s.querySelector('.quiz-opt')) return !!s.querySelector('.quiz-opt.right');
      return !!seen[i];
    }
    function render() {
      steps.forEach(function (s, i) { s.hidden = !showAll && i !== cur; });
      var nDone = 0;
      dot.forEach(function (d, i) {
        var done = isDone(i);
        if (done) nDone++;
        d.className = 'steps-dot' + (done ? ' done' : '') + (!showAll && i === cur ? ' now' : '');
      });
      label.textContent = showAll ? '共 ' + steps.length + ' 步,已完成 ' + nDone : '第 ' + (cur + 1) + ' / ' + steps.length + ' 步';
      all.textContent = showAll ? '一步一屏' : '全部展开';
      nav.hidden = showAll;
      prev.disabled = cur === 0;
      var last = cur === steps.length - 1;
      next.disabled = last;
      next.textContent = last ? (nDone === steps.length ? '这一部分做完了 ✓' : '最后一步') : '下一步 →';
      next.className = 'steps-next' + (isDone(cur) && !last ? ' ready' : '');
    }
    function show(i, scroll) {
      cur = i;
      seen[i] = true;
      try { localStorage.setItem(key, String(i)); } catch (e) { /* no storage */ }
      render();
      if (scroll) bar.scrollIntoView({ block: 'start' });
    }
    prev.addEventListener('click', function () { if (cur > 0) show(cur - 1, true); });
    next.addEventListener('click', function () { if (cur < steps.length - 1) show(cur + 1, true); });
    all.addEventListener('click', function () { showAll = !showAll; render(); });
    // exercises and questions mark themselves with classes; redraw the bar when that happens
    // (changes inside the bar and the buttons are our own and must not trigger another redraw)
    new MutationObserver(function (changes) {
      if (changes.some(function (c) { return !bar.contains(c.target) && !nav.contains(c.target); })) render();
    }).observe(box, { subtree: true, attributes: true, attributeFilter: ['class'] });
    if (document.documentElement.classList.contains('steps-all')) showAll = true;
    new MutationObserver(function () {
      if (document.documentElement.classList.contains('steps-all') && !showAll) { showAll = true; render(); }
    }).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    show(cur, false);
  }
  function init() { Array.prototype.forEach.call(document.querySelectorAll('.steps'), setup); }
  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
