// "想一想" quizzes in the course lessons: tap an option, see whether it is right
// and why. Markup (raw HTML in the lesson's Markdown):
//   <div class="quiz">
//     <p><b>想一想</b>: question</p>
//     <button class="quiz-opt" data-why="why this is wrong">option</button>
//     <button class="quiz-opt" data-correct data-why="why this is right">option</button>
//   </div>
// Any option may be tried; the explanation of the last one tapped is shown.
// The choice is kept in this browser (localStorage) and shown again when the page is reopened.
(function () {
  var QKEY = 'course-quiz:v1:' + location.pathname.replace(/index\.html$/, '');
  function readQ() { try { return JSON.parse(localStorage.getItem(QKEY)) || {}; } catch (e) { return {}; } }
  function quizzes() { return Array.prototype.slice.call(document.querySelectorAll('.quiz')); }
  function options(quiz) { return Array.prototype.slice.call(quiz.querySelectorAll('.quiz-opt')); }
  // silent: putting a saved choice back, without fireworks and without saving again
  function choose(btn, silent) {
    var quiz = btn.closest('.quiz');
    var right = btn.hasAttribute('data-correct');
    quiz.querySelectorAll('.quiz-opt').forEach(function (b) { b.classList.remove('right', 'wrong'); });
    btn.classList.add(right ? 'right' : 'wrong');
    var why = quiz.querySelector('.quiz-why');
    if (!why) {
      why = document.createElement('p');
      why.className = 'quiz-why';
      why.setAttribute('aria-live', 'polite');
      quiz.appendChild(why);
    }
    why.textContent = '';
    var tag = document.createElement('b');
    tag.className = right ? 'ok' : 'no';
    tag.textContent = right ? '对。' : '不对。';
    why.appendChild(tag);
    why.appendChild(document.createTextNode(btn.getAttribute('data-why') || ''));
    if (silent) return;
    var all = readQ();
    all[quizzes().indexOf(quiz)] = options(quiz).indexOf(btn);
    try { localStorage.setItem(QKEY, JSON.stringify(all)); } catch (e) { /* no storage */ }
    if (right && window.courseFireworks) {
      var r = btn.getBoundingClientRect();
      window.courseFireworks(r.left + r.width / 2, r.top + r.height / 2, false);
    }
  }
  function restore() {
    var all = readQ();
    quizzes().forEach(function (quiz, i) {
      var opt = options(quiz)[all[i]];
      if (opt) choose(opt, true);
    });
  }
  document.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('.quiz-opt');
    if (btn) choose(btn);
  });

  // Free-text answers in the homework (<textarea data-save>): kept in this browser only.
  function notes() {
    Array.prototype.forEach.call(document.querySelectorAll('textarea[data-save]'), function (box, i) {
      var key = 'course-notes:v1:' + location.pathname.replace(/index\.html$/, '') + ':' + i;
      try { var saved = localStorage.getItem(key); if (saved) box.value = saved; } catch (e) { /* no storage */ }
      box.addEventListener('input', function () {
        try { localStorage.setItem(key, box.value); } catch (e) { /* no storage */ }
      });
    });
  }
  function start() { notes(); restore(); }
  if (document.readyState !== 'loading') start();
  else document.addEventListener('DOMContentLoaded', start);
})();
