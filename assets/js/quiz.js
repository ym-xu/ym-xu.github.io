// "想一想" quizzes in the course lessons: tap an option, see whether it is right
// and why. Markup (raw HTML in the lesson's Markdown):
//   <div class="quiz">
//     <p><b>想一想</b>: question</p>
//     <button class="quiz-opt" data-why="why this is wrong">option</button>
//     <button class="quiz-opt" data-correct data-why="why this is right">option</button>
//   </div>
// Any option may be tried; the explanation of the last one tapped is shown.
(function () {
  function choose(btn) {
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
    if (right && window.courseFireworks) {
      var r = btn.getBoundingClientRect();
      window.courseFireworks(r.left + r.width / 2, r.top + r.height / 2, false);
    }
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
  if (document.readyState !== 'loading') notes();
  else document.addEventListener('DOMContentLoaded', notes);
})();
