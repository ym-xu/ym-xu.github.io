// Site-wide behaviour: dark-mode toggle + copy buttons on code cards.
(function () {
  var KEY = 'prefers-color-scheme';
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  function saved() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function isDark() {
    var t = root.getAttribute('data-theme');
    if (t) return t === 'dark';
    return !!(media && media.matches);
  }

  function init() {
    var btn = document.getElementById('theme-toggle');
    if (btn) {
      btn.setAttribute('aria-pressed', String(isDark()));
      btn.addEventListener('click', function () {
        var next = isDark() ? 'light' : 'dark';
        root.setAttribute('data-theme', next);
        try { localStorage.setItem(KEY, next); } catch (e) {}
        btn.setAttribute('aria-pressed', String(next === 'dark'));
      });
    }
    // No explicit choice saved → keep following the OS when it changes.
    if (media && media.addEventListener) {
      media.addEventListener('change', function () {
        var s = saved();
        if (s !== 'dark' && s !== 'light') {
          root.removeAttribute('data-theme');
          if (btn) btn.setAttribute('aria-pressed', String(isDark()));
        }
      });
    }
  }

  // Copy button for the code-card render hook.
  document.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('.code-copy');
    if (!btn) return;
    var card = btn.closest('.code-card');
    var pre = card && card.querySelector('pre');
    if (!pre || !navigator.clipboard) return;
    navigator.clipboard.writeText(pre.innerText).then(function () {
      var old = btn.textContent;
      btn.textContent = 'Copied';
      btn.classList.add('copied');
      setTimeout(function () { btn.textContent = old; btn.classList.remove('copied'); }, 1500);
    });
  });

  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
