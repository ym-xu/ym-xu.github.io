// Learning progress for the courses, kept in this browser only (localStorage). A course is read
// over several days: this file remembers where the reader stopped and shows it again.
//
// On a lesson or practice page ([data-lesson]):
//   course-visit:v1:<path>  { t, first, h, ht, end }  last visit, the section (h2) being read, end reached
//   course-last:v1:<root>   { path, t }               the page of this course opened last
//   and, on the next visit, a bar "上次读到 …" that jumps back to that section.
// On a course page ([data-course]): the panel "我的进度" (continue button, counts), the state of each
// row of the syllabus, and a link that carries the progress to another device (or keeps a backup).
//
// Other records read here are written elsewhere: course-progress:v1:<path> (exercises passed,
// pyodide-run.js), course-step:v1:<path> (course-steps.js), course-quiz:v1:<path> (quiz.js).
(function () {
  'use strict';
  var path = location.pathname.replace(/index\.html$/, '');
  var root = '/' + (path.split('/')[1] || '') + '/';          // "/python/" or "/stat-ml/"
  var SAFE_PATH = /^\/[A-Za-z0-9_\-\/]+\/$/;
  var PREFIXES = ['course-progress:v1:', 'course-step:v1:', 'course-visit:v1:', 'course-quiz:v1:'];

  function get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch (e) { return null; } }
  function set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* no storage */ } }
  function make(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function ago(t) {
    var days = Math.floor((new Date().setHours(0, 0, 0, 0) - new Date(t).setHours(0, 0, 0, 0)) / 864e5);
    return days <= 0 ? '今天' : days === 1 ? '昨天' : days + ' 天前';
  }

  // ------------------------------------------------------------------ lesson and practice pages
  function lessonPage(article) {
    var main = article.querySelector('.lesson-main') || article;
    var heads = Array.prototype.slice.call(main.querySelectorAll('.content h2[id]'));
    var stepped = !!main.querySelector('.steps');            // the practice keeps its own place (course-steps.js)
    var key = 'course-visit:v1:' + path;
    var v = get(key) || {};
    var before = { h: v.h, ht: v.ht, end: v.end };
    if (!v.first) v.first = Date.now();

    function save() {
      v.t = Date.now();
      set(key, v);
      set('course-last:v1:' + root, { path: path, t: v.t });
    }
    function look() {
      var cur = null;
      for (var i = 0; i < heads.length; i++) {
        if (heads[i].getBoundingClientRect().top - 100 <= 0) cur = heads[i];
        else break;
      }
      if (cur) { v.h = cur.id; v.ht = cur.textContent.trim(); }
      var pager = main.querySelector('.pager');
      if (!stepped && pager && pager.getBoundingClientRect().top < window.innerHeight) v.end = true;
    }
    var timer = null, ticking = false;
    window.addEventListener('scroll', function () {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(function () {
        ticking = false;
        look();
        clearTimeout(timer);
        timer = setTimeout(save, 800);
      });
    }, { passive: true });
    document.addEventListener('visibilitychange', function () { if (document.hidden) save(); });
    window.addEventListener('pagehide', save);
    save();

    // "上次读到 …": offered, not forced; only when the reader stopped in the middle of the lesson
    var target = before.h && document.getElementById(before.h);
    if (target && !before.end && heads.length && before.h !== heads[0].id && !location.hash) {
      var bar = make('p', 'resume-bar');
      bar.appendChild(make('span', null, '上次读到: '));
      bar.appendChild(make('b', null, before.ht || ''));
      var go = make('button', 'resume-go', '回到那里 →');
      go.type = 'button';
      go.addEventListener('click', function () { target.scrollIntoView({ block: 'start' }); bar.remove(); });
      bar.appendChild(go);
      var layout = article.querySelector('.lesson-layout');
      if (layout) layout.parentNode.insertBefore(bar, layout);
    }
  }

  // ------------------------------------------------------------------ the course page
  function coursePage(page) {
    var rows = Array.prototype.slice.call(page.querySelectorAll('a.lesson-row[data-progress-path]'));
    var side = page.querySelector('[data-course-side]');
    var note = page.querySelector('[data-continue-note]');

    function stateOf(row) {
      var p = row.getAttribute('data-progress-path');
      var prog = get('course-progress:v1:' + p) || {}, visit = get('course-visit:v1:' + p) || {};
      var total = parseInt(row.getAttribute('data-ex') || '0', 10) || prog.core || 0;
      var passed = Math.min(prog.coreDone || 0, total);
      var practice = row.hasAttribute('data-practice-row');
      var started = !!(visit.t || passed);
      var done = practice ? (total > 0 && passed >= total) : (!!visit.end && passed >= total);
      return { path: p, practice: practice, total: total, passed: passed, started: started, done: done, visit: visit };
    }

    function draw() {
      var states = rows.map(stateOf);
      var lessons = 0, lessonsDone = 0, ex = 0, exDone = 0, parts = 0, partsDone = 0;
      rows.forEach(function (row, i) {
        var s = states[i];
        Array.prototype.forEach.call(row.querySelectorAll('.lesson-row-tag'), function (t) { t.remove(); });
        row.classList.remove('is-next');
        if (s.practice) { parts++; if (s.done) partsDone++; }
        else { lessons++; ex += s.total; exDone += s.passed; if (s.done) lessonsDone++; }
        if (!s.started) return;
        var text;
        if (s.practice) text = s.done ? '做完了 ✓' : '已通过 ' + s.passed + ' / ' + s.total;
        else if (s.done) text = '学完 ✓';
        else text = (s.visit.end ? '读完' : '在学') + (s.total ? ' · 练习 ' + s.passed + ' / ' + s.total : '');
        var tag = make('span', 'lesson-row-tag' + (s.done ? ' done' : ' doing'), text);
        // a practice row is narrow: its tag goes on the second line, after the time and the number of steps
        row.querySelector(s.practice ? '.lesson-row-sum' : '.lesson-row-title').appendChild(tag);
      });

      // parts of the practice track that are in its outline but not written yet count too
      var planned = +(page.getAttribute('data-practice-total') || 0);
      if (planned > parts) parts = planned;
      // the head of the practice column: how many parts of the practice project are done
      var partsStarted = states.filter(function (s) { return s.practice && s.passed; }).length;
      Array.prototype.forEach.call(page.querySelectorAll('[data-practice-count]'), function (count) {
        if (partsStarted) count.textContent = '已完成 ' + partsDone + ' / ' + parts + ' 部分';
      });

      if (!side) return;
      // a count that has nothing to count on this page (no lessons, or no practice) is not shown
      function stat(name, n, total, unit) {
        var el = side.querySelector('[data-stat="' + name + '"]');
        if (!el) return;
        el.textContent = n + ' / ' + total + unit;
        el.parentNode.hidden = !total;
      }
      var tasks = 0, tasksDone = 0;
      states.forEach(function (s) { if (s.practice) { tasks += s.total; tasksDone += s.passed; } });
      stat('lessons', lessonsDone, lessons, ' 课');
      stat('ex', exDone, ex, ' 题');
      stat('practice', partsDone, parts, ' 部分');
      stat('tasks', tasksDone, lessons ? 0 : tasks, ' 题');     // only where the page lists nothing but practice

      // where to go on: the page opened last; if that one is finished, the next unfinished row after it
      var btn = side.querySelector('[data-continue]');
      var last = get('course-last:v1:' + root);
      var at = last && SAFE_PATH.test(last.path || '') ? rows.map(function (r) { return r.getAttribute('data-progress-path'); }).indexOf(last.path) : -1;
      if (at < 0) {
        // nothing recorded in this section yet (or the last page belongs to another list): start at the first unfinished row
        for (var k = 0; k < rows.length; k++) if (!states[k].done) { at = k; break; }
        if (at < 0 || !states.some(function (s) { return s.started; })) return;   // nothing started: the button keeps its text
      } else if (states[at].done) {
        var nxt = -1;
        for (var j = at + 1; j < rows.length; j++) if (!states[j].done) { nxt = j; break; }
        if (nxt < 0) for (var m = 0; m < at; m++) if (!states[m].done) { nxt = m; break; }
        if (nxt < 0) {
          btn.textContent = '全部学完了 ✓';
          btn.setAttribute('href', rows[at].getAttribute('href'));
          if (note && last) note.textContent = '上次学习: ' + ago(last.t);
          return;
        }
        at = nxt;
      }
      var row = rows[at], s = states[at];
      var title = row.getAttribute('data-title') || '';
      var here = !s.practice && s.visit.h && !s.visit.end;
      btn.textContent = (s.started ? '继续: ' : '下一步: ') + title + ' →';
      btn.setAttribute('href', row.getAttribute('href') + (here ? '#' + encodeURIComponent(s.visit.h) : ''));
      row.classList.add('is-next');
      if (note) note.textContent = (here && s.visit.ht ? '读到「' + s.visit.ht + '」' : '') + (last ? (here && s.visit.ht ? ' · ' : '') + '上次学习: ' + ago(last.t) : '');
    }

    // ---- the progress as a link: for another device, or as a backup ----
    function collect() {
      var out = {};
      try {
        for (var i = 0; i < localStorage.length; i++) {
          var k = localStorage.key(i);
          var hit = PREFIXES.some(function (p) { return k.indexOf(p + root) === 0; }) || k === 'course-last:v1:' + root;
          if (hit) out[k] = get(k);
        }
      } catch (e) { /* no storage */ }
      return out;
    }
    function b64(bytes) {
      var s = '';
      for (var i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
      return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    }
    function unb64(str) {
      var s = atob(str.replace(/-/g, '+').replace(/_/g, '/'));
      var bytes = new Uint8Array(s.length);
      for (var i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
      return bytes;
    }
    function pipe(bytes, stream) {
      return new Response(new Blob([bytes]).stream().pipeThrough(stream)).arrayBuffer().then(function (b) { return new Uint8Array(b); });
    }
    function encode(obj) {
      var raw = new TextEncoder().encode(JSON.stringify(obj));
      if (typeof CompressionStream === 'undefined') return Promise.resolve('j' + b64(raw));
      return pipe(raw, new CompressionStream('deflate-raw')).then(function (z) { return 'z' + b64(z); }, function () { return 'j' + b64(raw); });
    }
    function decode(str) {
      var bytes = unb64(str.slice(1));
      var p = str.charAt(0) === 'z' ? pipe(bytes, new DecompressionStream('deflate-raw')) : Promise.resolve(bytes);
      return p.then(function (raw) { return JSON.parse(new TextDecoder().decode(raw)); });
    }
    // Merging never loses anything: an exercise passed on either device counts, the later visit wins.
    function merge(incoming) {
      var n = 0;
      Object.keys(incoming || {}).forEach(function (k) {
        var v = incoming[k], mine = get(k);
        var kind = PREFIXES.filter(function (p) { return k.indexOf(p + root) === 0; })[0] || (k === 'course-last:v1:' + root ? 'last' : null);
        if (!kind || v == null || typeof v !== 'object' && typeof v !== 'number') return;
        if (kind !== 'last' && !SAFE_PATH.test(k.slice(kind.length))) return;
        if (kind === 'course-progress:v1:') {
          var a = (mine && mine.done) || [], b = Array.isArray(v.done) ? v.done : [], done = [];
          for (var i = 0; i < Math.max(a.length, b.length) && i < 200; i++) done[i] = !!(a[i] || b[i]);
          var core = Math.max((mine && mine.core) || 0, Number(v.core) || 0), hw = Math.max((mine && mine.hw) || 0, Number(v.hw) || 0);
          var hwDone = Math.min(hw, Math.max((mine && mine.hwDone) || 0, Number(v.hwDone) || 0));
          set(k, { done: done, total: core + hw, core: core, hw: hw, hwDone: hwDone, coreDone: Math.min(core, done.filter(Boolean).length - hwDone) });
        } else if (kind === 'course-visit:v1:') {
          var later = !mine || (Number(v.t) || 0) > (mine.t || 0) ? v : mine;
          var first = Math.min((mine && mine.first) || Infinity, Number(v.first) || Infinity);
          set(k, { t: Number(later.t) || 0, first: isFinite(first) ? first : undefined,
                   h: String(later.h || '').slice(0, 200), ht: String(later.ht || '').slice(0, 200), end: !!((mine && mine.end) || v.end) });
        } else if (kind === 'course-step:v1:') {
          if (mine == null) set(k, Math.max(0, Math.min(50, Number(v) || 0)));
        } else if (kind === 'course-quiz:v1:') {
          var q = mine || {};
          Object.keys(v).slice(0, 100).forEach(function (i) { if (!(i in q) && /^\d+$/.test(i)) q[i] = Number(v[i]) || 0; });
          set(k, q);
        } else if (kind === 'last') {
          if (SAFE_PATH.test(String(v.path || '')) && String(v.path).indexOf(root) === 0 && (!mine || (Number(v.t) || 0) > (mine.t || 0))) set(k, { path: String(v.path), t: Number(v.t) || 0 });
        }
        n++;
      });
      return n;
    }

    var m = /^#progress=([zj][A-Za-z0-9_\-]+)$/.exec(location.hash);
    if (m) {
      decode(m[1]).then(function (obj) {
        var n = merge(obj);
        history.replaceState(null, '', location.pathname);
        draw();
        if (side) side.querySelector('[data-progress-msg]').textContent = n ? '已把链接里的进度合并到这台设备。' : '链接里没有可以合并的进度。';
      }, function () {
        if (side) side.querySelector('[data-progress-msg]').textContent = '这个进度链接不完整,没能读出来。请重新复制一次。';
      });
    }

    var open = side && side.querySelector('[data-progress-export]'), box = side && side.querySelector('[data-progress-box]');
    if (open && box) {
      open.addEventListener('click', function () {
        if (!box.hidden) { box.hidden = true; return; }
        encode(collect()).then(function (code) {
          var field = box.querySelector('textarea');
          field.value = location.origin + root + '#progress=' + code;
          box.hidden = false;
          field.focus();
          field.select();
        });
      });
      box.querySelector('[data-progress-copy]').addEventListener('click', function () {
        var field = box.querySelector('textarea'), self = this;
        field.select();
        var done = function () { self.textContent = '已复制 ✓'; setTimeout(function () { self.textContent = '复制链接'; }, 2000); };
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(field.value).then(done, function () { document.execCommand('copy'); done(); });
        else { document.execCommand('copy'); done(); }
      });
    }

    draw();
    // coming back from a lesson with the back button: the page may be shown from the cache
    window.addEventListener('pageshow', function (e) { if (e.persisted) draw(); });
  }

  function init() {
    var article = document.querySelector('[data-lesson]');
    if (article) lessonPage(article);
    var page = document.querySelector('[data-course]');
    if (page) coursePage(page);
  }
  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
