// Interactive figures for the course lessons (/python/, /stat-ml/).
// Each figure is a small "explorable": one control, one readout, plain SVG, no dependencies.
// They work with touch (a native slider or buttons; nothing is dragged), follow the site's
// light and dark colours through CSS classes, and use simulated patients only. The generator
// is seeded, so every reader sees the same picture and the lesson text can quote its numbers.
//
// Markup in a lesson's Markdown:   <div class="widget" data-widget="overfit"></div>
//   overfit            a model with a flexibility dial: error on its own patients vs new patients
//   overfit + data-cv  the same, with the 5-fold cross-validation error as a third bar
//   kfold              step through the five rounds of one 5-fold cross-validation
//   repeatcv           repeat the 5-fold cross-validation with new random splits
//   ci                 many studies, each with its 95% interval: how many contain the true value
//   shuffle            a P value by shuffling the group labels of two groups
// All figures on a page share the same 30 simulated patients and the same first split.
(function () {
  'use strict';

  // ---------- simulated patients ----------
  function mulberry32(a) {               // small seeded generator, uniform on [0, 1)
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normal(rng) {                 // Box-Muller
    var u = 0; while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  // n patients: mean wall thickness x (mm, between lo and hi) and the change in vaginal D2cc
  // after switching plan, y (Gy). The true relation is a straight line plus noise (SD 2 Gy).
  function simulate(n, seed, lo, hi) {
    var rng = mulberry32(seed), pts = [];
    while (pts.length < n) {
      var x = Math.exp(Math.log(3.6) + 0.30 * normal(rng)), e = normal(rng);
      if (x < lo || x > hi) continue;
      pts.push({ x: Math.round(x * 100) / 100, y: Math.round((1.6 * (x - 5) + 2.0 * e) * 10) / 10 });
    }
    return pts;
  }

  // ---------- models ----------
  function fitLine(pts) {                // least-squares straight line
    var n = pts.length, mx = 0, my = 0, sxx = 0, sxy = 0;
    pts.forEach(function (p) { mx += p.x / n; my += p.y / n; });
    pts.forEach(function (p) { sxx += (p.x - mx) * (p.x - mx); sxy += (p.x - mx) * (p.y - my); });
    var b = sxy / sxx, a = my - b * mx;
    return function (x) { return a + b * x; };
  }
  // A model with a flexibility dial. Level 1 is the least-squares line. Higher levels add a
  // local average of the line's residuals (Gaussian kernel, bandwidth BANDS[level] in mm):
  // the smaller the bandwidth, the more the curve bends towards each nearby patient, until
  // (level 10) it passes through nearly every patient it was fitted on.
  var BANDS = [null, Infinity, 1.5, 0.9, 0.55, 0.35, 0.22, 0.14, 0.09, 0.05, 0.02];
  function fitFlex(pts, level) {
    var line = fitLine(pts);
    if (level <= 1) return line;
    var h = BANDS[level], res = pts.map(function (p) { return p.y - line(p.x); });
    return function (x) {
      var sw = 0, s = 0, best = Infinity, bi = 0;
      for (var i = 0; i < pts.length; i++) {
        var dx = Math.abs(x - pts[i].x);
        if (dx < best) { best = dx; bi = i; }
        var w = Math.exp(-0.5 * (dx / h) * (dx / h));
        sw += w; s += w * res[i];
      }
      return line(x) + (sw > 1e-12 ? s / sw : res[bi]);
    };
  }
  function mae(f, pts) {
    var s = 0;
    pts.forEach(function (p) { s += Math.abs(p.y - f(p.x)); });
    return s / pts.length;
  }
  function mean(a) { return a.reduce(function (s, v) { return s + v; }, 0) / a.length; }
  function split(n, k, rng) {            // a random permutation dealt into k folds
    var idx = [], folds = [], i, j, t;
    for (i = 0; i < n; i++) idx.push(i);
    for (i = n - 1; i > 0; i--) { j = Math.floor(rng() * (i + 1)); t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    for (i = 0; i < k; i++) folds.push([]);
    idx.forEach(function (id, pos) { folds[pos % k].push(id); });
    return folds;
  }
  // For each fold: fit on the other folds, predict this one.
  function crossValidate(pts, folds, level) {
    return folds.map(function (test) {
      var held = {};
      test.forEach(function (i) { held[i] = true; });
      var f = fitFlex(pts.filter(function (p, i) { return !held[i]; }), level);
      return { f: f, test: test, mae: mae(f, test.map(function (i) { return pts[i]; })) };
    });
  }

  // ---------- the shared scene ----------
  var SEED = 567, N = 30, K = 5, scene = null;
  function getScene() {
    if (!scene) {
      var train = simulate(N, SEED, 1.5, 7.5), xs = train.map(function (p) { return p.x; });
      var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
      var fresh = simulate(2000, SEED + 5000, xmin, xmax);     // new patients, within the range of the 30
      scene = { train: train, xmin: xmin, xmax: xmax, fresh: fresh, shown: fresh.slice(0, 60) };
    }
    return scene;
  }
  function splitStream() { return mulberry32(SEED * 7 + 1); }  // every figure starts from the same first split

  // ---------- drawing ----------
  var NS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, parent, text) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function h(tag, cls, parent, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function gy(v) { return v.toFixed(2) + ' Gy'; }
  function minus(v) { return String(v).replace('-', '−'); }

  var W = 360, H = 250, M = { l: 36, r: 8, t: 10, b: 34 }, XD = [1.5, 7.0], YD = [-9.5, 5.5], uid = 0;
  function sx(x) { return M.l + (x - XD[0]) / (XD[1] - XD[0]) * (W - M.l - M.r); }
  function sy(y) { return M.t + (YD[1] - y) / (YD[1] - YD[0]) * (H - M.t - M.b); }
  // Axes of the wall-thickness / dose-change plot; returns the clipped group to draw into.
  function frame(root, label) {
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': label }, root);
    var id = 'w-clip-' + (++uid), cp = el('clipPath', { id: id }, el('defs', {}, svg));
    el('rect', { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }, cp);
    [-8, -4, 0, 4].forEach(function (v) {
      el('line', { x1: M.l, x2: W - M.r, y1: sy(v), y2: sy(v), 'class': v === 0 ? 'w-zero' : 'w-grid' }, svg);
      el('text', { x: M.l - 5, y: sy(v) + 4, 'text-anchor': 'end' }, svg, minus(v));
    });
    [2, 3, 4, 5, 6, 7].forEach(function (v) {
      el('line', { x1: sx(v), x2: sx(v), y1: H - M.b, y2: H - M.b + 4, 'class': 'w-axis' }, svg);
      el('text', { x: sx(v), y: H - M.b + 16, 'text-anchor': v === 7 ? 'end' : 'middle' }, svg, String(v));
    });
    el('line', { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, 'class': 'w-axis' }, svg);
    el('line', { x1: M.l, x2: M.l, y1: M.t, y2: H - M.b, 'class': 'w-axis' }, svg);
    el('text', { x: (M.l + W - M.r) / 2, y: H - 3, 'text-anchor': 'middle' }, svg, '平均壁厚 (mm)');
    el('text', { x: M.l + 5, y: M.t + 10 }, svg, 'Δ (Gy)');
    return el('g', { 'clip-path': 'url(#' + id + ')' }, svg);
  }
  // The model's curve between x0 and x1, sampled finely and at every fitted patient
  // (at high flexibility the curve changes within a few hundredths of a millimetre).
  function curvePath(f, x0, x1, pts) {
    var xs = [], i;
    for (i = 0; i <= 600; i++) xs.push(x0 + (x1 - x0) * i / 600);
    (pts || []).forEach(function (p) { xs.push(p.x); });
    xs.sort(function (a, b) { return a - b; });
    return xs.map(function (x, k) { return (k ? 'L' : 'M') + sx(x).toFixed(1) + ' ' + sy(f(x)).toFixed(1); }).join('');
  }
  function legend(root, items) {
    var box = h('div', 'w-legend', root);
    items.forEach(function (it) {
      var s = h('span', '', box);
      h('i', 'w-key ' + it[0], s);
      s.appendChild(document.createTextNode(it[1]));
    });
  }
  function button(parent, text, cls) {
    var b = h('button', 'w-btn' + (cls ? ' ' + cls : ''), parent, text);
    b.type = 'button';
    return b;
  }

  // ---------- figure 1: flexibility against the two errors ----------
  function overfit(root) {
    var sc = getScene(), withCV = root.hasAttribute('data-cv');
    var folds = split(N, K, splitStream());
    var plot = frame(root, '散点图: 30 位建模病人、新病人和模型的曲线');
    sc.shown.forEach(function (p) { el('circle', { cx: sx(p.x), cy: sy(p.y), r: 2.6, 'class': 'w-pt new' }, plot); });
    var path = el('path', { 'class': 'w-curve' }, plot);
    sc.train.forEach(function (p) { el('circle', { cx: sx(p.x), cy: sy(p.y), r: 3.4, 'class': 'w-pt' }, plot); });
    legend(root, [['pt', '建模的 30 位病人'], ['new', '新病人'], ['line', '模型']]);

    var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
    input.type = 'range'; input.min = 1; input.max = 10; input.step = 1; input.value = 1;
    input.setAttribute('aria-label', '模型的灵活度');

    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', bars, '平均误差: 预测值和实际值平均差多少');
    function bar(label, cls) {
      var row = h('div', 'w-bar', bars);
      h('span', 'w-bar-label', row, label);
      var fill = h('span', 'w-bar-fill ' + cls, h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
      return function (v) { fill.style.width = Math.min(100, v / 2.6 * 100).toFixed(1) + '%'; val.textContent = gy(v); };
    }
    var setTrain = bar('建模的 30 位病人', 'train');
    var setCV = withCV ? bar('5 折交叉验证', 'cv') : null;
    var setNew = bar('2000 位新病人', 'new');

    function update() {
      var level = +input.value, f = fitFlex(sc.train, level);
      path.setAttribute('d', curvePath(f, sc.xmin - 0.15, sc.xmax + 0.15, sc.train));
      cap.textContent = '模型的灵活度: ' + level + (level === 1 ? '(直线)' : level === 10 ? '(最灵活)' : '');
      setTrain(mae(f, sc.train));
      if (setCV) setCV(mean(crossValidate(sc.train, folds, level).map(function (r) { return r.mae; })));
      setNew(mae(f, sc.fresh));
    }
    input.addEventListener('input', update);
    update();
  }

  // ---------- figure 2: the five rounds of one 5-fold cross-validation ----------
  function kfold(root) {
    var sc = getScene(), folds = split(N, K, splitStream()), rounds = crossValidate(sc.train, folds, 1);
    var cvMean = mean(rounds.map(function (r) { return r.mae; }));
    var truth = mae(fitLine(sc.train), sc.fresh);
    var step = 0;                         // 0 before the start, 1-5 the rounds, 6 the summary
    var plot = frame(root, '散点图: 5 折交叉验证的每一轮');
    var segs = el('g', {}, plot), path = el('path', { 'class': 'w-curve' }, plot);
    var dots = sc.train.map(function (p) { return el('circle', { cx: sx(p.x), cy: sy(p.y), r: 3.4 }, plot); });
    legend(root, [['pt', '拿出来预测的病人'], ['rest', '用来建模的病人'], ['line', '拟合的直线']]);

    var chips = h('div', 'w-chips', root), chip = [];
    for (var i = 0; i < K; i++) chip.push(h('span', 'w-chip', chips));
    var total = h('span', 'w-chip mean', chips);
    var ctls = h('div', 'w-ctls', root), next = button(ctls, '', 'primary');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');

    function render() {
      var r = step >= 1 && step <= K ? rounds[step - 1] : null, held = {};
      if (r) r.test.forEach(function (i) { held[i] = true; });
      while (segs.firstChild) segs.removeChild(segs.firstChild);
      dots.forEach(function (d, i) {
        d.setAttribute('class', 'w-pt' + (step === 0 ? ' idle' : r && !held[i] ? ' rest' : ''));
      });
      path.setAttribute('d', r ? curvePath(r.f, XD[0], XD[1]) : '');
      // error bars: this round's held-out patients, or (summary) every patient against the
      // prediction of the round that had not seen them
      (r ? [r] : step > K ? rounds : []).forEach(function (rr) {
        rr.test.forEach(function (i) {
          var p = sc.train[i];
          el('line', { x1: sx(p.x), x2: sx(p.x), y1: sy(p.y), y2: sy(rr.f(p.x)), 'class': 'w-res' }, segs);
          if (step > K) el('line', { x1: sx(p.x) - 3, x2: sx(p.x) + 3, y1: sy(rr.f(p.x)), y2: sy(rr.f(p.x)), 'class': 'w-res' }, segs);
        });
      });
      chip.forEach(function (c, i) {
        var done = step > i;
        c.className = 'w-chip' + (step === i + 1 ? ' now' : done ? ' done' : '');
        c.textContent = '第 ' + (i + 1) + ' 份' + (done ? ' ' + rounds[i].mae.toFixed(2) : '');
      });
      total.className = 'w-chip mean' + (step > K ? ' now' : '');
      total.textContent = '平均' + (step > K ? ' ' + cvMean.toFixed(2) : '');
      next.textContent = step === 0 ? '开始第 1 轮' : step < K ? '下一轮(第 ' + (step + 1) + ' 轮)' : step === K ? '看 5 轮的合计' : '从头再看一遍';
      status.textContent = step === 0
        ? '30 位模拟病人已经随机分成 5 份,每份 6 人。点"开始第 1 轮"。'
        : step <= K
          ? '第 ' + step + ' 轮: 用另外 24 位病人(灰点)拟合直线,去预测拿出来的 6 位(蓝点)。蓝色竖线是这 6 位的误差,平均 ' + gy(r.mae) + '。'
          : '5 轮做完,每位病人都恰好被预测了一次: 竖线连着每位病人的实际值和预测值(短横线),预测用的都是没见过这位病人的那条直线。5 份的平均是 ' + gy(cvMean) + '。对照: 用全部 30 位病人拟合的直线,在 2000 位新病人身上平均差 ' + gy(truth) + '。';
    }
    next.addEventListener('click', function () { step = step > K ? 0 : step + 1; render(); });
    render();
  }

  // ---------- figure 3: repeat the 5-fold cross-validation with new splits ----------
  function repeatcv(root) {
    var sc = getScene(), rng = splitStream(), vals = [], MAX = 101;
    var w = 360, hh = 150, base = 116, lo = 1.4, hi = 2.0, bin = 0.02;
    function px(v) { return 20 + (v - lo) / (hi - lo) * (w - 40); }
    var svg = el('svg', { viewBox: '0 0 ' + w + ' ' + hh, role: 'img', 'aria-label': '点图: 每一次 5 折交叉验证的平均误差' }, root);
    el('line', { x1: 14, x2: w - 14, y1: base, y2: base, 'class': 'w-axis' }, svg);
    for (var t = 0; t <= 6; t++) {
      var v = lo + t * 0.1;
      el('line', { x1: px(v), x2: px(v), y1: base, y2: base + 4, 'class': 'w-axis' }, svg);
      el('text', { x: px(v), y: base + 16, 'text-anchor': 'middle' }, svg, v.toFixed(1));
    }
    el('text', { x: w / 2, y: hh - 3, 'text-anchor': 'middle' }, svg, '一次 5 折交叉验证的平均误差 (Gy)');
    var marker = el('line', { y1: 8, y2: base, 'class': 'w-mean' }, svg), g = el('g', {}, svg);
    var ctls = h('div', 'w-ctls', root);
    var one = button(ctls, '再分 1 次', 'primary'), many = button(ctls, '再分 20 次'), reset = button(ctls, '重来');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');

    function add(k) {
      for (var i = 0; i < k && vals.length < MAX; i++) {
        vals.push(mean(crossValidate(sc.train, split(N, K, rng), 1).map(function (r) { return r.mae; })));
      }
      render();
    }
    function render() {
      var count = {}, m = mean(vals);
      while (g.firstChild) g.removeChild(g.firstChild);
      vals.forEach(function (v, i) {
        var b = Math.round((Math.min(hi, Math.max(lo, v)) - lo) / bin);
        count[b] = (count[b] || 0) + 1;
        el('circle', { cx: px(lo + b * bin), cy: Math.max(6, base - 5 - (count[b] - 1) * 6.4), r: 3, 'class': 'w-dot' + (i === vals.length - 1 ? ' now' : '') }, g);
      });
      marker.setAttribute('x1', px(m)); marker.setAttribute('x2', px(m));
      marker.style.display = vals.length > 1 ? '' : 'none';
      one.disabled = many.disabled = vals.length >= MAX;
      status.textContent = vals.length === 1
        ? '第 1 个点是上一张图那种分法的结果: ' + gy(vals[0]) + '。点"再分 1 次",换一种随机分法,把 5 折重做一遍。'
        : '已经做了 ' + vals.length + ' 次 5 折交叉验证,每次的分法都不同。最小 ' + gy(Math.min.apply(null, vals)) + ',最大 ' + gy(Math.max.apply(null, vals)) + ',平均 ' + gy(m) + '(竖线)。' + (vals.length >= MAX ? '图里最多画 ' + MAX + ' 次。' : '');
    }
    one.addEventListener('click', function () { add(1); });
    many.addEventListener('click', function () { add(20); });
    reset.addEventListener('click', function () { rng = splitStream(); vals = []; add(1); });
    add(1);
  }

  // ---------- figure 4: what a 95% interval means ----------
  // Each "study" draws 30 patients from a population whose true mean change is -2.0 Gy and
  // computes the usual 95% interval of the mean (mean +- 2.045 standard errors). About 95 of
  // 100 such intervals contain the true value; with this seed exactly 95 of the first 100 do.
  function ci(root) {
    var TRUE = -2.0, SD = 2.8, n = 30, T = 2.045, MAX = 100, rng, studies;
    var w = 360, top = 16, row = 3.1, base = top + MAX * row + 6, hh = base + 36, lo = -4.5, hi = 0.5;
    function px(v) { return 24 + (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * (w - 36); }
    var svg = el('svg', { viewBox: '0 0 ' + w + ' ' + hh, role: 'img', 'aria-label': '每个研究的平均值和 95% 区间' }, root);
    el('line', { x1: 16, x2: w - 8, y1: base, y2: base, 'class': 'w-axis' }, svg);
    for (var v = -4; v <= 0; v++) {
      el('line', { x1: px(v), x2: px(v), y1: base, y2: base + 4, 'class': 'w-axis' }, svg);
      el('text', { x: px(v), y: base + 16, 'text-anchor': 'middle' }, svg, minus(v));
    }
    el('text', { x: w / 2, y: hh - 3, 'text-anchor': 'middle' }, svg, '一个研究里 Δ 的平均值 (Gy)');
    el('line', { x1: px(TRUE), x2: px(TRUE), y1: top - 4, y2: base, 'class': 'w-mean' }, svg);
    el('text', { x: px(TRUE) + 5, y: top - 5 }, svg, '真实值 −2.0');
    var g = el('g', {}, svg);
    var ctls = h('div', 'w-ctls', root);
    var one = button(ctls, '做 1 个研究', 'primary'), many = button(ctls, '做 20 个'), again = button(ctls, '重来');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');

    function add(k) {
      for (var j = 0; j < k && studies.length < MAX; j++) {
        var x = [], i, m, ss = 0;
        for (i = 0; i < n; i++) x.push(TRUE + SD * normal(rng));
        m = mean(x);
        for (i = 0; i < n; i++) ss += (x[i] - m) * (x[i] - m);
        var half = T * Math.sqrt(ss / (n - 1)) / Math.sqrt(n);
        studies.push({ m: m, lo: m - half, hi: m + half, hit: m - half <= TRUE && TRUE <= m + half });
      }
      render();
    }
    function render() {
      while (g.firstChild) g.removeChild(g.firstChild);
      studies.forEach(function (st, i) {
        var y = top + (i + 0.5) * row;
        el('line', { x1: px(st.lo), x2: px(st.hi), y1: y, y2: y, 'class': 'w-int' + (st.hit ? '' : ' miss') }, g);
        el('circle', { cx: px(st.m), cy: y, r: 1.3, 'class': 'w-int-dot' + (st.hit ? '' : ' miss') }, g);
      });
      var hits = studies.filter(function (st) { return st.hit; }).length, N = studies.length;
      one.disabled = many.disabled = N >= MAX;
      status.textContent = N === 0
        ? '所有这类病人里,换方案后 Δ 的真实平均值是 −2.0 Gy(竖线)。点"做 1 个研究": 随机抽 30 位病人,算出平均值(圆点)和它的 95% 区间(横线)。'
        : '已经做了 ' + N + ' 个研究,每个研究 30 位病人。' + hits + ' 个研究的 95% 区间包含了真实值' + (N >= 10 ? '(占 ' + Math.round(hits / N * 100) + '%)' : '') + ';没包含的 ' + (N - hits) + ' 个画成红色粗线。' + (N >= MAX ? '图里最多画 ' + MAX + ' 个。' : '');
    }
    function reset() { rng = mulberry32(68); studies = []; render(); }
    one.addEventListener('click', function () { add(1); });
    many.addEventListener('click', function () { add(20); });
    again.addEventListener('click', reset);
    reset();
  }

  // ---------- figure 5: a P value by shuffling the group labels ----------
  // 40 simulated patients in two groups of 20. Shuffling the labels shows how large a difference
  // between the group means turns up by chance alone; P is the share of shuffles at least as far
  // from 0 as the real difference (about 0.03 here).
  function shuffleTest(root) {
    var rngData = mulberry32(115), vals = [], i, MAX = 2000;
    for (i = 0; i < 20; i++) vals.push(Math.round((-3.2 + 2.6 * normal(rngData)) * 10) / 10);
    for (i = 0; i < 20; i++) vals.push(Math.round((-1.6 + 2.6 * normal(rngData)) * 10) / 10);
    var real = vals.map(function (_, k) { return k < 20 ? 0 : 1; });
    function diffOf(lab) {
      var a = 0, b = 0;
      vals.forEach(function (v, k) { if (lab[k]) b += v; else a += v; });
      return { a: a / 20, b: b / 20, d: a / 20 - b / 20 };
    }
    var obs = diffOf(real), rng, lab, diffs;
    var w = 360, lo = -11, hi = 5;
    function px(v) { return 16 + (v - lo) / (hi - lo) * (w - 28); }

    // top: the 40 patients, one row per group
    var svg1 = el('svg', { viewBox: '0 0 ' + w + ' 104', role: 'img', 'aria-label': '两组病人的 Δ' }, root);
    var capA = el('text', { x: 16, y: 11 }, svg1), capB = el('text', { x: 16, y: 53 }, svg1);
    el('line', { x1: 12, x2: w - 8, y1: 82, y2: 82, 'class': 'w-axis' }, svg1);
    [-10, -8, -6, -4, -2, 0, 2, 4].forEach(function (v) {
      el('line', { x1: px(v), x2: px(v), y1: 82, y2: 86, 'class': 'w-axis' }, svg1);
      el('text', { x: px(v), y: 97, 'text-anchor': 'middle' }, svg1, minus(v));
    });
    var meanA = el('line', { y1: 15, y2: 39, 'class': 'w-res' }, svg1), meanB = el('line', { y1: 57, y2: 81, 'class': 'w-res' }, svg1);
    var dots = vals.map(function (v) { return el('circle', { cx: px(v), r: 3.2 }, svg1); });

    // bottom: the differences from the shuffles
    var hh = 150, base = 116, dlo = -3.4, dhi = 3.4, bin = 0.2, nb = Math.round((dhi - dlo) / bin);
    function dx(v) { return 16 + (v - dlo) / (dhi - dlo) * (w - 28); }
    var svg2 = el('svg', { viewBox: '0 0 ' + w + ' ' + hh, role: 'img', 'aria-label': '打乱以后两组的差' }, root);
    el('line', { x1: 12, x2: w - 8, y1: base, y2: base, 'class': 'w-axis' }, svg2);
    [-3, -2, -1, 0, 1, 2, 3].forEach(function (v) {
      el('line', { x1: dx(v), x2: dx(v), y1: base, y2: base + 4, 'class': 'w-axis' }, svg2);
      el('text', { x: dx(v), y: base + 16, 'text-anchor': 'middle' }, svg2, minus(v));
    });
    el('text', { x: w / 2, y: hh - 3, 'text-anchor': 'middle' }, svg2, '打乱后两组平均值的差 (Gy)');
    [obs.d, -obs.d].forEach(function (v) { el('line', { x1: dx(v), x2: dx(v), y1: 12, y2: base, 'class': 'w-mean' }, svg2); });
    el('text', { x: dx(obs.d) + 4, y: 10 }, svg2, '实际的差 ' + minus(obs.d.toFixed(2)));
    var bars = el('g', {}, svg2);

    var ctls = h('div', 'w-ctls', root);
    var one = button(ctls, '打乱 1 次', 'primary'), many = button(ctls, '打乱 200 次'), again = button(ctls, '恢复实际分组');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');

    function shuffleOnce() {
      var order = [], j, t, k;
      for (k = 0; k < 40; k++) order.push(k);
      for (k = 39; k > 0; k--) { j = Math.floor(rng() * (k + 1)); t = order[k]; order[k] = order[j]; order[j] = t; }
      lab = vals.map(function () { return 1; });
      for (k = 0; k < 20; k++) lab[order[k]] = 0;
      diffs.push(diffOf(lab).d);
    }
    function render() {
      var cur = diffOf(lab), N = diffs.length, count = [], extreme = 0, k;
      dots.forEach(function (d, k) {
        d.setAttribute('cy', (lab[k] ? 69 : 27) + ((k * 7) % 5 - 2) * 2.2);
        d.setAttribute('class', 'w-pt' + (lab[k] ? ' new' : ''));
      });
      meanA.setAttribute('x1', px(cur.a)); meanA.setAttribute('x2', px(cur.a));
      meanB.setAttribute('x1', px(cur.b)); meanB.setAttribute('x2', px(cur.b));
      capA.textContent = (N ? '打乱后的"单纯近距离"组' : '单纯近距离组') + ': 平均 ' + minus(cur.a.toFixed(2)) + ' Gy';
      capB.textContent = (N ? '打乱后的"外照射加近距离"组' : '外照射加近距离组') + ': 平均 ' + minus(cur.b.toFixed(2)) + ' Gy';
      for (k = 0; k < nb; k++) count.push(0);
      diffs.forEach(function (d) {
        if (Math.abs(d) >= Math.abs(obs.d) - 1e-9) extreme++;
        count[Math.min(nb - 1, Math.max(0, Math.floor((d - dlo) / bin)))]++;
      });
      while (bars.firstChild) bars.removeChild(bars.firstChild);
      var unit = Math.min(7, (base - 16) / Math.max.apply(null, count.concat([1])));
      count.forEach(function (c, k) {
        if (!c) return;
        var x0 = dlo + k * bin, out = Math.abs(x0 + bin / 2) >= Math.abs(obs.d);
        el('rect', { x: dx(x0) + 0.5, width: dx(x0 + bin) - dx(x0) - 1, y: base - c * unit, height: c * unit, 'class': 'w-hist' + (out ? ' miss' : '') }, bars);
      });
      one.disabled = many.disabled = N >= MAX;
      status.textContent = N === 0
        ? '实际的分组: 单纯近距离组平均 ' + minus(obs.a.toFixed(2)) + ' Gy,外照射加近距离组平均 ' + minus(obs.b.toFixed(2)) + ' Gy,相差 ' + Math.abs(obs.d).toFixed(2) + ' Gy。点"打乱 1 次": 把 40 位病人的组别随机重新分配,再算一次两组的差。'
        : '打乱了 ' + N + ' 次。这一次两组相差 ' + Math.abs(cur.d).toFixed(2) + ' Gy。到现在,有 ' + extreme + ' 次的差不小于实际的 ' + Math.abs(obs.d).toFixed(2) + ' Gy(红色),占 ' + extreme + ' / ' + N + ' = ' + (extreme / N).toFixed(3) + '。' + (N >= MAX ? '图里最多打乱 ' + MAX + ' 次。' : '');
    }
    function reset() { rng = mulberry32(115 * 13 + 5); lab = real.slice(); diffs = []; render(); }
    one.addEventListener('click', function () { shuffleOnce(); render(); });
    many.addEventListener('click', function () { for (var k = 0; k < 200 && diffs.length < MAX; k++) shuffleOnce(); render(); });
    again.addEventListener('click', reset);
    reset();
  }

  var FIGURES = { overfit: overfit, kfold: kfold, repeatcv: repeatcv, ci: ci, shuffle: shuffleTest };
  function init() {
    document.querySelectorAll('.widget[data-widget]').forEach(function (root) {
      var make = FIGURES[root.getAttribute('data-widget')];
      if (!make || root.getAttribute('data-ready')) return;
      root.textContent = '';
      root.setAttribute('data-ready', '1');
      make(root);
    });
  }
  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
