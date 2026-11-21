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
//   fitline            set a straight line by hand (height and slope) and watch the squared error
//   sigmoid            set an S-shaped probability curve by hand (midpoint and steepness) and watch the penalty
//   signif             more patients: the P value of a weak predictor shrinks, the prediction error does not
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

  // ---------- figure 6: a straight line set by hand ----------
  // Two sliders: the height of the line (its prediction at a wall thickness of 4 mm, the middle of the
  // patients) and its slope. An intercept slider would turn the line around x = 0, far outside the data,
  // and make it hard to place. The readout is the mean squared error; one button hands over to least squares.
  function fitline(root) {
    var sc = getScene(), pts = sc.train, n = pts.length, X0 = 4;
    var mx = 0, my = 0, sxx = 0, sxy = 0;
    pts.forEach(function (p) { mx += p.x / n; my += p.y / n; });
    pts.forEach(function (p) { sxx += (p.x - mx) * (p.x - mx); sxy += (p.x - mx) * (p.y - my); });
    var bestB = sxy / sxx, bestH = my + bestB * (X0 - mx);
    function mse(hh, b) {
      var t = 0;
      pts.forEach(function (p) { var e = p.y - (hh + b * (p.x - X0)); t += e * e; });
      return t / n;
    }
    var start = { h: Math.round(my * 10) / 10, b: 0 }, cur = { h: start.h, b: start.b }, exact = false;
    var top = mse(start.h, start.b) * 1.5;

    var plot = frame(root, '散点图: 30 位病人和一条可以调的直线');
    var segs = el('g', {}, plot), path = el('path', { 'class': 'w-curve' }, plot);
    pts.forEach(function (p) { el('circle', { cx: sx(p.x), cy: sy(p.y), r: 3.4, 'class': 'w-pt' }, plot); });
    legend(root, [['pt', '30 位病人'], ['line', '你的直线'], ['res', '每位病人的误差']]);

    function slider(text, min, max, value) {
      var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
      input.type = 'range'; input.min = min; input.max = max; input.step = 0.1; input.value = value;
      input.setAttribute('aria-label', text);
      return { cap: cap, input: input };
    }
    var sh = slider('直线的高度', -6, 2, cur.h), sb = slider('直线的斜率', -1, 4, cur.b);

    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', bars, '误差平方的平均: 每位病人的误差先平方,再取平均');
    var row = h('div', 'w-bar', bars);
    h('span', 'w-bar-label', row, '这条直线');
    var fill = h('span', 'w-bar-fill train', h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
    var ctls = h('div', 'w-ctls', root), solve = button(ctls, '交给最小二乘', 'primary'), again = button(ctls, '重来');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');

    function render() {
      var f = function (x) { return cur.h + cur.b * (x - X0); }, a = cur.h - cur.b * X0, v = mse(cur.h, cur.b);
      path.setAttribute('d', 'M' + sx(XD[0]).toFixed(1) + ' ' + sy(f(XD[0])).toFixed(1) + 'L' + sx(XD[1]).toFixed(1) + ' ' + sy(f(XD[1])).toFixed(1));
      while (segs.firstChild) segs.removeChild(segs.firstChild);
      pts.forEach(function (p) { el('line', { x1: sx(p.x), x2: sx(p.x), y1: sy(p.y), y2: sy(f(p.x)), 'class': 'w-res' }, segs); });
      sh.cap.textContent = '高度(壁厚 4 mm 处的预测值): ' + minus(cur.h.toFixed(exact ? 2 : 1)) + ' Gy';
      sb.cap.textContent = '斜率(壁厚每厚 1 mm,预测值变多少): ' + minus(cur.b.toFixed(exact ? 2 : 1)) + ' Gy';
      fill.style.width = Math.min(100, v / top * 100).toFixed(1) + '%';
      val.textContent = v.toFixed(2);
      var formula = '预测 Δ = ' + minus(a.toFixed(2)) + (cur.b < 0 ? ' − ' : ' + ') + Math.abs(cur.b).toFixed(2) + ' × 壁厚';
      status.textContent = exact
        ? '最小二乘的直线: ' + formula + '。误差平方的平均是 ' + v.toFixed(2) + ',别的直线都比它大。'
        : '你的直线: ' + formula + '。';
    }
    function fromSliders() { cur.h = +sh.input.value; cur.b = +sb.input.value; exact = false; render(); }
    sh.input.addEventListener('input', fromSliders);
    sb.input.addEventListener('input', fromSliders);
    solve.addEventListener('click', function () {
      cur.h = bestH; cur.b = bestB; exact = true;
      sh.input.value = bestH.toFixed(1); sb.input.value = bestB.toFixed(1);
      render();
    });
    again.addEventListener('click', function () {
      cur.h = start.h; cur.b = start.b; exact = false;
      sh.input.value = start.h; sb.input.value = start.b;
      render();
    });
    render();
  }

  // ---------- figure 7: an S-shaped curve set by hand ----------
  // 40 simulated patients whose 5 mm plan is above 68 Gy. x: how far above (Gy). y: 1 if the optimised plan
  // brings the D2cc below 68 Gy, 0 if not. The curve is the probability of 1; it falls with x.
  // Two sliders: the midpoint (where the probability is 50%) and the steepness. The readout is the mean
  // penalty, -ln(probability given to what actually happened); one button hands over to maximum likelihood.
  function sigmoid(root) {
    var rng = mulberry32(16), pts = [];
    while (pts.length < 40) {
      var x = Math.round((0.2 + 11.8 * Math.pow(rng(), 1.3)) * 10) / 10;
      pts.push({ x: x, y: rng() < 1 / (1 + Math.exp(-(1.2 - 0.65 * x))) ? 1 : 0 });
    }
    function prob(m, s, x) { return 1 / (1 + Math.exp(s * (x - m))); }
    function loss(m, s) {
      var t = 0;
      pts.forEach(function (q) {
        var p = Math.min(1 - 1e-9, Math.max(1e-9, prob(m, s, q.x)));
        t -= Math.log(q.y ? p : 1 - p);
      });
      return t / pts.length;
    }
    // maximum likelihood by Newton's method on log-odds = b0 + b1 x
    var b0 = 0, b1 = 0;
    for (var it = 0; it < 50; it++) {
      var g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
      pts.forEach(function (q) {
        var p = 1 / (1 + Math.exp(-(b0 + b1 * q.x))), w = p * (1 - p);
        g0 += q.y - p; g1 += (q.y - p) * q.x; h00 += w; h01 += w * q.x; h11 += w * q.x * q.x;
      });
      var det = h00 * h11 - h01 * h01;
      b0 += (h11 * g0 - h01 * g1) / det; b1 += (h00 * g1 - h01 * g0) / det;
    }
    var best = { m: -b0 / b1, s: -b1 }, start = { m: 6, s: 0.3 }, cur = { m: start.m, s: start.s }, exact = false;

    var PW = 360, PH = 230, PM = { l: 36, r: 10, t: 12, b: 34 }, XMAX = 12.5;
    function px(x) { return PM.l + x / XMAX * (PW - PM.l - PM.r); }
    function py(p) { return PM.t + (1 - p) * (PH - PM.t - PM.b); }
    var svg = el('svg', { viewBox: '0 0 ' + PW + ' ' + PH, role: 'img', 'aria-label': '40 位病人的结局(是或否)和一条可以调的概率曲线' }, root);
    [0, 0.5, 1].forEach(function (v) {
      el('line', { x1: PM.l, x2: PW - PM.r, y1: py(v), y2: py(v), 'class': v === 0.5 ? 'w-zero' : 'w-grid' }, svg);
      el('text', { x: PM.l - 5, y: py(v) + 4, 'text-anchor': 'end' }, svg, v === 0.5 ? '50%' : v ? '100%' : '0');
    });
    [0, 2, 4, 6, 8, 10, 12].forEach(function (v) {
      el('line', { x1: px(v), x2: px(v), y1: PH - PM.b, y2: PH - PM.b + 4, 'class': 'w-axis' }, svg);
      el('text', { x: px(v), y: PH - PM.b + 16, 'text-anchor': 'middle' }, svg, String(v));
    });
    el('line', { x1: PM.l, x2: PW - PM.r, y1: PH - PM.b, y2: PH - PM.b, 'class': 'w-axis' }, svg);
    el('line', { x1: PM.l, x2: PM.l, y1: PM.t, y2: PH - PM.b, 'class': 'w-axis' }, svg);
    el('text', { x: (PM.l + PW - PM.r) / 2, y: PH - 3, 'text-anchor': 'middle' }, svg, '5 mm 计划超过 68 Gy 多少 (Gy)');
    var segs = el('g', {}, svg), path = el('path', { 'class': 'w-curve' }, svg);
    pts.forEach(function (q) { el('circle', { cx: px(q.x), cy: py(q.y), r: 3.4, 'class': 'w-pt' + (q.y ? '' : ' no') }, svg); });
    legend(root, [['pt', '降到了 68 Gy 以下'], ['no', '没有降到'], ['line', '模型说的概率']]);

    function slider(text, min, max, step, value) {
      var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
      input.type = 'range'; input.min = min; input.max = max; input.step = step; input.value = value;
      input.setAttribute('aria-label', text);
      return { cap: cap, input: input };
    }
    var sm = slider('曲线的中点', 0, 8, 0.1, cur.m), ss = slider('曲线的陡度', 0.1, 2, 0.05, cur.s);
    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', bars, '平均罚分: 模型给"实际发生的结果"的概率越低,罚得越重');
    var row = h('div', 'w-bar', bars);
    h('span', 'w-bar-label', row, '这条曲线');
    var fill = h('span', 'w-bar-fill train', h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
    var ctls = h('div', 'w-ctls', root), solve = button(ctls, '交给最大似然', 'primary'), again = button(ctls, '重来');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');

    function render() {
      var d = '', i, x;
      for (i = 0; i <= 250; i++) { x = XMAX * i / 250; d += (i ? 'L' : 'M') + px(x).toFixed(1) + ' ' + py(prob(cur.m, cur.s, x)).toFixed(1); }
      path.setAttribute('d', d);
      while (segs.firstChild) segs.removeChild(segs.firstChild);
      pts.forEach(function (q) { el('line', { x1: px(q.x), x2: px(q.x), y1: py(q.y), y2: py(prob(cur.m, cur.s, q.x)), 'class': 'w-res' }, segs); });
      var v = loss(cur.m, cur.s), or = Math.exp(-cur.s);
      sm.cap.textContent = '中点(概率是 50% 的位置): 超过 ' + cur.m.toFixed(exact ? 2 : 1) + ' Gy';
      ss.cap.textContent = '陡度: ' + cur.s.toFixed(2) + '(每多超 1 Gy,odds 乘以 ' + or.toFixed(2) + ')';
      fill.style.width = Math.min(100, v / 1.0 * 100).toFixed(1) + '%';
      val.textContent = v.toFixed(2);
      var formula = 'log-odds = ' + (cur.s * cur.m).toFixed(2) + ' − ' + cur.s.toFixed(2) + ' × 超出的剂量';
      status.textContent = exact
        ? '最大似然的曲线: ' + formula + '。平均罚分 ' + v.toFixed(2) + ',别的曲线都比它大。'
        : '你的曲线: ' + formula + '。';
    }
    function fromSliders() { cur.m = +sm.input.value; cur.s = +ss.input.value; exact = false; render(); }
    sm.input.addEventListener('input', fromSliders);
    ss.input.addEventListener('input', fromSliders);
    solve.addEventListener('click', function () {
      cur.m = best.m; cur.s = best.s; exact = true;
      sm.input.value = best.m.toFixed(1); ss.input.value = (Math.round(best.s * 20) / 20).toFixed(2);
      render();
    });
    again.addEventListener('click', function () {
      cur.m = start.m; cur.s = start.s; exact = false;
      sm.input.value = start.m; ss.input.value = start.s;
      render();
    });
    render();
  }

  // ---------- figure 8: significance is not accuracy ----------
  // A predictor with a small true effect (age: 0.03 Gy per year, against 2.2 Gy of noise). One slider: how many
  // patients the line is fitted on (the first n of the same 3000). Readouts: the slope with its P value, and the
  // mean absolute error on 5000 new patients, with and without the predictor.
  function lgamma(x) {
    var c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    var y = x, t = x + 5.5, sum = 1.000000000190015;
    t -= (x + 0.5) * Math.log(t);
    for (var j = 0; j < 6; j++) sum += c[j] / ++y;
    return -t + Math.log(2.5066282746310005 * sum / x);
  }
  function betacf(a, b, x) {            // continued fraction of the incomplete beta function
    var qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    d = 1 / d;
    var hh = d;
    for (var m = 1; m <= 200; m++) {
      var m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < 1e-30) d = 1e-30;
      c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1 / d; hh *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < 1e-30) d = 1e-30;
      c = 1 + aa / c; if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1 / d;
      var del = d * c;
      hh *= del;
      if (Math.abs(del - 1) < 3e-12) break;
    }
    return hh;
  }
  function ibeta(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    var bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  function pTwoSided(t, df) { return ibeta(df / 2, 0.5, df / (df + t * t)); }   // Student's t
  function signif(root) {
    function gen(n, seed) {
      var rng = mulberry32(seed), pts = [];
      for (var i = 0; i < n; i++) {
        var age = Math.max(40, Math.min(90, 64 + 9 * normal(rng)));
        pts.push({ x: Math.round(age), y: Math.round((-2.1 + 0.03 * (age - 64) + 2.2 * normal(rng)) * 10) / 10 });
      }
      return pts;
    }
    var all = gen(3000, 12), fresh = gen(5000, 1012), SIZES = [30, 100, 300, 1000, 3000];
    function fit(pts) {
      var n = pts.length, mx = 0, my = 0, sxx = 0, sxy = 0, sse = 0;
      pts.forEach(function (p) { mx += p.x / n; my += p.y / n; });
      pts.forEach(function (p) { sxx += (p.x - mx) * (p.x - mx); sxy += (p.x - mx) * (p.y - my); });
      var b = sxy / sxx, a = my - b * mx;
      pts.forEach(function (p) { var e = p.y - a - b * p.x; sse += e * e; });
      var t = b / Math.sqrt(sse / (n - 2) / sxx), eb = 0, em = 0;
      fresh.forEach(function (p) { eb += Math.abs(p.y - my) / fresh.length; em += Math.abs(p.y - a - b * p.x) / fresh.length; });
      return { a: a, b: b, p: pTwoSided(t, n - 2), base: eb, model: em };
    }

    var PW = 360, PH = 230, PM = { l: 36, r: 10, t: 12, b: 34 }, XR = [40, 90], YR = [-9.5, 5.5];
    function px(x) { return PM.l + (x - XR[0]) / (XR[1] - XR[0]) * (PW - PM.l - PM.r); }
    function py(y) { return PM.t + (YR[1] - y) / (YR[1] - YR[0]) * (PH - PM.t - PM.b); }
    var svg = el('svg', { viewBox: '0 0 ' + PW + ' ' + PH, role: 'img', 'aria-label': '散点图: 年龄和 Δ,以及拟合的直线' }, root);
    [-8, -4, 0, 4].forEach(function (v) {
      el('line', { x1: PM.l, x2: PW - PM.r, y1: py(v), y2: py(v), 'class': v === 0 ? 'w-zero' : 'w-grid' }, svg);
      el('text', { x: PM.l - 5, y: py(v) + 4, 'text-anchor': 'end' }, svg, minus(v));
    });
    [40, 50, 60, 70, 80, 90].forEach(function (v) {
      el('line', { x1: px(v), x2: px(v), y1: PH - PM.b, y2: PH - PM.b + 4, 'class': 'w-axis' }, svg);
      el('text', { x: px(v), y: PH - PM.b + 16, 'text-anchor': v === 90 ? 'end' : 'middle' }, svg, String(v));
    });
    el('line', { x1: PM.l, x2: PW - PM.r, y1: PH - PM.b, y2: PH - PM.b, 'class': 'w-axis' }, svg);
    el('line', { x1: PM.l, x2: PM.l, y1: PM.t, y2: PH - PM.b, 'class': 'w-axis' }, svg);
    el('text', { x: (PM.l + PW - PM.r) / 2, y: PH - 3, 'text-anchor': 'middle' }, svg, '年龄 (岁)');
    el('text', { x: PM.l + 5, y: PM.t + 10 }, svg, 'Δ (Gy)');
    var dots = el('g', {}, svg), path = el('path', { 'class': 'w-curve' }, svg);
    legend(root, [['pt', '建模的病人(最多画 300 位)'], ['line', '拟合的直线']]);

    var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
    input.type = 'range'; input.min = 0; input.max = SIZES.length - 1; input.step = 1; input.value = 0;
    input.setAttribute('aria-label', '建模的病人数');
    var chips = h('div', 'w-chips', root), chipB = h('span', 'w-chip done', chips), chipP = h('span', 'w-chip done', chips);
    chips.style.gridTemplateColumns = 'repeat(2, minmax(0, 1fr))';
    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', bars, '在 5000 位新病人身上,预测值和实际值平均差多少');
    function bar(label, cls) {
      var row = h('div', 'w-bar', bars);
      h('span', 'w-bar-label', row, label);
      var fill = h('span', 'w-bar-fill ' + cls, h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
      return function (v) { fill.style.width = Math.min(100, v / 2.6 * 100).toFixed(1) + '%'; val.textContent = gy(v); };
    }
    var setBase = bar('都猜平均值', 'new'), setModel = bar('用上年龄', 'train');

    function update() {
      var n = SIZES[+input.value], pts = all.slice(0, n), f = fit(pts);
      while (dots.firstChild) dots.removeChild(dots.firstChild);
      pts.slice(0, 300).forEach(function (p) { el('circle', { cx: px(p.x), cy: py(p.y), r: 2.4, 'class': 'w-pt', opacity: n > 100 ? 0.45 : 0.9 }, dots); });
      path.setAttribute('d', 'M' + px(XR[0]).toFixed(1) + ' ' + py(f.a + f.b * XR[0]).toFixed(1) + 'L' + px(XR[1]).toFixed(1) + ' ' + py(f.a + f.b * XR[1]).toFixed(1));
      cap.textContent = '建模的病人数: ' + n;
      chipB.textContent = '斜率 ' + minus(f.b.toFixed(3)) + ' Gy/岁';
      chipP.textContent = f.p < 0.001 ? 'P < 0.001' : 'P = ' + f.p.toFixed(3);
      setBase(f.base); setModel(f.model);
    }
    input.addEventListener('input', update);
    update();
  }

  var FIGURES = { overfit: overfit, kfold: kfold, repeatcv: repeatcv, ci: ci, shuffle: shuffleTest, fitline: fitline, sigmoid: sigmoid, signif: signif };
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
