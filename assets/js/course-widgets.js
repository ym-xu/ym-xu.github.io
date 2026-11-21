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
//   leak               variables chosen before cross-validation: pure noise looks predictive
//   shrink             ridge or lasso: the stronger the constraint, the smaller the coefficients
//   split              one split of a tree, placed by hand: two groups, each predicted by its own mean
//   permute            shuffle one variable among the test patients: how much worse the predictions get
//   rsquared           turn a line through the mean point: R² from 0 up to its best value and down below 0
//   threshold          move the cut-off on predicted probabilities: sensitivity against specificity, one point of the ROC curve
//   calib              a calibration plot under four kinds of stated probabilities: right, too high, too extreme, too timid
//   netbenefit         the threshold probability against the net benefit of three strategies: by the model, everyone, no one
//   samplesize         the three criteria of Riley et al. for a yes/no outcome: parameters, event proportion, expected R²
//   models             a stepped diagram (class "widget stage", the steps' text inside): one new patient through six models
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
    return axes(el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': label }, root));
  }
  function axes(svg) {
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

  // ---------- figure 9: variables chosen before cross-validation ----------
  // 60 simulated patients, an outcome that is pure noise, and up to 1000 candidate variables that are noise too.
  // The 5 candidates most correlated with the outcome go into a linear model, checked by 5-fold cross-validation.
  // "Leaky": the 5 are chosen once, using all 60 patients, and only the fitting is cross-validated.
  // "Honest": the choosing is redone inside every round, without the held-out patients.
  // One slider: the number of candidates. Nothing can be predicted, so the honest error cannot beat the mean.
  function leakScene(seed, mulberry32, normal) {
    var N = 60, K = 5, PMAX = 1000, KEEP = 5, rng = mulberry32(seed), y = [], X = [], i, j;
    for (i = 0; i < N; i++) y.push(Math.round((-2.1 + 2.2 * normal(rng)) * 10) / 10);
    for (j = 0; j < PMAX; j++) { var col = new Float64Array(N); for (i = 0; i < N; i++) col[i] = normal(rng); X.push(col); }
    var fold = []; for (i = 0; i < N; i++) fold.push(i % K);              // the patients come in random order already
    function pick(p, rows) {                                              // the KEEP candidates most correlated with y among rows
      var n = rows.length, my = 0, sy = 0, out = [];
      rows.forEach(function (r) { my += y[r] / n; });
      rows.forEach(function (r) { sy += (y[r] - my) * (y[r] - my); });
      for (var c = 0; c < p; c++) {
        var mx = 0, sxx = 0, sxy = 0;
        rows.forEach(function (r) { mx += X[c][r] / n; });
        rows.forEach(function (r) { var d = X[c][r] - mx; sxx += d * d; sxy += d * (y[r] - my); });
        out.push({ c: c, r: Math.abs(sxy / Math.sqrt(sxx * sy)) });
      }
      out.sort(function (a, b) { return b.r - a.r; });
      return out.slice(0, KEEP).map(function (o) { return o.c; });
    }
    function fit(cols, rows) {                                            // least squares with an intercept, by the normal equations
      var m = cols.length + 1, A = [], b = [], r, s, t;
      for (r = 0; r < m; r++) { A.push(new Float64Array(m)); b.push(0); }
      rows.forEach(function (row) {
        var v = [1]; cols.forEach(function (c) { v.push(X[c][row]); });
        for (r = 0; r < m; r++) { b[r] += v[r] * y[row]; for (s = 0; s < m; s++) A[r][s] += v[r] * v[s]; }
      });
      for (r = 0; r < m; r++) {                                           // Gaussian elimination
        var piv = r; for (s = r + 1; s < m; s++) if (Math.abs(A[s][r]) > Math.abs(A[piv][r])) piv = s;
        t = A[r]; A[r] = A[piv]; A[piv] = t; t = b[r]; b[r] = b[piv]; b[piv] = t;
        for (s = r + 1; s < m; s++) { var f = A[s][r] / A[r][r]; b[s] -= f * b[r]; for (t = r; t < m; t++) A[s][t] -= f * A[r][t]; }
      }
      var w = new Float64Array(m);
      for (r = m - 1; r >= 0; r--) { var acc = b[r]; for (s = r + 1; s < m; s++) acc -= A[r][s] * w[s]; w[r] = acc / A[r][r]; }
      return function (row) { var v = w[0]; cols.forEach(function (c, k) { v += w[k + 1] * X[c][row]; }); return v; };
    }
    function run(p) {
      var all = [], leaky = 0, honest = 0, base = 0;
      for (i = 0; i < N; i++) all.push(i);
      var chosen = pick(p, all);                                          // the leak: chosen with every patient, also the held-out ones
      for (var k = 0; k < K; k++) {
        var train = all.filter(function (r) { return fold[r] !== k; }), test = all.filter(function (r) { return fold[r] === k; });
        var mean = 0; train.forEach(function (r) { mean += y[r] / train.length; });
        var fl = fit(chosen, train), fh = fit(pick(p, train), train);
        test.forEach(function (r) { base += Math.abs(y[r] - mean) / N; leaky += Math.abs(y[r] - fl(r)) / N; honest += Math.abs(y[r] - fh(r)) / N; });
      }
      return { base: base, leaky: leaky, honest: honest };
    }
    return { run: run, sizes: [10, 30, 100, 300, 1000] };
  }
  function leak(root) {
    var sc = leakScene(33, mulberry32, normal);
    var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
    input.type = 'range'; input.min = 0; input.max = sc.sizes.length - 1; input.step = 1; input.value = 0;
    input.setAttribute('aria-label', '候选变量的个数');
    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', bars, '5 折交叉验证的平均误差(60 位病人;结局和所有候选变量都是随机数)');
    function bar(label, cls) {
      var row = h('div', 'w-bar', bars);
      h('span', 'w-bar-label', row, label);
      var fill = h('span', 'w-bar-fill ' + cls, h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
      return function (v) { fill.style.width = Math.min(100, v / 2.6 * 100).toFixed(1) + '%'; val.textContent = gy(v); };
    }
    var setBase = bar('都猜平均值', 'new'), setLeaky = bar('先挑变量,再验证', 'train'), setHonest = bar('挑变量也放进验证', 'cv');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');
    function update() {
      var p = sc.sizes[+input.value], r = sc.run(p);
      cap.textContent = '候选变量: ' + p + ' 个(从中挑 5 个)';
      setBase(r.base); setLeaky(r.leaky); setHonest(r.honest);
      status.textContent = r.leaky < r.base
        ? '先挑变量再验证,看起来比"都猜平均值"准了 ' + (r.base - r.leaky).toFixed(2) + ' Gy。这些变量是随机数,这个"准"是假的。'
        : '候选变量还少,挑出来的 5 个和结局碰巧相关得不多,假象还不明显。';
    }
    input.addEventListener('input', update);
    update();
  }

  // ---------- figure 10: a constraint on the coefficients ----------
  // 40 simulated patients and 12 standardised variables: the first two have a real effect (1.5 and -0.8),
  // the other ten are noise. Ridge (closed form) or lasso (coordinate descent), fitted in the browser.
  // A slider for the strength of the constraint and two buttons for its kind. One bar per coefficient,
  // and the mean absolute error on 2000 new patients, without the constraint and with it.
  function shrinkScene(seed, mulberry32, normal) {
    var N = 40, P = 12, M = 2000, TRUE = [1.5, -0.8], rng = mulberry32(seed), i, j;
    function draw(n) {
      var X = [], y = [];
      for (i = 0; i < n; i++) {
        var row = []; for (j = 0; j < P; j++) row.push(normal(rng));
        X.push(row); y.push(TRUE[0] * row[0] + TRUE[1] * row[1] + 2.0 * normal(rng));
      }
      return { X: X, y: y };
    }
    var tr = draw(N), te = draw(M), mean = 0;
    // centre the outcome and standardise the columns with the training patients' means and SDs
    tr.y.forEach(function (v) { mean += v / N; });
    var mu = [], sd = [];
    for (j = 0; j < P; j++) {
      var m = 0, s = 0;
      tr.X.forEach(function (r) { m += r[j] / N; });
      tr.X.forEach(function (r) { s += (r[j] - m) * (r[j] - m) / N; });
      mu.push(m); sd.push(Math.sqrt(s));
    }
    function z(r) { return r.map(function (v, k) { return (v - mu[k]) / sd[k]; }); }
    var Z = tr.X.map(z), yc = tr.y.map(function (v) { return v - mean; }), ZT = te.X.map(z);
    function ridge(lam) {                       // (Z'Z + lam I) b = Z'y, by Gaussian elimination
      var A = [], b = [], r, s, t;
      for (r = 0; r < P; r++) { A.push(new Float64Array(P)); b.push(0); }
      Z.forEach(function (row, n) { for (r = 0; r < P; r++) { b[r] += row[r] * yc[n]; for (s = 0; s < P; s++) A[r][s] += row[r] * row[s]; } });
      for (r = 0; r < P; r++) A[r][r] += lam;
      for (r = 0; r < P; r++) {
        var piv = r; for (s = r + 1; s < P; s++) if (Math.abs(A[s][r]) > Math.abs(A[piv][r])) piv = s;
        t = A[r]; A[r] = A[piv]; A[piv] = t; t = b[r]; b[r] = b[piv]; b[piv] = t;
        for (s = r + 1; s < P; s++) { var f = A[s][r] / A[r][r]; b[s] -= f * b[r]; for (t = r; t < P; t++) A[s][t] -= f * A[r][t]; }
      }
      var w = new Array(P);
      for (r = P - 1; r >= 0; r--) { var acc = b[r]; for (s = r + 1; s < P; s++) acc -= A[r][s] * w[s]; w[r] = acc / A[r][r]; }
      return w;
    }
    function lasso(lam) {                       // coordinate descent on (1/2n)|y - Zb|^2 + lam |b|_1
      var w = new Array(P).fill(0), res = yc.slice(), it, k, n;
      for (it = 0; it < 400; it++) {
        var change = 0;
        for (k = 0; k < P; k++) {
          var rho = 0;
          for (n = 0; n < N; n++) rho += Z[n][k] * (res[n] + Z[n][k] * w[k]) / N;      // columns have variance 1
          var nw = rho > lam ? rho - lam : rho < -lam ? rho + lam : 0;
          if (nw !== w[k]) { for (n = 0; n < N; n++) res[n] += Z[n][k] * (w[k] - nw); change = Math.max(change, Math.abs(nw - w[k])); w[k] = nw; }
        }
        if (change < 1e-9) break;
      }
      return w;
    }
    function error(w) {
      var e = 0;
      ZT.forEach(function (row, n) { var p = mean; for (var k = 0; k < P; k++) p += w[k] * row[k]; e += Math.abs(te.y[n] - p) / M; });
      return e;
    }
    return { P: P, ridge: ridge, lasso: lasso, error: error, RIDGE: [0, 3, 10, 30, 60, 120, 250, 500], LASSO: [0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.7, 1.0] };
  }
  function shrink(root) {
    var sc = shrinkScene(45, mulberry32, normal), kind = 'lasso';
    var LABELS = ['壁厚', '次数'];
    for (var q = 1; q <= sc.P - 2; q++) LABELS.push('随机 ' + q);
    var ROW = 17, PW = 360, PH = sc.P * ROW + 26, X0 = 70, XR = 1.9;          // coefficients between -1.9 and 1.9
    function px(v) { return X0 + (v + XR) / (2 * XR) * (PW - X0 - 10); }
    var svg = el('svg', { viewBox: '0 0 ' + PW + ' ' + PH, role: 'img', 'aria-label': '每个变量的系数' }, root);
    [-1, 0, 1].forEach(function (v) {
      el('line', { x1: px(v), x2: px(v), y1: 4, y2: sc.P * ROW + 6, 'class': v === 0 ? 'w-axis' : 'w-grid' }, svg);
      el('text', { x: px(v), y: PH - 6, 'text-anchor': 'middle' }, svg, minus(v));
    });
    var bars = LABELS.map(function (name, k) {
      el('text', { x: X0 - 8, y: 6 + k * ROW + 12, 'text-anchor': 'end' }, svg, name);
      return el('rect', { y: 6 + k * ROW + 3, height: ROW - 6, 'class': k < 2 ? 'w-coef real' : 'w-coef' }, svg);
    });
    var ctls = h('div', 'w-ctls', root), bLasso = button(ctls, 'LASSO', 'primary'), bRidge = button(ctls, '岭回归');
    var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
    input.type = 'range'; input.min = 0; input.max = sc.RIDGE.length - 1; input.step = 1; input.value = 0;
    input.setAttribute('aria-label', '约束的强度');
    var box = h('div', 'w-bars', root);
    box.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', box, '在 2000 位新病人身上,预测值和实际值平均差多少');
    function bar(label, cls) {
      var row = h('div', 'w-bar', box);
      h('span', 'w-bar-label', row, label);
      var fill = h('span', 'w-bar-fill ' + cls, h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
      return function (v) { fill.style.width = Math.min(100, v / 2.6 * 100).toFixed(1) + '%'; val.textContent = gy(v); };
    }
    var setFree = bar('不加约束', 'new'), setNow = bar('现在的强度', 'train');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');
    var free = sc.error(sc.ridge(0));
    function update() {
      var level = +input.value, w = kind === 'lasso' ? sc.lasso(sc.LASSO[level]) : sc.ridge(sc.RIDGE[level]);
      bars.forEach(function (r, k) {
        var a = px(0), b = px(Math.max(-XR, Math.min(XR, w[k])));
        r.setAttribute('x', Math.min(a, b)); r.setAttribute('width', Math.max(Math.abs(b - a), w[k] === 0 ? 0 : 0.8));
      });
      var kept = w.filter(function (v) { return v !== 0; }).length;
      cap.textContent = '约束的强度: ' + level + (level === 0 ? '(不加约束)' : level === sc.RIDGE.length - 1 ? '(最强)' : '');
      setFree(free); setNow(sc.error(w));
      bLasso.className = 'w-btn' + (kind === 'lasso' ? ' primary' : '');
      bRidge.className = 'w-btn' + (kind === 'ridge' ? ' primary' : '');
      status.textContent = (kind === 'lasso' ? 'LASSO' : '岭回归') + ': ' + sc.P + ' 个系数里,' + kept + ' 个不是 0。';
    }
    bLasso.addEventListener('click', function () { kind = 'lasso'; update(); });
    bRidge.addEventListener('click', function () { kind = 'ridge'; update(); });
    input.addEventListener('input', update);
    update();
  }

  // ---------- figure 11: one split of a tree ----------
  // The 30 patients of the other figures. One slider: the wall thickness at which they are split in two.
  // Each side is predicted by its own mean; the readout is the mean squared error, as in "fitline".
  function splitTree(root) {
    var sc = getScene(), pts = sc.train, n = pts.length;
    function at(c) {
      var l = pts.filter(function (p) { return p.x <= c; }), r = pts.filter(function (p) { return p.x > c; });
      var ml = l.length ? mean(l.map(function (p) { return p.y; })) : 0, mr = r.length ? mean(r.map(function (p) { return p.y; })) : 0, e = 0;
      pts.forEach(function (p) { var d = p.y - (p.x <= c ? ml : mr); e += d * d / n; });
      return { ml: ml, mr: mr, nl: l.length, nr: r.length, mse: e };
    }
    var best = null, c;
    for (c = 20; c <= 60; c++) { var t = at(c / 10); if (t.nl && t.nr && (!best || t.mse < best.mse)) { best = t; best.c = c / 10; } }
    var my = mean(pts.map(function (p) { return p.y; })), flat = 0;
    pts.forEach(function (p) { flat += (p.y - my) * (p.y - my) / n; });

    var plot = frame(root, '散点图: 30 位病人按壁厚分成两组,每组用自己的平均值预测');
    var segs = el('g', {}, plot), cutLine = el('line', { 'class': 'w-mean' }, plot);
    var left = el('line', { 'class': 'w-curve' }, plot), right = el('line', { 'class': 'w-curve' }, plot);
    pts.forEach(function (p) { el('circle', { cx: sx(p.x), cy: sy(p.y), r: 3.4, 'class': 'w-pt' }, plot); });
    legend(root, [['pt', '30 位病人'], ['line', '每一组的预测值'], ['res', '每位病人的误差']]);
    var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
    input.type = 'range'; input.min = 2; input.max = 6; input.step = 0.1; input.value = 2.5;
    input.setAttribute('aria-label', '分叉点的壁厚');
    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', bars, '误差平方的平均');
    function bar(label, cls) {
      var row = h('div', 'w-bar', bars);
      h('span', 'w-bar-label', row, label);
      var fill = h('span', 'w-bar-fill ' + cls, h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
      return function (v) { fill.style.width = Math.min(100, v / (flat * 1.1) * 100).toFixed(1) + '%'; val.textContent = v.toFixed(2); };
    }
    var setFlat = bar('不分(都猜平均值)', 'new'), setNow = bar('分成两组', 'train');
    var ctls = h('div', 'w-ctls', root), solve = button(ctls, '找最好的分叉点', 'primary');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');
    function update() {
      var cut = +input.value, t = at(cut);
      cutLine.setAttribute('x1', sx(cut)); cutLine.setAttribute('x2', sx(cut)); cutLine.setAttribute('y1', sy(YD[0])); cutLine.setAttribute('y2', sy(YD[1]));
      left.setAttribute('x1', sx(XD[0])); left.setAttribute('x2', sx(cut)); left.setAttribute('y1', sy(t.ml)); left.setAttribute('y2', sy(t.ml));
      right.setAttribute('x1', sx(cut)); right.setAttribute('x2', sx(XD[1])); right.setAttribute('y1', sy(t.mr)); right.setAttribute('y2', sy(t.mr));
      while (segs.firstChild) segs.removeChild(segs.firstChild);
      pts.forEach(function (p) { el('line', { x1: sx(p.x), x2: sx(p.x), y1: sy(p.y), y2: sy(p.x <= cut ? t.ml : t.mr), 'class': 'w-res' }, segs); });
      cap.textContent = '分叉点: 壁厚 ' + cut.toFixed(1) + ' mm';
      setFlat(flat); setNow(t.mse);
      status.textContent = '壁厚不超过 ' + cut.toFixed(1) + ' mm 的 ' + t.nl + ' 位,都猜 ' + minus(t.ml.toFixed(2)) + ' Gy;超过的 ' + t.nr + ' 位,都猜 ' + minus(t.mr.toFixed(2)) + ' Gy。'
        + (Math.abs(cut - best.c) < 0.05 ? '这是误差最小的分法。' : '');
    }
    input.addEventListener('input', update);
    solve.addEventListener('click', function () { input.value = best.c; update(); });
    update();
  }

  // ---------- figure 12: permutation importance ----------
  // The scene of the figure "permute": 80 patients to fit a straight-line model on three variables (wall
  // thickness, number of fractions, age; age has no effect), 40 more to test it on. shuffled(v, k) gives the
  // test patients with variable v dealt out again among them; the k-th shuffle of a variable is always the same.
  function permScene(seed, mulberry32, normal) {
    var rng = mulberry32(seed), NTR = 80, NTE = 40, rows = [], i, r, s, t;
    for (i = 0; i < NTR + NTE; i++) {
      var wall = Math.round(Math.exp(Math.log(3.6) + 0.30 * normal(rng)) * 10) / 10, fx = rng() < 0.5 ? 2 : 3, age = Math.round(64 + 9 * normal(rng));
      rows.push({ x: [wall, fx, age], y: Math.round(((fx / 2.5) * 1.6 * (wall - 5) + 2.0 * normal(rng)) * 10) / 10 });
    }
    var train = rows.slice(0, NTR), test = rows.slice(NTR), m = 4, A = [], b = [];
    for (r = 0; r < m; r++) { A.push(new Float64Array(m)); b.push(0); }
    train.forEach(function (p) {
      var v = [1].concat(p.x);
      for (r = 0; r < m; r++) { b[r] += v[r] * p.y; for (s = 0; s < m; s++) A[r][s] += v[r] * v[s]; }
    });
    for (r = 0; r < m; r++) {                                             // Gaussian elimination
      var piv = r; for (s = r + 1; s < m; s++) if (Math.abs(A[s][r]) > Math.abs(A[piv][r])) piv = s;
      t = A[r]; A[r] = A[piv]; A[piv] = t; t = b[r]; b[r] = b[piv]; b[piv] = t;
      for (s = r + 1; s < m; s++) { var f = A[s][r] / A[r][r]; b[s] -= f * b[r]; for (t = r; t < m; t++) A[s][t] -= f * A[r][t]; }
    }
    var w = new Float64Array(m);
    for (r = m - 1; r >= 0; r--) { var acc = b[r]; for (s = r + 1; s < m; s++) acc -= A[r][s] * w[s]; w[r] = acc / A[r][r]; }
    function predict(x) { return w[0] + w[1] * x[0] + w[2] * x[1] + w[3] * x[2]; }
    function shuffled(v, k) {
      var g = mulberry32(seed * 100 + v * 1000 + k), order = test.map(function (_, j) { return j; }), j, q;
      for (i = order.length - 1; i > 0; i--) { j = Math.floor(g() * (i + 1)); q = order[i]; order[i] = order[j]; order[j] = q; }
      return test.map(function (p, n) { var x = p.x.slice(); x[v] = test[order[n]].x[v]; return x; });
    }
    return { test: test, predict: predict, shuffled: shuffled, coef: w };
  }
  function permute(root) {
    var sc = permScene(67, mulberry32, normal), test = sc.test, n = test.length, names = ['壁厚', '次数', '年龄'];
    var base = test.map(function (p) { return sc.predict(p.x); });
    function maeOf(pred) { var e = 0; test.forEach(function (p, i) { e += Math.abs(p.y - pred[i]) / n; }); return e; }
    var baseErr = maeOf(base), clicks = [0, 0, 0], sums = [0, 0, 0];
    var w = 360, hgt = 260, m = { l: 36, r: 8, t: 10, b: 34 }, lo = -9.5, hi = 5.5;
    function px(v) { return m.l + (v - lo) / (hi - lo) * (w - m.l - m.r); }
    function py(v) { return m.t + (hi - v) / (hi - lo) * (hgt - m.t - m.b); }
    var svg = el('svg', { viewBox: '0 0 ' + w + ' ' + hgt, role: 'img', 'aria-label': '散点图: 40 位病人,模型的预测值和实际的 Δ' }, root);
    [-8, -4, 0, 4].forEach(function (v) {
      el('line', { x1: m.l, x2: w - m.r, y1: py(v), y2: py(v), 'class': 'w-grid' }, svg);
      el('text', { x: m.l - 5, y: py(v) + 4, 'text-anchor': 'end' }, svg, minus(v));
      el('line', { x1: px(v), x2: px(v), y1: hgt - m.b, y2: hgt - m.b + 4, 'class': 'w-axis' }, svg);
      el('text', { x: px(v), y: hgt - m.b + 16, 'text-anchor': 'middle' }, svg, minus(v));
    });
    el('line', { x1: m.l, x2: w - m.r, y1: hgt - m.b, y2: hgt - m.b, 'class': 'w-axis' }, svg);
    el('line', { x1: m.l, x2: m.l, y1: m.t, y2: hgt - m.b, 'class': 'w-axis' }, svg);
    el('text', { x: (m.l + w - m.r) / 2, y: hgt - 3, 'text-anchor': 'middle' }, svg, '模型的预测值 (Gy)');
    el('text', { x: m.l + 5, y: m.t + 10 }, svg, '实际的 Δ (Gy)');
    el('line', { x1: px(lo), x2: px(hi), y1: py(lo), y2: py(hi), 'class': 'w-mean' }, svg);
    var moves = el('g', {}, svg);
    var dots = test.map(function (p, i) { return el('circle', { cx: px(base[i]), cy: py(p.y), r: 3.4, 'class': 'w-pt' }, svg); });
    legend(root, [['pt', '40 位没参与建模的病人'], ['dash', '预测值 = 实际值'], ['hres', '打乱以后预测值挪了多远']]);
    var ctls = h('div', 'w-ctls', root);
    var btns = names.map(function (nm, v) { return button(ctls, '打乱' + nm, v === 0 ? 'primary' : ''); }), back = button(ctls, '恢复');
    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', bars, '预测值和实际值平均差多少');
    function bar(label, cls) {
      var row = h('div', 'w-bar', bars);
      h('span', 'w-bar-label', row, label);
      var fill = h('span', 'w-bar-fill ' + cls, h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
      return function (v) { fill.style.width = Math.min(100, v / 3.6 * 100).toFixed(1) + '%'; val.textContent = gy(v); };
    }
    var setBase = bar('原来', 'new'), setNow = bar('打乱以后', 'train');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');
    function show(pred, text) {
      while (moves.firstChild) moves.removeChild(moves.firstChild);
      test.forEach(function (p, i) {
        if (Math.abs(pred[i] - base[i]) > 0.02) el('line', { x1: px(base[i]), x2: px(pred[i]), y1: py(p.y), y2: py(p.y), 'class': 'w-res' }, moves);
        dots[i].setAttribute('cx', px(Math.max(lo, Math.min(hi, pred[i]))));
      });
      setBase(baseErr); setNow(maeOf(pred));
      status.textContent = text;
    }
    btns.forEach(function (b, v) {
      b.addEventListener('click', function () {
        clicks[v]++;
        var pred = sc.shuffled(v, clicks[v]).map(sc.predict), d = maeOf(pred) - baseErr;
        sums[v] += d;
        show(pred, '第 ' + clicks[v] + ' 次打乱' + names[v] + ': 误差' + (d >= 0.005 ? '多了 ' + d.toFixed(2) : d <= -0.005 ? '少了 ' + (-d).toFixed(2) : '没有变') + (Math.abs(d) >= 0.005 ? ' Gy' : '')
          + '。' + (clicks[v] > 1 ? '打乱' + names[v] + '的 ' + clicks[v] + ' 次,平均' + (Math.abs(sums[v] / clicks[v]) < 0.005 ? '没有变。' : (sums[v] > 0 ? '多 ' : '少 ') + Math.abs(sums[v] / clicks[v]).toFixed(2) + ' Gy。') : ''));
      });
    });
    back.addEventListener('click', function () { clicks = [0, 0, 0]; sums = [0, 0, 0]; show(base, START); });
    var START = '每个点是一位病人: 越靠近斜线,预测越准。点一个按钮,把这 40 位病人的那一个变量随机互换,别的不动。';
    show(base, START);
  }

  // ---------- figure 13: R² of a line that is turned ----------
  // One slider: the slope of a line through the mean point of the 30 patients. Slope 0 is the baseline (every
  // patient gets the mean, R² = 0), the least-squares slope gives the largest R², and a line more than twice
  // as steep as that has a larger squared error than the baseline: R² below 0.
  function rsquared(root) {
    var sc = getScene(), pts = sc.train, n = pts.length;
    var mx = mean(pts.map(function (p) { return p.x; })), my = mean(pts.map(function (p) { return p.y; }));
    function mseOf(b) { var e = 0; pts.forEach(function (p) { var d = p.y - my - b * (p.x - mx); e += d * d / n; }); return e; }
    var flat = mseOf(0), top = mseOf(4);
    var plot = frame(root, '散点图: 30 位病人,一条经过平均点、可以转动的线');
    var segs = el('g', {}, plot);
    el('line', { x1: sx(XD[0]), x2: sx(XD[1]), y1: sy(my), y2: sy(my), 'class': 'w-mean' }, plot);
    var line = el('line', { 'class': 'w-curve' }, plot);
    pts.forEach(function (p) { el('circle', { cx: sx(p.x), cy: sy(p.y), r: 3.4, 'class': 'w-pt' }, plot); });
    legend(root, [['pt', '30 位病人'], ['dash', '都猜平均值'], ['line', '模型的线'], ['res', '每位病人的误差']]);
    var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
    input.type = 'range'; input.min = 0; input.max = 4; input.step = 0.1; input.value = 0;
    input.setAttribute('aria-label', '线的斜率');
    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', bars, '误差平方的平均');
    function bar(label, cls) {
      var row = h('div', 'w-bar', bars);
      h('span', 'w-bar-label', row, label);
      var fill = h('span', 'w-bar-fill ' + cls, h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
      return function (v) { fill.style.width = Math.min(100, v / (top * 1.03) * 100).toFixed(1) + '%'; val.textContent = v.toFixed(2); };
    }
    var setFlat = bar('都猜平均值', 'new'), setNow = bar('这条线', 'train');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');
    function update() {
      var b = +input.value, e = mseOf(b), r2 = 1 - e / flat;
      line.setAttribute('x1', sx(XD[0])); line.setAttribute('x2', sx(XD[1]));
      line.setAttribute('y1', sy(my + b * (XD[0] - mx))); line.setAttribute('y2', sy(my + b * (XD[1] - mx)));
      while (segs.firstChild) segs.removeChild(segs.firstChild);
      pts.forEach(function (p) { el('line', { x1: sx(p.x), x2: sx(p.x), y1: sy(p.y), y2: sy(my + b * (p.x - mx)), 'class': 'w-res' }, segs); });
      cap.textContent = '线的斜率: ' + b.toFixed(1);
      setFlat(flat); setNow(e);
      status.textContent = 'R² = 1 − ' + e.toFixed(2) + ' / ' + flat.toFixed(2) + ' = ' + minus(r2.toFixed(2)) + '。'
        + (b === 0 ? '这条线就是都猜平均值。' : r2 >= 0.005 ? '比都猜平均值好: 误差平方少了 ' + Math.round(r2 * 100) + '%。'
          : r2 > -0.005 ? '和都猜平均值一样。' : '比都猜平均值还差: R² 是负的。');
    }
    input.addEventListener('input', update);
    update();
  }

  // ---------- figure 14: a cut-off on predicted probabilities ----------
  // 40 patients with a predicted probability each: 12 had the event, 28 did not. One slider: the cut-off.
  // Above: the patients along the probability axis, the cut-off as a line. Below: the ROC curve with the
  // point of this cut-off. Readout: sensitivity and specificity.
  function probScene() {                                                  // the same 40 patients in "threshold" and "netbenefit"
    var rng = mulberry32(17), pts = [], i;
    for (i = 0; i < 40; i++) { var ev = i < 12, z = (ev ? 0.7 : -1.5) + 1.15 * normal(rng); pts.push({ ev: ev, p: Math.round(100 / (1 + Math.exp(-z))) / 100 }); }
    return pts;
  }
  function thresholdFig(root) {
    var pts = probScene(), i;
    var evs = pts.filter(function (q) { return q.ev; }), nos = pts.filter(function (q) { return !q.ev; });
    function at(t) {
      var tp = evs.filter(function (q) { return q.p >= t; }).length, tn = nos.filter(function (q) { return q.p < t; }).length;
      return { tp: tp, tn: tn, sens: tp / evs.length, spec: tn / nos.length };
    }
    var conc = 0;
    evs.forEach(function (a) { nos.forEach(function (b) { conc += a.p > b.p ? 1 : a.p === b.p ? 0.5 : 0; }); });
    var auc = conc / (evs.length * nos.length);

    var w = 360;
    function px(v) { return 18 + v * (w - 36); }
    var svg1 = el('svg', { viewBox: '0 0 ' + w + ' 136', role: 'img', 'aria-label': '40 位病人按模型给的概率排开,竖线是阈值' }, root);
    el('text', { x: 18, y: 12 }, svg1, '降下来的 12 位');
    el('text', { x: 18, y: 60 }, svg1, '没降下来的 28 位');
    el('line', { x1: 14, x2: w - 14, y1: 100, y2: 100, 'class': 'w-axis' }, svg1);
    [0, 0.2, 0.4, 0.6, 0.8, 1].forEach(function (v) {
      el('line', { x1: px(v), x2: px(v), y1: 100, y2: 104, 'class': 'w-axis' }, svg1);
      el('text', { x: px(v), y: 115, 'text-anchor': 'middle' }, svg1, v.toFixed(1));
    });
    el('text', { x: w / 2, y: 132, 'text-anchor': 'middle' }, svg1, '模型给的概率');
    var cut = el('line', { y1: 4, y2: 100, 'class': 'w-mean' }, svg1);
    var dots = pts.map(function (q, k) { return el('circle', { cx: px(q.p), cy: (q.ev ? 30 : 78) + ((k * 7) % 5 - 2) * 3, r: 3.4 }, svg1); });

    var S = 176, ox = 92, oy = 12;                                        // the ROC square
    function rx(v) { return ox + v * S; }
    function ry(v) { return oy + (1 - v) * S; }
    var svg2 = el('svg', { viewBox: '0 0 ' + w + ' 226', role: 'img', 'aria-label': 'ROC 曲线: 每个阈值是曲线上的一个点' }, root);
    el('rect', { x: ox, y: oy, width: S, height: S, 'class': 'w-box' }, svg2);
    el('line', { x1: rx(0), y1: ry(0), x2: rx(1), y2: ry(1), 'class': 'w-zero' }, svg2);
    [0, 0.5, 1].forEach(function (v) {
      el('text', { x: rx(v), y: oy + S + 14, 'text-anchor': 'middle' }, svg2, v === 0.5 ? '0.5' : String(v));
      el('text', { x: ox - 6, y: ry(v) + 4, 'text-anchor': 'end' }, svg2, v === 0.5 ? '0.5' : String(v));
    });
    el('text', { x: ox + S / 2, y: oy + S + 30, 'text-anchor': 'middle' }, svg2, '1 − 特异度');
    el('text', { x: ox - 34, y: oy + S / 2, 'text-anchor': 'middle', transform: 'rotate(-90 ' + (ox - 34) + ' ' + (oy + S / 2) + ')' }, svg2, '灵敏度');
    var cuts = pts.map(function (q) { return q.p; }).sort(function (a, b) { return b - a; }), d = 'M' + rx(0) + ' ' + ry(0), last = null;
    cuts.forEach(function (t) { if (t === last) return; last = t; var a = at(t); d += 'L' + rx(1 - a.spec).toFixed(1) + ' ' + ry(a.sens).toFixed(1); });
    el('path', { d: d + 'L' + rx(1) + ' ' + ry(1), 'class': 'w-curve' }, svg2);
    el('text', { x: rx(1) - 6, y: ry(0) - 8, 'text-anchor': 'end' }, svg2, 'AUC ' + auc.toFixed(2));
    var here = el('circle', { r: 5.5, 'class': 'w-pt' }, svg2);

    legend(root, [['pt', '降下来的病人'], ['new', '没降下来的病人'], ['dash', '阈值'], ['line', 'ROC 曲线']]);
    var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
    input.type = 'range'; input.min = 0.05; input.max = 0.95; input.step = 0.05; input.value = 0.5;
    input.setAttribute('aria-label', '阈值');
    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    function bar(label, cls) {
      var row = h('div', 'w-bar', bars);
      h('span', 'w-bar-label', row, label);
      var fill = h('span', 'w-bar-fill ' + cls, h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
      return function (v) { fill.style.width = (v * 100).toFixed(1) + '%'; val.textContent = Math.round(v * 100) + '%'; };
    }
    var setSens = bar('灵敏度', 'train'), setSpec = bar('特异度', 'new');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');
    function update() {
      var t = Math.round(+input.value * 100) / 100, a = at(t);
      cut.setAttribute('x1', px(t)); cut.setAttribute('x2', px(t));
      dots.forEach(function (c, k) { c.setAttribute('class', 'w-pt' + (pts[k].ev ? '' : ' new')); });
      here.setAttribute('cx', rx(1 - a.spec)); here.setAttribute('cy', ry(a.sens));
      cap.textContent = '阈值: 概率达到 ' + t.toFixed(2) + ' 就判为"会降"';
      setSens(a.sens); setSpec(a.spec);
      status.textContent = '降下来的 ' + evs.length + ' 位里,判对 ' + a.tp + ' 位,漏掉 ' + (evs.length - a.tp) + ' 位。没降下来的 ' + nos.length + ' 位里,判对 ' + a.tn + ' 位,误判 ' + (nos.length - a.tn) + ' 位。';
    }
    input.addEventListener('input', update);
    update();
  }

  // ---------- figure 15: reading a calibration plot ----------
  // 1500 simulated patients with a true probability each and an outcome drawn from it. Four buttons change
  // what the "model" states: the true probability, one shifted upwards, one stretched away from the middle
  // (too extreme) and one squeezed towards it (too timid). The plot groups the patients by stated probability
  // (eight groups) and shows the observed proportion in each; the readout is O/E and the calibration slope.
  function calibFig(root) {
    var rng = mulberry32(204), n = 1500, MID = -1.2, pts = [], i;
    function inv(z) { return 1 / (1 + Math.exp(-z)); }
    for (i = 0; i < n; i++) { var t = MID + 1.4 * normal(rng); pts.push({ t: t, y: rng() < inv(t) ? 1 : 0 }); }
    var KINDS = [
      { name: '准的', a: 0, b: 1, say: '点落在斜线上: 模型说多少,实际就是多少。' },
      { name: '整体偏高', a: 0.9, b: 1, say: '点都在斜线下面: 每一组实际出事的比例都比模型说的低。O/E 小于 1,斜率还是 1 左右。' },
      { name: '说得太极端', a: null, b: 1.9, say: '总数对得上,O/E 看不出毛病。可是左边的点在斜线上面,右边的点在下面: 模型说很低的,实际没那么低;说很高的,实际没那么高。斜率小于 1。过拟合的模型是这个样子。' },
      { name: '说得太保守', a: null, b: 0.55, say: '总数也对得上。左边的点在斜线下面,右边的点在上面: 模型的概率都挤在中间,实际的差别更大。斜率大于 1。' }
    ];
    function stated(k) { return pts.map(function (q) { return MID + k.b * (q.t - MID) + k.a; }); }   // on the logit scale
    function total(k) { var e = 0; stated(k).forEach(function (z) { e += inv(z); }); return e; }
    KINDS.forEach(function (k) {                                          // a stretched or squeezed model still states the right total
      if (k.a !== null) return;
      var lo = -5, hi = 5, want = total(KINDS[0]), it;
      for (it = 0; it < 40; it++) { k.a = (lo + hi) / 2; if (total(k) < want) lo = k.a; else hi = k.a; }
    });
    function slopeOf(m) {                                                 // logistic regression of the outcome on m, by Newton steps
      var a = 0, b = 1, it, j;
      for (it = 0; it < 25; it++) {
        var g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
        for (j = 0; j < n; j++) { var pr = inv(a + b * m[j]), wv = pr * (1 - pr), r = pts[j].y - pr; g0 += r; g1 += r * m[j]; h00 += wv; h01 += wv * m[j]; h11 += wv * m[j] * m[j]; }
        var det = h00 * h11 - h01 * h01; a += (h11 * g0 - h01 * g1) / det; b += (h00 * g1 - h01 * g0) / det;
      }
      return b;
    }
    var w = 360, hgt = 300, m = { l: 40, r: 10, t: 10, b: 38 };
    function px(v) { return m.l + v * (w - m.l - m.r); }
    function py(v) { return m.t + (1 - v) * (hgt - m.t - m.b); }
    var svg = el('svg', { viewBox: '0 0 ' + w + ' ' + hgt, role: 'img', 'aria-label': '校准图: 模型说的概率和实际出事的比例' }, root);
    [0, 0.2, 0.4, 0.6, 0.8, 1].forEach(function (v) {
      el('line', { x1: px(0), x2: px(1), y1: py(v), y2: py(v), 'class': 'w-grid' }, svg);
      el('text', { x: m.l - 5, y: py(v) + 4, 'text-anchor': 'end' }, svg, v.toFixed(1));
      el('line', { x1: px(v), x2: px(v), y1: py(0), y2: py(0) + 4, 'class': 'w-axis' }, svg);
      el('text', { x: px(v), y: py(0) + 16, 'text-anchor': 'middle' }, svg, v.toFixed(1));
    });
    el('line', { x1: px(0), x2: px(1), y1: py(0), y2: py(0), 'class': 'w-axis' }, svg);
    el('line', { x1: px(0), x2: px(0), y1: py(0), y2: py(1), 'class': 'w-axis' }, svg);
    el('text', { x: (px(0) + px(1)) / 2, y: hgt - 4, 'text-anchor': 'middle' }, svg, '模型说的概率(每组的平均)');
    el('text', { x: px(0) + 5, y: m.t + 10 }, svg, '实际出事的比例');
    el('line', { x1: px(0), y1: py(0), x2: px(1), y2: py(1), 'class': 'w-mean' }, svg);
    var path = el('path', { 'class': 'w-curve' }, svg), marks = el('g', {}, svg);
    legend(root, [['pt', '一组病人(按模型说的概率从低到高分成 8 组)'], ['dash', '模型说的 = 实际的']]);
    var ctls = h('div', 'w-ctls', root), status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');
    var btns = KINDS.map(function (k, idx) { return button(ctls, k.name, idx === 0 ? 'primary' : ''); });
    function show(idx) {
      var k = KINDS[idx], mm = stated(k), order = pts.map(function (_, j) { return j; }), G = 8, d = '', E = 0, O = 0, g, j;
      order.sort(function (x, y) { return mm[x] - mm[y]; });
      while (marks.firstChild) marks.removeChild(marks.firstChild);
      for (g = 0; g < G; g++) {
        var from = Math.round(g * n / G), to = Math.round((g + 1) * n / G), sp = 0, so = 0;
        for (j = from; j < to; j++) { sp += inv(mm[order[j]]); so += pts[order[j]].y; }
        var xm = sp / (to - from), ym = so / (to - from);
        d += (g ? 'L' : 'M') + px(xm).toFixed(1) + ' ' + py(ym).toFixed(1);
        el('circle', { cx: px(xm), cy: py(ym), r: 4.2, 'class': 'w-pt' }, marks);
      }
      path.setAttribute('d', d);
      for (j = 0; j < n; j++) { E += inv(mm[j]); O += pts[j].y; }
      btns.forEach(function (b, x) { b.className = 'w-btn' + (x === idx ? ' primary' : ''); });
      status.textContent = k.name + ': 模型一共说会有 ' + Math.round(E) + ' 位出事,实际 ' + O + ' 位,O/E = ' + (O / E).toFixed(2) + ';校准斜率 ' + slopeOf(mm).toFixed(2) + '。' + k.say;
    }
    btns.forEach(function (b, idx) { b.addEventListener('click', function () { show(idx); }); });
    show(0);
  }

  // ---------- figure 16: net benefit ----------
  // The 40 patients of "threshold" (12 would benefit from a new plan, 28 would not). One slider: the threshold
  // probability, which is also the price of a wasted plan (pt / (1 - pt) useful plans). The plot is the decision
  // curve of three strategies with the current threshold marked; the bars are their net benefit there.
  function netBenefitFig(root) {
    var pts = probScene(), n = pts.length, nev = pts.filter(function (q) { return q.ev; }).length;
    function model(t) {
      var tp = pts.filter(function (q) { return q.ev && q.p >= t; }).length, fp = pts.filter(function (q) { return !q.ev && q.p >= t; }).length;
      return { tp: tp, fp: fp, nb: tp / n - fp / n * t / (1 - t) };
    }
    function all(t) { return nev / n - (n - nev) / n * t / (1 - t); }
    var w = 360, hgt = 250, m = { l: 40, r: 10, t: 10, b: 38 }, X0 = 0, X1 = 0.65, Y0 = -0.1, Y1 = 0.34;
    function px(v) { return m.l + (v - X0) / (X1 - X0) * (w - m.l - m.r); }
    function py(v) { return m.t + (Y1 - v) / (Y1 - Y0) * (hgt - m.t - m.b); }
    var svg = el('svg', { viewBox: '0 0 ' + w + ' ' + hgt, role: 'img', 'aria-label': '决策曲线: 三种做法在各个阈值概率下的净获益' }, root);
    var id = 'w-clip-' + (++uid), cp = el('clipPath', { id: id }, el('defs', {}, svg));
    el('rect', { x: m.l, y: m.t, width: w - m.l - m.r, height: hgt - m.t - m.b }, cp);
    [-0.1, 0, 0.1, 0.2, 0.3].forEach(function (v) {
      el('line', { x1: m.l, x2: w - m.r, y1: py(v), y2: py(v), 'class': 'w-grid' }, svg);
      el('text', { x: m.l - 5, y: py(v) + 4, 'text-anchor': 'end' }, svg, minus(v.toFixed(1)));
    });
    [0, 0.2, 0.4, 0.6].forEach(function (v) {
      el('line', { x1: px(v), x2: px(v), y1: hgt - m.b, y2: hgt - m.b + 4, 'class': 'w-axis' }, svg);
      el('text', { x: px(v), y: hgt - m.b + 16, 'text-anchor': 'middle' }, svg, v.toFixed(1));
    });
    el('line', { x1: m.l, x2: w - m.r, y1: hgt - m.b, y2: hgt - m.b, 'class': 'w-axis' }, svg);
    el('line', { x1: m.l, x2: m.l, y1: m.t, y2: hgt - m.b, 'class': 'w-axis' }, svg);
    el('text', { x: (m.l + w - m.r) / 2, y: hgt - 4, 'text-anchor': 'middle' }, svg, '阈值概率');
    el('text', { x: m.l + 5, y: m.t + 10 }, svg, '净获益');
    var plot = el('g', { 'clip-path': 'url(#' + id + ')' }, svg), dA = '', dM = '', k;
    for (k = 1; k <= 64; k++) { var t = k / 100; dA += (k > 1 ? 'L' : 'M') + px(t).toFixed(1) + ' ' + py(all(t)).toFixed(1); dM += (k > 1 ? 'L' : 'M') + px(t).toFixed(1) + ' ' + py(model(t).nb).toFixed(1); }
    el('line', { x1: px(X0), x2: px(X1), y1: py(0), y2: py(0), 'class': 'w-zero' }, plot);
    el('path', { d: dA, 'class': 'w-curve thin' }, plot);
    el('path', { d: dM, 'class': 'w-curve' }, plot);
    var mark = el('line', { y1: m.t, y2: hgt - m.b, 'class': 'w-mean' }, plot), dotM = el('circle', { r: 5, 'class': 'w-pt' }, plot), dotA = el('circle', { r: 4, 'class': 'w-pt no' }, plot);
    legend(root, [['line', '按模型'], ['thin', '全都做'], ['zero', '都不做(净获益是 0)'], ['dash', '现在的阈值概率']]);
    var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
    input.type = 'range'; input.min = 0.05; input.max = 0.6; input.step = 0.05; input.value = 0.2;
    input.setAttribute('aria-label', '阈值概率');
    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', bars, '净获益(每位病人)');
    function bar(label, cls) {
      var row = h('div', 'w-bar', bars);
      h('span', 'w-bar-label', row, label);
      var fill = h('span', 'w-bar-fill ' + cls, h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
      return function (v) { fill.style.width = Math.max(0, Math.min(100, v / 0.32 * 100)).toFixed(1) + '%'; val.textContent = minus(v.toFixed(3)); };
    }
    var setM = bar('按模型', 'train'), setA = bar('全都做', 'new'), setN = bar('都不做', 'new');
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');
    function update() {
      var t = Math.round(+input.value * 100) / 100, a = model(t), price = t / (1 - t), nbAll = all(t);
      mark.setAttribute('x1', px(t)); mark.setAttribute('x2', px(t));
      dotM.setAttribute('cx', px(t)); dotM.setAttribute('cy', py(a.nb));
      dotA.setAttribute('cx', px(t)); dotA.setAttribute('cy', py(Math.max(Y0, nbAll)));
      cap.textContent = '阈值概率 ' + t.toFixed(2) + ': 白做 1 次计划,抵掉 ' + price.toFixed(2) + ' 次有用的';
      setM(a.nb); setA(nbAll); setN(0);
      status.textContent = '按模型: 做 ' + (a.tp + a.fp) + ' 次计划,' + a.tp + ' 次有用,' + a.fp + ' 次白做。全都做: 40 次,' + nev + ' 次有用,' + (n - nev) + ' 次白做。'
        + (a.nb > nbAll + 1e-9 && a.nb > 1e-9 ? '按模型最好。' : nbAll > a.nb + 1e-9 && nbAll > 0 ? '全都做最好。' : Math.abs(a.nb - nbAll) < 1e-9 && a.nb > 0 ? '按模型和全都做一样。' : '都不做最好。');
    }
    input.addEventListener('input', update);
    update();
  }

  // ---------- figure 17: sample size for a yes/no outcome ----------
  // The three criteria of Riley et al. (as in the R package pmsampsize): little overfitting (expected shrinkage
  // 0.9), a small gap between apparent and adjusted R², and a precise overall risk (within 0.05). Three sliders:
  // candidate parameters, proportion of events, expected Cox-Snell R². No random numbers.
  function sampleSizeFig(root) {
    function need(p, phi, r2) {
      var mx = 1 - Math.exp(2 * (phi * Math.log(phi) + (1 - phi) * Math.log(1 - phi))), s2 = r2 / (r2 + 0.05 * mx);
      return [Math.ceil(p / ((0.9 - 1) * Math.log(1 - r2 / 0.9))), Math.ceil(p / ((s2 - 1) * Math.log(1 - r2 / s2))), Math.ceil(Math.pow(1.96 / 0.05, 2) * phi * (1 - phi))];
    }
    function slider(label, lo, hi, step, value, fmt) {
      var ctl = h('label', 'w-ctl', root), cap = h('span', '', ctl), input = h('input', '', ctl);
      input.type = 'range'; input.min = lo; input.max = hi; input.step = step; input.value = value;
      input.setAttribute('aria-label', label);
      return { input: input, show: function () { cap.textContent = label + ': ' + fmt(+input.value); } };
    }
    var sp = slider('候选参数的个数', 1, 30, 1, 8, function (v) { return String(v); });
    var sf = slider('事件的比例', 0.05, 0.5, 0.01, 0.18, function (v) { return Math.round(v * 100) + '%'; });
    var sr = slider('预期的 R²(Cox–Snell)', 0.01, 0.3, 0.01, 0.05, function (v) { return v.toFixed(2); });
    var bars = h('div', 'w-bars', root);
    bars.setAttribute('aria-live', 'polite');
    h('p', 'w-bars-title', bars, '每条准则要求的病人数');
    function bar(label) {
      var row = h('div', 'w-bar', bars);
      h('span', 'w-bar-label', row, label);
      var fill = h('span', 'w-bar-fill train', h('span', 'w-bar-track', row)), val = h('span', 'w-bar-val', row);
      return function (v, top, lead) { fill.style.width = Math.min(100, v / top * 100).toFixed(1) + '%'; fill.className = 'w-bar-fill ' + (lead ? 'train' : 'new'); val.textContent = String(v); };
    }
    var set = [bar('1 过拟合要小'), bar('2 成绩不虚高'), bar('3 总体风险准')];
    var status = h('p', 'w-status', root);
    status.setAttribute('aria-live', 'polite');
    function update() {
      var p = +sp.input.value, phi = +sf.input.value, r2 = +sr.input.value, n = need(p, phi, r2), top = Math.max.apply(null, n), lead = n.indexOf(top);
      sp.show(); sf.show(); sr.show();
      n.forEach(function (v, k) { set[k](v, top, k === lead); });
      status.textContent = '至少需要 ' + top + ' 位病人,其中约 ' + Math.round(top * phi) + ' 个事件(每个参数 ' + (top * phi / p).toFixed(1) + ' 个事件)。要求最高的是准则 ' + (lead + 1) + '。';
    }
    [sp, sf, sr].forEach(function (sl) { sl.input.addEventListener('input', update); });
    update();
  }

  // ---------- stepped diagrams ----------
  // One drawing that changes step by step, with the text of each step under it: the reader taps
  // "下一步", the drawing plays that step, the numbers on top fill in. Used where a lesson walks
  // through how something works, instead of exploring one dial.
  // Markup:  <div class="widget stage" data-widget="models">
  //            <div class="stage-step" data-label="线性回归"> Markdown of this step </div> …
  //          </div>
  // Without JavaScript the steps read as plain paragraphs, one after another. With the attribute
  // data-still (or the system setting "reduce motion") every step jumps to its end state.
  function easeInOut(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function num(v, d) { return minus(v.toFixed(d == null ? 2 : d)); }

  function stage(root, scene) {
    var texts = [].slice.call(root.querySelectorAll('.stage-step')), n = texts.length, first = texts[0];
    var cur = 0, timers = [], frames = [], seen = false;
    var fig = h('div', 'stage-fig'); root.insertBefore(fig, first);
    var tileBox = h('div', 'stage-tiles', fig);
    var tiles = scene.tiles.map(function (label) {
      var t = h('div', 'stage-tile', tileBox);
      return { label: label, k: h('span', 'stage-tile-k', t, label), v: h('span', 'stage-tile-v', t, '—') };
    });
    var svg = el('svg', { viewBox: '0 0 360 ' + scene.height, role: 'img', 'aria-label': scene.label }, fig);
    var prog = h('div', 'stage-progress'); root.insertBefore(prog, first);
    var segs = texts.map(function () { return h('i', '', prog); });
    var nav = h('div', 'stage-nav'); root.insertBefore(nav, first);
    var back = button(nav, '← 上一步', 'stage-back'), again = button(nav, '↻', 'stage-again'), next = button(nav, '', 'primary stage-next');
    again.setAttribute('aria-label', '重播这一步'); again.title = '重播这一步';
    var status = h('p', 'w-status stage-status'); root.insertBefore(status, first);
    status.setAttribute('aria-live', 'polite');
    var body = h('div', 'stage-text'); root.insertBefore(body, first);
    texts.forEach(function (t) { body.appendChild(t); });

    // What a scene gets: the drawing, the tiles, and a clock. "at" and "tween" are cancelled when
    // the reader moves on; a tween calls fn with 0 … 1 and never sets a starting state by itself.
    var api = {
      svg: svg, instant: false,
      tile: function (i, value, label) { tiles[i].v.textContent = value; tiles[i].k.textContent = label || tiles[i].label; },
      at: function (ms, fn) { if (api.instant) fn(); else timers.push(setTimeout(fn, ms)); },
      tween: function (ms, dur, fn) {
        if (api.instant) { fn(1); return; }
        timers.push(setTimeout(function () {
          var t0 = null;
          function frame(now) {
            if (t0 === null) t0 = now;
            var t = Math.min(1, (now - t0) / dur);
            fn(t);
            if (t < 1) frames.push(requestAnimationFrame(frame));
          }
          frames.push(requestAnimationFrame(frame));
        }, ms));
      }
    };
    var enter = scene.build(api);
    function still() {
      return root.hasAttribute('data-still') || !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    }
    function go(i, quiet) {
      timers.forEach(clearTimeout); frames.forEach(cancelAnimationFrame); timers = []; frames = [];
      cur = Math.max(0, Math.min(n - 1, i));
      api.instant = !!quiet || still();
      tiles.forEach(function (t, k) { api.tile(k, '—'); });
      enter(cur);
      texts.forEach(function (t, k) { t.classList.toggle('on', k === cur); });
      segs.forEach(function (s, k) { s.className = k < cur ? 'done' : k === cur ? 'now' : ''; });
      back.disabled = cur === 0;
      next.disabled = cur === n - 1;
      next.textContent = cur === n - 1 ? '已是最后一步' : '下一步: ' + texts[cur + 1].getAttribute('data-label') + ' →';
      status.textContent = '第 ' + (cur + 1) + ' 步 / 共 ' + n + ' 步 · ' + texts[cur].getAttribute('data-label');
    }
    // the text box keeps the height of the longest step, so the page below does not jump
    function fit() {
      var max = 0;
      texts.forEach(function (t) { t.classList.add('on'); max = Math.max(max, t.offsetHeight); });
      texts.forEach(function (t, k) { t.classList.toggle('on', k === cur); });
      body.style.minHeight = max + 'px';
    }
    back.addEventListener('click', function () { go(cur - 1); });
    next.addEventListener('click', function () { go(cur + 1); });
    again.addEventListener('click', function () { go(cur); });
    root.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight' && cur < n - 1) { go(cur + 1); e.preventDefault(); }
      if (e.key === 'ArrowLeft' && cur > 0) { go(cur - 1); e.preventDefault(); }
    });
    go(0, true);
    fit();
    window.addEventListener('resize', fit);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);
    // the first step plays once, when the drawing first comes into view
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (entries) {
        if (!seen && entries[0].isIntersecting) { seen = true; io.disconnect(); if (cur === 0) go(0); }
      }, { threshold: 0.6 });
      io.observe(fig);
    }
  }
  function staged(scene) {
    var make = function (root) { stage(root, scene); };
    make.keep = true;                     // the steps' text is in the page: do not empty the box
    return make;
  }

  // --- scene: one new patient through six models (stat-ml lesson 15) ---
  // Every number is printed by the lesson's own code: 196 simulated patients, three predictors,
  // outcome Δ (volume plan); the patient has wall 3.0 mm, applicator 30 mm, 3 fractions.
  // FOREST: what each of the 300 trees of the random forest predicts for this patient.
  // BOOST: the prediction of the gradient boosting model after 1, 2, … 100 small trees.
  var FOREST = [-5.89,-5.45,-4.88,-3.95,-3.71,-5.19,-3.91,-5.09,-5.36,-6.96,-4.25,-4.91,-5.21,-5.08,-4.93,-4.56,-5.18,-4.6,-4.75,-6.2,-4.3,-3.71,-4.56,-4.42,-3.71,-4.96,-5.95,-5.18,-4.65,-4.62,-5.55,-5.65,-4.69,-5.37,-5.69,-5.51,-4.66,-5.26,-4.25,-4.84,-4.67,-4.38,-4.49,-6.02,-5.13,-5.29,-3.67,-4.72,-5.56,-4.04,-5.73,-4.66,-3.95,-4.63,-4.97,-4.81,-4.63,-3.64,-5.31,-4.68,-4.8,-4.27,-4.97,-4.9,-5.65,-5.64,-5.78,-4.02,-4.9,-2.85,-5.37,-5.5,-4.42,-4.15,-5.34,-5.07,-4.08,-4.63,-5.65,-4.65,-4.71,-5.23,-4.11,-4.69,-5.15,-4.82,-4.68,-5.2,-4.69,-4.16,-4.14,-6.12,-4.21,-4.54,-5.55,-4.48,-5.52,-4.61,-2.75,-5.23,-4.32,-4.96,-5.34,-6.98,-3.35,-7.23,-4.83,-4.39,-4.54,-4.65,-5.03,-4.45,-4.9,-5.9,-4.68,-3.71,-5.41,-6.54,-4.8,-3.51,-5.64,-4.73,-5.41,-5.37,-5.63,-4.53,-3.15,-5.02,-5.92,-4.89,-4.02,-5.93,-4.08,-4.45,-6.33,-5.35,-5.31,-6.33,-3.23,-4.15,-5.48,-4.32,-5.66,-5.56,-4.2,-4.24,-4.5,-5.14,-4.59,-5.26,-4.55,-5.72,-5.42,-6.19,-5.5,-5.12,-3.78,-4.22,-3.99,-4.69,-4.08,-2.95,-4.87,-4.9,-4.64,-5.19,-4.33,-5.58,-4.91,-6.37,-5.98,-4.0,-4.64,-4.01,-5.11,-6.33,-5.63,-5.71,-4.81,-4.2,-4.46,-5.08,-5.48,-5.76,-5.39,-3.76,-2.78,-4.96,-4.78,-4.74,-4.65,-6.25,-4.44,-4.53,-4.96,-6.08,-5.71,-5.53,-4.97,-5.12,-3.94,-3.9,-4.72,-5.62,-4.09,-5.17,-5.69,-4.45,-4.65,-3.06,-4.36,-4.0,-3.08,-4.69,-4.22,-4.71,-4.55,-4.62,-5.28,-3.96,-4.27,-4.34,-4.81,-3.93,-4.21,-4.35,-4.15,-6.12,-5.87,-4.05,-4.77,-5.34,-4.42,-4.91,-5.18,-4.38,-3.06,-4.61,-5.27,-4.75,-3.73,-3.62,-6.16,-6.15,-5.02,-5.05,-5.16,-5.76,-5.44,-4.68,-4.55,-3.67,-4.31,-5.65,-4.91,-4.0,-6.08,-3.7,-5.8,-4.79,-4.37,-4.97,-5.17,-3.99,-3.86,-3.95,-6.26,-4.85,-5.06,-5.44,-4.56,-5.66,-5.36,-5.52,-5.16,-5.83,-4.65,-4.19,-4.03,-3.27,-4.9,-4.44,-4.82,-4.27,-5.46,-5.66,-5.37,-5.2,-3.74,-5.42,-5.06,-6.48,-5.08,-7.08,-6.14,-4.2,-5.24,-3.86,-4.81,-5.28];
  var BOOST = [-2.239,-2.384,-2.521,-2.67,-2.795,-2.93,-3.043,-3.166,-3.283,-3.38,-3.486,-3.587,-3.671,-3.762,-3.849,-3.921,-4.008,-4.067,-4.111,-4.189,-4.221,-4.279,-4.309,-4.378,-4.406,-4.465,-4.491,-4.54,-4.598,-4.621,-4.662,-4.715,-4.736,-4.786,-4.804,-4.823,-4.84,-4.821,-4.839,-4.872,-4.854,-4.876,-4.86,-4.857,-4.871,-4.868,-4.882,-4.879,-4.865,-4.892,-4.905,-4.893,-4.895,-4.907,-4.904,-4.905,-4.894,-4.916,-4.91,-4.921,-4.922,-4.922,-4.923,-4.913,-4.926,-4.92,-4.926,-4.918,-4.918,-4.938,-4.93,-4.93,-4.941,-4.946,-4.947,-4.939,-4.94,-4.945,-4.946,-4.948,-4.948,-4.941,-4.946,-4.947,-4.94,-4.937,-4.949,-4.943,-4.943,-4.947,-4.963,-4.963,-4.962,-4.972,-4.973,-4.975,-4.975,-4.976,-4.975,-4.976];
  var modelsScene = {
    label: '分步图: 同一位新病人依次经过六种模型,每种模型怎样算出预测值',
    height: 300,
    tiles: ['模型', '这位病人的预测', '新病人身上的误差'],
    build: function (api) {
      var svg = api.svg, L = {}, E = easeInOut, MEAN = -2.0867;
      function X(v) { return 30 + (v + 8) * 37.5; }                // the Δ axis: −8 … 0 Gy
      function XP(p) { return 30 + p * 300; }                      // the probability axis
      function T(parent, x, y, s, cls, anchor) { return el('text', { x: x, y: y, 'class': cls || '', 'text-anchor': anchor || 'start' }, parent, s); }
      function layer(name) { return (L[name] = el('g', { 'class': 'stage-layer' }, svg)); }
      function show(names) { for (var k in L) L[k].setAttribute('class', 'stage-layer' + (names.indexOf(k) >= 0 ? ' on' : '')); }
      function fade(node, v) { node.style.opacity = v; }
      function widthOf(s) { var w = 0; for (var i = 0; i < s.length; i++) w += s.charCodeAt(i) > 255 ? 11.4 : 6.6; return w; }

      // the machine: a panel with the model's name on its edge
      var panel = el('rect', { x: 10, y: 52, width: 340, height: 154, rx: 12, 'class': 's-panel' }, svg);

      // --- what is inside the panel, one layer per step ---
      var ask = layer('ask');
      el('rect', { x: 118, y: 100, width: 124, height: 44, rx: 10, 'class': 's-node hot' }, ask);
      T(ask, 180, 127, '模型', 's-big2', 'middle');
      T(ask, 180, 172, '一条规则: 三个数进去,一个预测出来', 's-t', 'middle');
      T(ask, 180, 246, '?', 's-q', 'middle');

      var base = layer('base');
      T(base, 180, 96, '不看这位病人的任何信息', 's-t', 'middle');
      var baseNum = T(base, 180, 140, '', 's-big', 'middle');
      T(base, 180, 170, '196 位建模病人的平均值', '', 'middle');

      // rows of "variable, how it differs, contribution": used by the straight line and by LASSO
      function rows(parent, items) {
        el('line', { x1: 268, x2: 268, y1: 78, y2: 168, 'class': 'w-zero' }, parent);
        return items.map(function (it, k) {
          var y = 92 + k * 32, g = el('g', {}, parent);
          T(g, 24, y, it[0], 's-v');
          var note = T(g, 24, y + 14, it[1], 's-k');
          var ghost = el('rect', { y: y - 10, height: 12, rx: 2, 'class': 's-bar ghost' }, g);
          var bar = el('rect', { y: y - 10, height: 12, rx: 2, 'class': 's-bar' }, g);
          var val = T(g, 338, y, '', 's-v s-n', 'end');
          function size(node, v) { var w = Math.max(0.8, Math.abs(v) * 70); node.setAttribute('x', 268 - w); node.setAttribute('width', w); }
          return { g: g, note: note, set: function (v) { size(bar, v); val.textContent = num(v); }, ghost: function (v) { size(ghost, v); fade(ghost, v == null ? 0 : 1); } };
        });
      }
      var lin = layer('lin');
      T(lin, 338, 72, '和平均的病人比,加或减多少 (Gy)', 's-k', 'end');
      var LIN = [['壁厚', '比平均薄 0.69 mm,每 mm 1.57 Gy', -1.0891], ['直径', '比平均小 0.05 mm,每 mm 0.17 Gy', -0.0085], ['近距离次数', '比平均多 0.55 次,每次 −1.21 Gy', -0.6581]];
      var linRows = rows(lin, LIN), linSum = el('g', {}, lin);
      T(linSum, 24, 196, '从平均值 −2.09 出发,加上这三项', 's-t');
      T(linSum, 338, 196, '−3.84 Gy', 's-v s-n', 'end');

      var lasso = layer('lasso');
      T(lasso, 338, 72, '约束强度 alpha = 0.2;虚线框是没加约束时', 's-k', 'end');
      var LAS = [['壁厚', '系数 1.57 → 1.40', -1.0891, -0.9734], ['直径', '系数 0.17 → 0.11', -0.0085, -0.0055], ['近距离次数', '系数 −1.21 → −0.80', -0.6581, -0.4348]];
      var lasRows = rows(lasso, LAS), lasSum = el('g', {}, lasso);
      T(lasSum, 24, 196, '三项都往 0 收了一点', 's-t');
      T(lasSum, 338, 196, '−3.50 Gy', 's-v s-n', 'end');

      var tree = layer('tree');
      function edge(a, b, word) {
        var ln = el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], 'class': 's-edge' }, tree);
        T(tree, (a[0] + b[0]) / 2 + (b[0] < a[0] ? -9 : 9), (a[1] + b[1]) / 2 + 1, word, 's-k', 'middle');
        return ln;
      }
      function node(cx, cy, w, s, leaf) {
        var r = el('rect', { x: cx - w / 2, y: cy - 12, width: w, height: 24, rx: leaf ? 12 : 6, 'class': 's-node' }, tree);
        T(tree, cx, cy + 4, s, leaf ? 's-v s-n' : 's-t', 'middle');
        return r;
      }
      var eRootL = edge([150, 102], [98, 122], '是'), eRootR = edge([210, 102], [262, 122], '否');
      var eLL = edge([80, 146], [52, 170], '是'), eLR = edge([116, 146], [136, 170], '否');
      edge([244, 146], [224, 170], '是'); edge([280, 146], [308, 170], '否');
      var nRoot = node(180, 90, 132, '壁厚 ≤ 3.35 mm?'), nLeft = node(98, 134, 108, '近距离做 2 次?');
      node(262, 134, 122, '壁厚 ≤ 5.10 mm?');
      node(52, 182, 60, '−2.69', true);
      var nHit = node(136, 182, 60, '−5.13', true);
      node(224, 182, 60, '−1.34', true); node(308, 182, 60, '+1.40', true);
      var treeParts = [nRoot, nLeft, nHit, eRootL, eLR];

      var forest = layer('forest'), minis = [];
      FOREST.forEach(function (v, i) {
        var x = 27 + (i % 30) * 10.3, y = 76 + Math.floor(i / 30) * 10.6;
        minis.push(el('path', { d: 'M' + (x + 3.5) + ' ' + y + 'l-3.5 6.5m3.5 -6.5l3.5 6.5', 'class': 's-mini' }, forest));
      });
      var forestNote = T(forest, 24, 198, '', 's-t');

      var boost = layer('boost');
      T(boost, 24, 78, '新预测 = 旧预测 + 0.05 × 这棵小树的修正', 's-t');
      var boostCount = T(boost, 338, 78, '', 's-v s-n', 'end');
      function BX(k) { return 46 + k * 2.86; }
      function BY(v) { return 98 + (-2 - v) * 24; }
      [[MEAN, '−2.09'], [-4.976, '−4.98']].forEach(function (g) {
        el('line', { x1: 46, x2: 332, y1: BY(g[0]), y2: BY(g[0]), 'class': 'w-zero' }, boost);
        T(boost, 42, BY(g[0]) + 3.5, g[1], 's-n', 'end');
      });
      T(boost, 46, 196, '第 1 棵', '', 'start'); T(boost, 332, 196, '第 100 棵', '', 'end');
      var boostLine = el('polyline', { 'class': 's-line', points: '' }, boost), boostHead = el('circle', { r: 4, 'class': 's-token' }, boost);

      var logit = layer('logit');
      var LOG = [['7.24', '起点'], ['− 0.83 × 2.0', '超出'], ['− 1.59 × 3.0', '壁厚']].map(function (it, k) {
        var g = el('g', {}, logit);
        T(g, 24, 94 + k * 21, it[0], 's-t s-n'); T(g, 142, 94 + k * 21, it[1], 's-k', 'end');
        return g;
      });
      var logSum = el('g', {}, logit);
      el('line', { x1: 24, x2: 142, y1: 144, y2: 144, 'class': 'w-axis' }, logSum);
      T(logSum, 24, 163, '= 0.81', 's-v s-n'); T(logSum, 24, 179, '加起来的数', 's-k');
      function SX(v) { return 176 + (v + 6) * 13; }
      function SY(p) { return 180 - p * 92; }
      function sig(v) { return 1 / (1 + Math.exp(-v)); }
      el('line', { x1: 176, x2: 332, y1: 180, y2: 180, 'class': 'w-axis' }, logit);
      el('line', { x1: 176, x2: 176, y1: 88, y2: 180, 'class': 'w-axis' }, logit);
      T(logit, 172, 183, '0', 's-n', 'end'); T(logit, 172, 92, '1', 's-n', 'end');
      T(logit, 176, 193, '−6', 's-n', 'middle'); T(logit, SX(0), 193, '0', 's-n', 'middle'); T(logit, 332, 193, '6', 's-n', 'middle');
      T(logit, 328, 84, '概率', 's-k', 'end');
      var sPts = [];
      for (var sv = -6; sv <= 6.001; sv += 0.25) sPts.push(SX(sv).toFixed(1) + ',' + SY(sig(sv)).toFixed(1));
      el('polyline', { 'class': 's-line', points: sPts.join(' ') }, logit);
      var logGuide = el('g', {}, logit);
      el('line', { x1: SX(0.807), x2: SX(0.807), y1: 180, y2: SY(0.691), 'class': 'w-mean' }, logGuide);
      el('line', { x1: 176, x2: SX(0.807), y1: SY(0.691), y2: SY(0.691), 'class': 'w-mean' }, logGuide);
      T(logGuide, 181, SY(0.691) - 5, '0.69', 's-v s-n');
      var logDot = el('circle', { r: 4.5, 'class': 's-token' }, logit);

      var six = layer('six');
      var SIX = [['都猜平均值', -2.09], ['线性回归', -3.84], ['LASSO', -3.50], ['一棵树', -5.13], ['随机森林', -4.86], ['梯度提升', -4.98]];
      var sixRows = SIX.map(function (it, k) {
        var y = 86 + k * 19.5, g = el('g', {}, six);
        el('line', { x1: 92, x2: 292, y1: y - 4, y2: y - 4, 'class': 'w-grid' }, g);
        T(g, 24, y, it[0], 's-t');
        el('circle', { cx: X(it[1]), cy: y - 4, r: 4.5, 'class': 's-dot' }, g);
        T(g, 338, y, num(it[1]), 's-v s-n', 'end');
        return g;
      });
      var truth = el('line', { x1: X(-3.84), x2: X(-3.84), y1: 70, y2: 70, 'class': 's-truth' }, six);
      var truthNote = T(six, X(-3.84) + 6, 201, '生成数据的公式: 平均 −3.84', 's-k');

      var bars = layer('bars');
      var BARS = [['线性回归', 1.73], ['LASSO', 1.74], ['梯度提升', 1.78], ['随机森林', 1.83], ['一棵树', 1.92], ['都猜平均值', 2.28]];
      var barRows = BARS.map(function (it, k) {
        var y = 92 + k * 29, g = el('g', {}, bars);
        T(g, 24, y, it[0], 's-t');
        var r = el('rect', { x: 100, y: y - 10, height: 12, rx: 2, 'class': 's-bar' + (k === 0 ? '' : k === 5 ? ' dim' : ' mid') }, g);
        T(g, 338, y, it[1].toFixed(2), 's-v s-n', 'end');
        return { g: g, set: function (u) { r.setAttribute('width', Math.max(0.5, it[1] * 86 * u)); } };
      });
      T(bars, 180, 276, '4900 位新病人身上的平均误差 (Gy),越短越好', '', 'middle');

      // --- the patient's numbers, on top ---
      T(svg, 10, 12, '一位新病人', 's-k');
      var chips = [0, 1, 2].map(function () {
        var g = el('g', { 'class': 's-chip-g' }, svg), stub = el('line', { y1: 40, y2: 52, 'class': 's-stub' }, g);
        var r = el('rect', { y: 18, height: 22, rx: 11, 'class': 's-chip' }, g), tx = el('text', { y: 33, 'text-anchor': 'middle' }, g);
        var k = el('tspan', { 'class': 's-k' }, tx), v = el('tspan', { 'class': 's-v s-n', dx: 4 }, tx), c = { cx: 0 };
        c.set = function (x, w, label, value) {
          c.cx = x + w / 2;
          r.setAttribute('x', x); r.setAttribute('width', w); tx.setAttribute('x', c.cx);
          stub.setAttribute('x1', c.cx); stub.setAttribute('x2', c.cx);
          k.textContent = label; v.textContent = value; g.setAttribute('class', 's-chip-g');
        };
        c.off = function () { g.setAttribute('class', 's-chip-g off'); };
        c.hide = function () { g.setAttribute('class', 's-chip-g gone'); };
        return c;
      });
      var tabBox = el('rect', { x: 22, y: 43, height: 18, rx: 9, 'class': 's-tab-box' }, svg), tab = T(svg, 31, 56, '', 's-tab');
      function setTab(s) { tab.textContent = s; tabBox.setAttribute('width', widthOf(s) + 18); }

      // --- the answer, below: an axis with one pin, and a small mark for every model seen so far ---
      function makeAxis(name, ticks, where, title) {
        var g = layer(name);
        el('line', { x1: 30, x2: 330, y1: 264, y2: 264, 'class': 'w-axis' }, g);
        ticks.forEach(function (tk) {
          el('line', { x1: where(tk[0]), x2: where(tk[0]), y1: 264, y2: 268, 'class': 'w-axis' }, g);
          T(g, where(tk[0]), 280, tk[1], 's-n', 'middle');
        });
        T(g, 180, 294, title, '', 'middle');
        var extra = el('g', {}, g), trail = el('line', { y1: 264, y2: 264, 'class': 's-trail' }, g);
        var pin = el('g', { 'class': 's-pin' }, g);
        el('line', { x1: 0, x2: 0, y1: 237, y2: 264, 'class': 's-pin-stem' }, pin);
        el('circle', { cx: 0, cy: 232, r: 5, 'class': 's-pin-dot' }, pin);
        var label = T(pin, 0, 221, '', 's-pin-label s-n', 'middle');
        return {
          g: g, extra: extra,
          pin: function (v, s) { pin.setAttribute('transform', 'translate(' + where(v).toFixed(1) + ' 0)'); label.textContent = s; fade(pin, 1); },
          trail: function (a, b) { trail.setAttribute('x1', where(a)); trail.setAttribute('x2', where(b)); fade(trail, 1); },
          clear: function () { fade(pin, 0); fade(trail, 0); }
        };
      }
      var axis = makeAxis('axis', [[-8, '−8'], [-6, '−6'], [-4, '−4'], [-2, '−2'], [0, '0']], X, '这位病人的预测 Δ (Gy)');
      var paxis = makeAxis('paxis', [[0, '0%'], [0.25, '25%'], [0.5, '50%'], [0.75, '75%'], [1, '100%']], XP, '降到 68 Gy 以下的概率');
      var histBars = [], histG = el('g', {}, axis.extra);
      for (var b = 0; b < 18; b++) histBars.push(el('rect', { x: X(-7.25 + b * 0.25) + 0.5, width: 8.4, y: 264, height: 0, 'class': 's-hist' }, histG));
      var ORDER = [-2.09, -3.84, -3.50, -5.13, -4.86, -4.98];
      var marks = ORDER.map(function (v) { return el('circle', { cx: X(v), cy: 264, r: 3.4, 'class': 's-mark' }, axis.extra); });

      var tokens = [0, 1, 2, 3].map(function () { return el('circle', { r: 4, 'class': 's-token' }, svg); });
      function fly(k, a, c, ms, dur) {
        api.tween(ms, dur, function (u) {
          var e = E(u), tk = tokens[k];
          tk.setAttribute('cx', lerp(a[0], c[0], e)); tk.setAttribute('cy', lerp(a[1], c[1], e));
          fade(tk, u < 1 ? 1 : 0);
        });
      }
      function hot(node, on) { node.setAttribute('class', node.getAttribute('class').replace(/ (hot|hit)/g, '') + (on ? ' ' + on : '')); }

      // the state every step starts from; "seen" models already have their mark on the axis
      function reset(seen, name, layers) {
        chips[0].set(10, 108, '壁厚', '3.0 mm'); chips[1].set(126, 108, '直径', '30 mm'); chips[2].set(242, 108, '近距离', '3 次');
        panel.setAttribute('height', 154);
        setTab(name);
        show(layers);
        marks.forEach(function (m, i) { fade(m, i < seen ? 1 : 0); });
        axis.clear(); paxis.clear(); fade(histG, 0);
        tokens.forEach(function (tk) { fade(tk, 0); });
      }
      function land(ms, value, text, error) {          // the answer arrives on the axis
        api.at(ms, function () { axis.pin(value, text); api.tile(1, text + ' Gy'); api.tile(2, error); });
      }

      var steps = [
        function () {                                   // 1 the question
          reset(0, '模型', ['ask', 'axis']);
          api.tile(0, '?'); api.tile(1, '? Gy');
          [0, 1, 2].forEach(function (k) { fly(k, [chips[k].cx, 40], [180, 104], 200 + k * 180, 620); });
          fly(3, [180, 144], [180, 232], 1300, 620);
        },
        function () {                                   // 2 the baseline
          reset(0, '基线', ['base', 'axis']);
          chips.forEach(function (c) { c.off(); });
          api.tile(0, '基线');
          baseNum.textContent = '0.00 Gy';
          api.tween(250, 700, function (u) { baseNum.textContent = num(MEAN * E(u)) + ' Gy'; });
          fly(3, [180, 176], [X(MEAN), 232], 1000, 500);
          land(1500, MEAN, '−2.09', '2.28 Gy');
        },
        function () {                                   // 3 the straight line
          reset(1, '线性回归', ['lin', 'axis']);
          api.tile(0, '线性回归');
          var at = MEAN;
          linRows.forEach(function (r, k) { fade(r.g, 0); r.set(0); r.ghost(null); });
          fade(linSum, 0);
          axis.pin(MEAN, '−2.09'); axis.trail(MEAN, MEAN);
          linRows.forEach(function (r, k) {
            var from = at, to = at + LIN[k][2];
            at = to;
            fly(k, [chips[k].cx, 40], [chips[k].cx, 70], 150 + k * 800, 300);
            api.tween(350 + k * 800, 600, function (u) {
              var e = E(u), v = lerp(from, to, e);
              fade(r.g, Math.min(1, u * 3)); r.set(LIN[k][2] * e);
              if (u === 1) r.set(LIN[k][2]);
              axis.pin(v, num(v)); axis.trail(MEAN, v);
            });
          });
          api.tween(2750, 300, function (u) { fade(linSum, u); });
          land(2750, -3.8424, '−3.84', '1.73 Gy');
        },
        function () {                                   // 4 the same formula, coefficients pulled towards 0
          reset(2, 'LASSO', ['lasso', 'axis']);
          api.tile(0, 'LASSO');
          lasRows.forEach(function (r, k) { r.set(LAS[k][2]); r.ghost(LAS[k][2]); });
          fade(lasSum, 0);
          axis.pin(-3.8424, '−3.84');
          api.tween(600, 1000, function (u) {
            var e = E(u), v = lerp(-3.8424, -3.5004, e);
            lasRows.forEach(function (r, k) { r.set(lerp(LAS[k][2], LAS[k][3], e)); });
            axis.pin(v, num(v));
          });
          api.tween(1650, 300, function (u) { fade(lasSum, u); });
          land(1650, -3.5004, '−3.50', '1.74 Gy');
        },
        function () {                                   // 5 one tree
          reset(3, '一棵树', ['tree', 'axis']);
          api.tile(0, '一棵树');
          treeParts.forEach(function (p) { hot(p, ''); });
          fly(0, [180, 40], [180, 78], 200, 380);
          api.at(580, function () { hot(nRoot, 'hot'); });
          api.at(780, function () { hot(eRootL, 'hot'); });
          fly(0, [150, 102], [98, 122], 780, 420);
          api.at(1200, function () { hot(nLeft, 'hot'); });
          api.at(1400, function () { hot(eLR, 'hot'); });
          fly(0, [116, 146], [136, 170], 1400, 420);
          api.at(1820, function () { hot(nHit, 'hit'); });
          fly(0, [136, 194], [X(-5.13), 232], 2000, 400);
          land(2400, -5.13, '−5.13', '1.92 Gy');
        },
        function () {                                   // 6 three hundred trees, averaged
          reset(4, '随机森林', ['forest', 'axis']);
          api.tile(0, '随机森林');
          fade(histG, 1);
          var shown = -1;
          function grow(k) {                            // the first k trees have answered
            if (k === shown) return;
            var counts = [], sum = 0, i;
            for (i = 0; i < 18; i++) counts.push(0);
            for (i = 0; i < k; i++) { sum += FOREST[i]; counts[Math.min(17, Math.max(0, Math.floor((FOREST[i] + 7.25) / 0.25)))]++; }
            minis.forEach(function (m, j) { m.setAttribute('class', 's-mini' + (j < k ? ' on' : '')); });
            histBars.forEach(function (r, j) { r.setAttribute('height', counts[j] * 0.56); r.setAttribute('y', 264 - counts[j] * 0.56); });
            forestNote.textContent = k ? '已经问了 ' + k + ' 棵,平均 ' + num(sum / k) + ' Gy' : '300 棵树,每棵用重新抽的一批病人长成';
            if (k) { axis.pin(sum / k, num(sum / k)); api.tile(1, num(sum / k) + ' Gy'); }
            shown = k;
          }
          grow(0);
          api.tween(500, 3000, function (u) { grow(Math.round(300 * u)); });
          api.at(3500, function () { api.tile(2, '1.83 Gy'); });
        },
        function () {                                   // 7 a hundred small trees, one after another
          reset(5, '梯度提升', ['boost', 'axis']);
          api.tile(0, '梯度提升');
          function upto(k) {
            var pts = [BX(0) + ',' + BY(MEAN)], v = MEAN, i;
            for (i = 1; i <= k; i++) { v = BOOST[i - 1]; pts.push(BX(i).toFixed(1) + ',' + BY(v).toFixed(1)); }
            boostLine.setAttribute('points', pts.join(' '));
            boostHead.setAttribute('cx', BX(k)); boostHead.setAttribute('cy', BY(v));
            boostCount.textContent = k ? '第 ' + k + ' 棵' : '还没有树';
            axis.pin(v, num(v)); axis.trail(MEAN, v); api.tile(1, num(v) + ' Gy');
          }
          upto(0);
          api.tween(500, 3000, function (u) { upto(Math.round(100 * u)); });
          api.at(3500, function () { api.tile(2, '1.78 Gy'); });
        },
        function () {                                   // 8 a yes/no outcome: the sum goes through an S-shaped curve
          reset(6, '逻辑回归', ['logit', 'paxis']);
          chips[1].set(126, 224, '5 mm 计划超出 68 Gy', '2.0 Gy'); chips[2].hide();
          api.tile(0, '逻辑回归'); api.tile(1, '—', '这位病人的概率'); api.tile(2, '—', '新病人的 AUC');
          LOG.forEach(function (g, k) { fade(g, 0); api.tween(200 + k * 350, 250, function (u) { fade(g, u); }); });
          fade(logSum, 0); fade(logGuide, 0); fade(logDot, 0);
          api.tween(1250, 250, function (u) { fade(logSum, u); });
          api.tween(1700, 900, function (u) {
            var v = lerp(-6, 0.807, E(u));
            logDot.setAttribute('cx', SX(v)); logDot.setAttribute('cy', SY(sig(v))); fade(logDot, 1);
          });
          api.tween(2600, 300, function (u) { fade(logGuide, u); });
          api.at(2900, function () { paxis.pin(0.691, '69%'); api.tile(1, '69%', '这位病人的概率'); api.tile(2, '0.91', '新病人的 AUC'); });
        },
        function () {                                   // 9 six answers for one patient
          reset(6, '同一位病人', ['six', 'axis']);
          api.tile(0, '六个模型'); api.tile(1, '—', '最多相差'); api.tile(2, '—', '生成数据的公式');
          sixRows.forEach(function (g, k) { fade(g, 0); api.tween(150 + k * 170, 250, function (u) { fade(g, u); }); });
          truth.setAttribute('y2', 70); fade(truthNote, 0);
          api.tween(1400, 500, function (u) { truth.setAttribute('y2', lerp(70, 264, E(u))); fade(truthNote, u); });
          api.at(1400, function () { api.tile(1, '3.0 Gy', '最多相差'); api.tile(2, '−3.84 Gy', '生成数据的公式'); });
        },
        function () {                                   // 10 4900 new patients
          reset(6, '新病人', ['bars']);
          chips.forEach(function (c) { c.off(); });
          api.tile(0, '线性回归', '误差最小的'); api.tile(1, '4900 位', '新病人'); api.tile(2, '—');
          barRows.forEach(function (r, k) { r.set(0); api.tween(350 + k * 130, 500, function (u) { r.set(E(u)); }); });
          api.tween(0, 350, function (u) { panel.setAttribute('height', lerp(154, 238, E(u))); });
          api.at(1500, function () { api.tile(2, '1.73 Gy'); });
        }
      ];
      return function (i) { steps[i](); };
    }
  };

  // --- scene: how a regression tree is grown, cut by cut (the algorithm of stat-ml lesson 7) ---
  // The algorithm really runs here, on the 30 simulated patients all figures share: every place a
  // group can be cut is tried, the cut with the smallest squared error is kept, and the same is
  // done again inside each group. The tiles show what it computes.
  var treeScene = {
    label: '分步图: 一棵回归树怎样一刀一刀长出来',
    height: 338,
    tiles: ['分成几组', '建模病人的误差', '这一步'],
    build: function (api) {
      var svg = api.svg, sc = getScene(), n = sc.train.length, L = {}, E = easeInOut, i;
      var P = sc.train.slice().sort(function (a, b) { return a.x - b.x; });
      function T(parent, x, y, s, cls, anchor) { return el('text', { x: x, y: y, 'class': cls || '', 'text-anchor': anchor || 'start' }, parent, s); }
      function layer(name) { return (L[name] = el('g', { 'class': 'stage-layer' }, svg)); }
      function show(names) { for (var k in L) L[k].setAttribute('class', 'stage-layer' + (names.indexOf(k) >= 0 ? ' on' : '')); }
      function fade(node, v) { node.style.opacity = v; }

      // --- the algorithm ---
      function stat(lo, hi) {                       // mean and squared error of the patients lo … hi−1
        var m = 0, s = 0, k;
        for (k = lo; k < hi; k++) m += P[k].y / (hi - lo);
        for (k = lo; k < hi; k++) s += (P[k].y - m) * (P[k].y - m);
        return { mean: m, sse: s };
      }
      function cuts(lo, hi) {                       // every place this group can be cut: between two neighbours
        var out = [], k;
        for (k = lo + 1; k < hi; k++) {
          if (P[k].x > P[k - 1].x) out.push({ i: k, thr: (P[k - 1].x + P[k].x) / 2, sse: stat(lo, k).sse + stat(k, hi).sse });
        }
        return out;
      }
      function bestOf(list) { return list.reduce(function (a, c) { return c.sse < a.sse ? c : a; }); }
      function groups(idx) {                        // the groups a list of cuts makes, each with its mean
        var b = [0].concat(idx.slice().sort(function (a, c) { return a - c; }), [n]), g = [], k;
        for (k = 0; k + 1 < b.length; k++) {
          var st = stat(b[k], b[k + 1]);
          g.push({ lo: b[k], hi: b[k + 1], mean: st.mean, sse: st.sse,
            x0: b[k] ? (P[b[k] - 1].x + P[b[k]].x) / 2 : XD[0], x1: b[k + 1] < n ? (P[b[k + 1] - 1].x + P[b[k + 1]].x) / 2 : XD[1] });
        }
        return g;
      }
      function mse(g) { return g.reduce(function (s, q) { return s + q.sse; }, 0) / n; }
      function predict(g, x) { for (var k = 0; k < g.length - 1; k++) if (x <= g[k].x1) return g[k].mean; return g[g.length - 1].mean; }
      function mseNew(g) { var s = 0; sc.fresh.forEach(function (p) { var d = p.y - predict(g, p.x); s += d * d; }); return s / sc.fresh.length; }

      var all = cuts(0, n), first = bestOf(all);
      var leftC = cuts(0, first.i), rightC = cuts(first.i, n), second = bestOf(leftC), third = bestOf(rightC);
      var G1 = groups([]), G2 = groups([first.i]), G4 = groups([first.i, second.i, third.i]), fullIdx = [];
      for (i = 1; i < n; i++) if (P[i].x > P[i - 1].x) fullIdx.push(i);
      var GF = groups(fullIdx), NEWX = 3.0;

      // --- the plot ---
      var plot = axes(svg);
      var zone = el('rect', { y: M.t, height: H - M.t - M.b, 'class': 's-zone' }, plot);
      var fresh = el('g', {}, plot);
      sc.shown.forEach(function (p) { el('circle', { cx: sx(p.x), cy: sy(p.y), r: 2.6, 'class': 'w-pt new' }, fresh); });
      var sticks = P.map(function (p) { return el('line', { x1: sx(p.x), x2: sx(p.x), y1: sy(p.y), y2: sy(p.y), 'class': 'w-res s-res' }, plot); });
      var fixed = el('g', {}, plot), scan = el('line', { y1: M.t, y2: H - M.b, 'class': 's-scan' }, plot);
      var stair = el('path', { 'class': 'w-curve' }, plot);
      P.forEach(function (p) { el('circle', { cx: sx(p.x), cy: sy(p.y), r: 3.4, 'class': 'w-pt' }, plot); });
      var probe = el('g', {}, plot), probeLine = el('line', { x1: sx(NEWX), x2: sx(NEWX), y1: sy(YD[0]), 'class': 's-scan' }, probe);
      var probeDot = el('circle', { cx: sx(NEWX), r: 5, 'class': 's-token' }, probe);

      function model(g, stickAmount) {              // draw a tree's prediction: one level per group
        stair.setAttribute('d', g.map(function (q) { return 'M' + sx(q.x0).toFixed(1) + ' ' + sy(q.mean).toFixed(1) + 'H' + sx(q.x1).toFixed(1); }).join(''));
        g.forEach(function (q) {
          for (var k = q.lo; k < q.hi; k++) sticks[k].setAttribute('y2', lerp(sy(P[k].y), sy(q.mean), stickAmount == null ? 1 : stickAmount));
        });
      }
      function fixedCuts(list) {
        while (fixed.firstChild) fixed.removeChild(fixed.firstChild);
        list.forEach(function (c) { el('line', { x1: sx(c.thr), x2: sx(c.thr), y1: M.t, y2: H - M.b, 'class': 'w-mean' }, fixed); });
      }
      function scanAt(thr) { scan.setAttribute('x1', sx(thr)); scan.setAttribute('x2', sx(thr)); fade(scan, 1); }
      function zoneOn(x0, x1) { zone.setAttribute('x', sx(x0)); zone.setAttribute('width', sx(x1) - sx(x0)); fade(zone, ''); }

      // --- under the plot: the error at every place tried (same x axis as the plot) ---
      var curve = layer('curve');
      T(curve, M.l + 4, 264, '切在每个位置时的误差(越低越好)', 's-k');
      el('line', { x1: M.l, x2: W - M.r, y1: 332, y2: 332, 'class': 'w-grid' }, curve);
      var lines = [el('polyline', { 'class': 's-curve2', points: '' }, curve), el('polyline', { 'class': 's-curve2', points: '' }, curve)];
      var marks = [0, 1].map(function () {
        var g = el('g', {}, curve);
        el('circle', { r: 4, 'class': 's-dot' }, g);
        return { g: g, label: T(g, 0, -9, '', 's-v s-n s-halo', 'middle') };
      });
      var now = el('circle', { r: 3.4, 'class': 's-token' }, curve), range = [0, 1];
      function CY(v) { return 328 - (v - range[0]) / (range[1] - range[0]) * 50; }
      function trace(k, list, other, upto) {         // the first "upto" places of one search
        lines[k].setAttribute('points', list.slice(0, upto).map(function (c) { return sx(c.thr).toFixed(1) + ',' + CY((c.sse + other) / n).toFixed(1); }).join(' '));
      }
      function mark(k, c, other) {
        if (!c) { fade(marks[k].g, 0); return; }
        marks[k].g.setAttribute('transform', 'translate(' + sx(c.thr).toFixed(1) + ' ' + CY((c.sse + other) / n).toFixed(1) + ')');
        marks[k].label.textContent = ((c.sse + other) / n).toFixed(2); fade(marks[k].g, 1);
      }

      // --- under the plot: the finished tree as questions ---
      var tree = layer('tree'), parts = {};
      function edge(a, b) { return el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], 'class': 's-edge' }, tree); }
      function node(cx, cy, w, s, leaf) {
        var r = el('rect', { x: cx - w / 2, y: cy - 9, width: w, height: 18, rx: leaf ? 9 : 5, 'class': 's-node' }, tree);
        T(tree, cx, cy + 3.6, s, (leaf ? 's-v s-n' : 's-t') + ' s-small', 'middle');
        return r;
      }
      var LX = [50, 138, 222, 310];
      parts.e0 = edge([150, 274], [96, 283]); parts.e1 = edge([210, 274], [264, 283]);
      parts.e00 = edge([78, 301], [LX[0], 311]); parts.e01 = edge([114, 301], [LX[1], 311]);
      parts.e10 = edge([246, 301], [LX[2], 311]); parts.e11 = edge([282, 301], [LX[3], 311]);
      parts.root = node(180, 265, 116, '壁厚 ≤ ' + first.thr.toFixed(2) + ' mm?');
      parts.n0 = node(96, 292, 110, '壁厚 ≤ ' + second.thr.toFixed(2) + ' mm?');
      parts.n1 = node(264, 292, 110, '壁厚 ≤ ' + third.thr.toFixed(2) + ' mm?');
      G4.forEach(function (q, k) { parts['l' + k] = node(LX[k], 320, 58, num(q.mean), true); });
      T(tree, W - M.r, 262, '左: 是,右: 否', 's-k', 'end');
      function hot(node, on) { node.setAttribute('class', node.getAttribute('class').replace(/ (hot|hit)/g, '') + (on ? ' ' + on : '')); }
      var goLeft = NEWX <= first.thr, goLeft2 = NEWX <= (goLeft ? second.thr : third.thr), leafK = (goLeft ? 0 : 2) + (goLeft2 ? 0 : 1);
      var path = [parts.root, goLeft ? parts.e0 : parts.e1, goLeft ? parts.n0 : parts.n1, parts['e' + (goLeft ? 0 : 1) + (goLeft2 ? 0 : 1)], parts['l' + leafK]];

      // --- under the plot: a small tree against a tree grown to the end ---
      var bars = layer('bars'), newSmall = mseNew(G4), newFull = mseNew(GF), top = Math.max(newSmall, newFull, mse(G4));
      [[G4.length + ' 组', mse(G4), newSmall, 262], [GF.length + ' 组', mse(GF), newFull, 302]].forEach(function (row) {
        T(bars, M.l, row[3] + 13, '分成 ' + row[0], 's-t');
        [[row[1], 's-bar', '建模病人 '], [row[2], 's-bar dim', '新病人 ']].forEach(function (b, k) {
          var w = Math.max(1, b[0] / top * 150), y = row[3] + k * 12;
          el('rect', { x: 104, y: y, width: w, height: 8, rx: 2, 'class': b[1] }, bars);
          T(bars, 104 + w + 5, y + 8, b[2] + b[0].toFixed(2), 's-k s-n');
        });
      });

      function reset(layers) {
        show(layers);
        fade(zone, 0); fade(scan, 0); fade(fresh, 0); fade(probe, 0); fade(now, 0);
        sticks.forEach(function (s) { fade(s, 1); });
        fixedCuts([]);
        lines.forEach(function (l) { l.setAttribute('points', ''); });
        mark(0, null); mark(1, null);
        path.forEach(function (p) { hot(p, ''); });
      }
      function search(k, list, other, lo, hi, base, ms, dur) {   // try every place of one group, one after another
        api.tween(ms, dur, function (u) {
          var upto = Math.min(list.length, Math.floor(u * list.length) + 1), c = list[upto - 1];
          scanAt(c.thr); trace(k, list, other, upto);
          model(groups(base.concat([c.i])));
          now.setAttribute('cx', sx(c.thr)); now.setAttribute('cy', CY((c.sse + other) / n)); fade(now, 1);
          api.tile(1, ((c.sse + other) / n).toFixed(2)); api.tile(2, upto + ' / ' + list.length, '试过的位置');
        });
      }

      var steps = [
        function () {                                 // 1 the patients
          reset([]);
          stair.setAttribute('d', ''); model(G1, 0); stair.setAttribute('d', '');
          api.tile(0, '—'); api.tile(1, '—'); api.tile(2, n + ' 位', '建模病人');
        },
        function () {                                 // 2 no cut: everyone gets the mean
          reset([]);
          model(G1, 0);
          api.tile(0, '1 组'); api.tile(2, num(G1[0].mean) + ' Gy', '都猜平均值');
          api.tween(300, 900, function (u) { model(G1, E(u)); });
          api.at(1200, function () { api.tile(1, mse(G1).toFixed(2)); });
        },
        function () {                                 // 3 try every place
          reset(['curve']);
          range = [first.sse / n - 0.25, mse(G1) + 0.1];
          api.tile(0, '2 组');
          model(G1);
          search(0, all, 0, 0, n, [], 400, 3600);
          api.at(4000, function () { mark(0, first, 0); });
        },
        function () {                                 // 4 keep the best one
          reset(['curve']);
          range = [first.sse / n - 0.25, mse(G1) + 0.1];
          trace(0, all, 0, all.length); mark(0, first, 0);
          var last = all[all.length - 1];
          scanAt(last.thr); model(groups([last.i]));
          api.tile(0, '2 组'); api.tile(1, ((last.sse) / n).toFixed(2)); api.tile(2, first.thr.toFixed(2) + ' mm', '误差最小的位置');
          api.tween(300, 700, function (u) { scanAt(lerp(last.thr, first.thr, E(u))); });
          api.at(1000, function () { fade(scan, 0); fixedCuts([first]); model(G2); api.tile(1, mse(G2).toFixed(2)); });
        },
        function () {                                 // 5 the same again inside each group
          reset(['curve']);
          var lo = Math.min(second.sse + G2[1].sse, third.sse + G2[0].sse) / n;
          range = [lo - 0.2, mse(G2) + 0.1];
          fixedCuts([first]); model(G2);
          api.tile(0, '4 组'); api.tile(1, mse(G2).toFixed(2));
          zoneOn(XD[0], first.thr);
          search(0, leftC, G2[1].sse, 0, first.i, [first.i], 400, 2200);
          api.at(2650, function () { mark(0, second, G2[1].sse); fixedCuts([first, second]); model(groups([first.i, second.i])); zoneOn(first.thr, XD[1]); });
          search(1, rightC, stat(0, second.i).sse + stat(second.i, first.i).sse, first.i, n, [first.i, second.i], 2900, 1300);
          api.at(4250, function () {
            mark(1, third, stat(0, second.i).sse + stat(second.i, first.i).sse);
            fade(zone, 0); fade(scan, 0); fade(now, 0); fixedCuts([first, second, third]); model(G4);
            api.tile(1, mse(G4).toFixed(2)); api.tile(2, (leftC.length + rightC.length) + ' 个', '又试了');
          });
        },
        function () {                                 // 6 a new patient
          reset(['tree']);
          fixedCuts([first, second, third]); model(G4);
          sticks.forEach(function (s) { fade(s, 0.25); });
          var yhat = predict(G4, NEWX);
          api.tile(0, '4 组'); api.tile(1, mse(G4).toFixed(2)); api.tile(2, '—', '新病人的预测');
          probeLine.setAttribute('y2', sy(YD[0])); probeDot.setAttribute('cy', sy(YD[0])); fade(probe, 1);
          path.forEach(function (p, k) { api.at(300 + k * 260, function () { hot(p, k === path.length - 1 ? 'hit' : 'hot'); }); });
          api.tween(1500, 700, function (u) { var y = lerp(sy(YD[0]), sy(yhat), E(u)); probeLine.setAttribute('y2', y); probeDot.setAttribute('cy', y); });
          api.at(2200, function () { api.tile(2, num(yhat) + ' Gy', '新病人的预测'); });
        },
        function () {                                 // 7 cut until nothing is left to cut
          reset(['bars']);
          fixedCuts([]); model(G4);
          api.tile(0, GF.length + ' 组'); api.tile(1, mse(G4).toFixed(2)); api.tile(2, newSmall.toFixed(2), '新病人身上的误差');
          api.tween(300, 2400, function (u) {
            var k = Math.max(3, Math.round(u * fullIdx.length));
            // the cuts are added in the order the algorithm would find them: the first three, then left to right
            var idx = [first.i, second.i, third.i].concat(fullIdx.filter(function (v) { return v !== first.i && v !== second.i && v !== third.i; })).slice(0, k);
            var g = groups(idx);
            model(g); api.tile(0, g.length + ' 组'); api.tile(1, mse(g).toFixed(2)); api.tile(2, mseNew(g).toFixed(2), '新病人身上的误差');
          });
          api.tween(2700, 500, function (u) { fade(fresh, u); });
        }
      ];
      return function (k) { steps[k](); };
    }
  };

  var FIGURES = { overfit: overfit, kfold: kfold, repeatcv: repeatcv, ci: ci, shuffle: shuffleTest, fitline: fitline, sigmoid: sigmoid, signif: signif, leak: leak, shrink: shrink, split: splitTree, permute: permute, rsquared: rsquared, threshold: thresholdFig, calib: calibFig, netbenefit: netBenefitFig, samplesize: sampleSizeFig, models: staged(modelsScene), growtree: staged(treeScene) };
  function init() {
    document.querySelectorAll('.widget[data-widget]').forEach(function (root) {
      var make = FIGURES[root.getAttribute('data-widget')];
      if (!make || root.getAttribute('data-ready')) return;
      if (!make.keep) root.textContent = '';
      root.setAttribute('data-ready', '1');
      make(root);
    });
  }
  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
