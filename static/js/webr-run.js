// In-browser R for the course code blocks, powered by WebR (R compiled to WebAssembly).
// Lazy-loaded: the WebR runtime only boots on the first "Run" click.
import { WebR } from 'https://webr.r-wasm.org/latest/webr.mjs';

let webRPromise = null;
const installed = new Set();

// Base / recommended packages bundled with WebR — never need installing.
const BASE_PKGS = new Set([
  'base', 'stats', 'stats4', 'utils', 'graphics', 'grDevices', 'methods',
  'datasets', 'tools', 'parallel', 'compiler', 'splines', 'grid', 'tcltk'
]);

function bootWebR(setStatus) {
  if (!webRPromise) {
    setStatus('正在启动 R 运行时(首次较慢,约 10–30 秒)…');
    const webR = new WebR();
    webRPromise = webR.init().then(() => webR);
  }
  return webRPromise;
}

// Parse library()/require()/pkg:: from the code to know what to install.
function neededPackages(code) {
  const pkgs = new Set();
  const reLib = /(?:library|require)\(\s*["']?([A-Za-z][A-Za-z0-9.]*)["']?\s*\)/g;
  const reNs = /\b([A-Za-z][A-Za-z0-9.]*)::/g;
  let m;
  while ((m = reLib.exec(code))) pkgs.add(m[1]);
  while ((m = reNs.exec(code))) pkgs.add(m[1]);
  return [...pkgs].filter((p) => !BASE_PKGS.has(p) && !installed.has(p));
}

// gt / gtsummary objects print as a big HTML blob; detect and render it as real HTML.
function looksLikeHTML(s) {
  return /<table[\s>]/i.test(s) || /<div[^>]*\sid=/i.test(s);
}

function renderResult(out, result) {
  out.innerHTML = '';
  const text = (result.output || [])
    .filter((o) => o.type === 'stdout' || o.type === 'stderr')
    .map((o) => o.data)
    .join('\n');
  if (text.trim()) {
    if (looksLikeHTML(text)) {
      const div = document.createElement('div');
      div.className = 'code-output-html';
      div.innerHTML = text;        // 本地 R 自己生成的 gt 表格,安全
      out.appendChild(div);
    } else {
      const pre = document.createElement('pre');
      pre.className = 'code-output-text';
      pre.textContent = text;
      out.appendChild(pre);
    }
  }
  (result.images || []).forEach((img) => {
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.className = 'code-output-plot';
    canvas.getContext('2d').drawImage(img, 0, 0);
    out.appendChild(canvas);
  });
  if (!out.childNodes.length) {
    const pre = document.createElement('pre');
    pre.className = 'code-output-text';
    pre.textContent = '(完成,无文字输出)';
    out.appendChild(pre);
  }
}

async function runCard(card) {
  const codeEl = card.querySelector('pre');
  const out = card.querySelector('.code-output');
  const btn = card.querySelector('.code-run');
  if (!codeEl || !out || !btn) return;

  const code = codeEl.innerText;
  out.hidden = false;
  const setStatus = (msg) => { out.innerHTML = '<pre class="code-output-text code-output-status">' + msg + '</pre>'; };
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = '运行中…';

  try {
    const webR = await bootWebR(setStatus);
    const pkgs = neededPackages(code);
    if (pkgs.length) {
      setStatus('正在安装包:' + pkgs.join(', ') + '(首次需下载)…');
      await webR.installPackages(pkgs, { quiet: true });
      pkgs.forEach((p) => installed.add(p));
    }
    setStatus('运行中…');
    const shelter = await new webR.Shelter();
    try {
      const result = await shelter.captureR(code, { withAutoprint: true, captureGraphics: true });
      renderResult(out, result);
    } finally {
      shelter.purge();
    }
  } catch (err) {
    out.innerHTML = '';
    const pre = document.createElement('pre');
    pre.className = 'code-output-err';
    pre.textContent = String((err && err.message) || err);
    out.appendChild(pre);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // Make runnable code blocks editable so learners can tweak before running.
  document.querySelectorAll('.code-runnable pre').forEach((pre) => {
    pre.setAttribute('contenteditable', 'true');
    pre.setAttribute('spellcheck', 'false');
    pre.setAttribute('title', '可直接在这里修改代码再点 Run');
  });
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.code-run');
    if (btn) runCard(btn.closest('.code-card'));
  });
});
