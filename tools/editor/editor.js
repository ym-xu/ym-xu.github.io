// The code editor of the course code boxes (/python/ and /stat-ml/): CodeMirror 6.
// tools/editor/build.sh bundles this file into assets/js/vendor/py-editor.js, which
// layouts/partials/scripts.html loads before pyodide-run.js. It only defines window.PyEditor;
// pyodide-run.js decides which blocks become editors, and keeps the plain editable <pre> when this
// file did not load.
//
// What the reader gets:
// - Python colours while typing, the same as the static blocks (the highlight classes are Chroma's,
//   coloured by syntax.css, light and dark);
// - Enter keeps the indentation and indents after a colon; Tab / Shift-Tab indent and dedent;
// - brackets and quotes close themselves; undo and redo;
// - Shift-Enter or Ctrl/Cmd-Enter runs the block;
// - a curly quote from a phone keyboard or a Chinese input method becomes a straight quote
//   (not inside a comment), since Python only accepts straight ones;
// - with opts.assist (pyodide-run.js passes it): completion of names, attributes and column names,
//   and a hint with the parameters of the function being called. On a computer Tab opens the list
//   (as in JupyterLab); on a phone it opens by itself after "." or two letters, and Enter stays a new
//   line. The answers come from the running Python, through the functions in opts.assist.
import { EditorState, EditorSelection, RangeSetBuilder, StateField, StateEffect } from '@codemirror/state';
import { EditorView, keymap, ViewPlugin, Decoration, showTooltip, tooltips } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentMore, indentLess } from '@codemirror/commands';
import { LanguageSupport, indentUnit, indentOnInput, bracketMatching, syntaxHighlighting, HighlightStyle, syntaxTree } from '@codemirror/language';
import { pythonLanguage } from '@codemirror/lang-python';
import { closeBrackets, closeBracketsKeymap, insertBracket, autocompletion, startCompletion, acceptCompletion, closeCompletion,
  completionStatus, moveCompletionSelection } from '@codemirror/autocomplete';
import { tags as t } from '@lezer/highlight';

// Token -> Chroma class, as the static blocks have them (Hugo's highlighter, github / github-dark).
const chroma = HighlightStyle.define([
  { tag: t.keyword, class: 'k' },
  { tag: t.moduleKeyword, class: 'kn' },
  { tag: t.operatorKeyword, class: 'ow' },
  { tag: [t.bool, t.null], class: 'kc' },
  { tag: t.comment, class: 'c1' },
  { tag: [t.string, t.special(t.string)], class: 's2' },
  { tag: t.escape, class: 'se' },
  { tag: t.number, class: 'mi' },
  { tag: [t.operator, t.modifier], class: 'o' },
  { tag: t.punctuation, class: 'p' },
  { tag: t.function(t.definition(t.variableName)), class: 'nf' },
  { tag: t.definition(t.className), class: 'nc' },
  { tag: t.meta, class: 'nd' },
]);

// Chroma colours the built-in functions (print, len, ...) when they stand on their own; the grammar
// does not tell them apart from other names, so they are marked here.
const BUILTINS = new Set(('abs all any bool callable chr dict dir divmod enumerate filter float format frozenset '
  + 'getattr hasattr hash help id input int isinstance issubclass iter len list map max min next object '
  + 'open ord pow print range repr reversed round set setattr slice sorted str sum super tuple type vars zip').split(' '));
const builtinMark = Decoration.mark({ class: 'nb' });
function builtinRanges(view) {
  const b = new RangeSetBuilder();
  for (const { from, to } of view.visibleRanges) {
    syntaxTree(view.state).iterate({
      from, to,
      enter: (n) => {
        if (n.name === 'VariableName' && BUILTINS.has(view.state.doc.sliceString(n.from, n.to))) b.add(n.from, n.to, builtinMark);
      },
    });
  }
  return b.finish();
}
const builtinNames = ViewPlugin.fromClass(class {
  constructor(view) { this.decorations = builtinRanges(view); }
  update(u) {
    if (u.docChanged || u.viewportChanged || syntaxTree(u.startState) !== syntaxTree(u.state)) this.decorations = builtinRanges(u.view);
  }
}, { decorations: (v) => v.decorations });

// Tab: with nothing selected, spaces up to the next multiple of 4; with a selection, indent the lines.
function tabIndent(view) {
  const { state } = view;
  if (state.selection.ranges.some((r) => !r.empty)) return indentMore(view);
  view.dispatch(state.changeByRange((r) => {
    const n = 4 - ((r.head - state.doc.lineAt(r.head).from) % 4);
    return { changes: { from: r.head, insert: ' '.repeat(n) }, range: EditorSelection.cursor(r.head + n) };
  }), { userEvent: 'input.indent', scrollIntoView: true });
  return true;
}

// Curly quotes typed outside a comment become straight ones.
const STRAIGHT = { '“': '"', '”': '"', '„': '"', '‘': "'", '’': "'" };
function inComment(state, pos) {
  for (let n = syntaxTree(state).resolveInner(pos, -1); n; n = n.parent) if (n.name === 'Comment') return true;
  return false;
}
const straightQuotes = EditorView.inputHandler.of((view, from, to, text) => {
  const q = STRAIGHT[text];
  if (!q || inComment(view.state, from)) return false;
  const sel = view.state.selection;
  const tr = sel.ranges.length === 1 && sel.main.from === from && sel.main.to === to ? insertBracket(view.state, q) : null;
  view.dispatch(tr || { changes: { from, to, insert: q }, selection: { anchor: from + 1 }, userEvent: 'input.type' });
  return true;
});

// ---- completion and call hints (opts.assist) ----
const TOUCH = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
const EXPR = String.raw`[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[(?:"[^"\n]*"|'[^'\n]*'|-?\d+)\])*`;   // name, .attr, ["key"] or [0]
const KEY_BEFORE = new RegExp('(' + EXPR + String.raw`)\[\s*(["'])([^"'\n]*)$`);
const ATTR_BEFORE = new RegExp('(' + EXPR + String.raw`)\.(\w*)$`);

function insideNode(state, pos, names) {
  for (let n = syntaxTree(state).resolveInner(pos, -1); n; n = n.parent) if (names.includes(n.name)) return true;
  return false;
}

function completionSource(assist) {
  return (ctx) => {
    const line = ctx.state.doc.lineAt(ctx.pos);
    const before = line.text.slice(0, ctx.pos - line.from);
    const key = KEY_BEFORE.exec(before);                            // df["wa|  a column name or a dict key
    if (key) {
      const keys = assist.keys(key[1]);
      if (!keys || !keys.length) return null;
      return { from: ctx.pos - key[3].length, options: keys.map((k) => ({ label: k, detail: '列', type: 'property' })), validFor: /^[^"'\n]*$/ };
    }
    if (insideNode(ctx.state, ctx.pos, ['Comment', 'String', 'FormatString'])) return null;
    const attr = ATTR_BEFORE.exec(before);                          // np.me|  df["x"].me|
    if (attr) {
      const names = assist.attrs(attr[1]);
      if (!names || !names.length) return null;
      return { from: ctx.pos - attr[2].length, options: names.map((n) => ({ label: n })), validFor: /^\w*$/ };
    }
    const word = /(?:^|[^\w.])([A-Za-z_]\w*)$/.exec(before);       // pri|
    if (!word && !ctx.explicit) return null;
    if (word && !ctx.explicit && word[1].length < 2) return null;
    const from = ctx.pos - (word ? word[1].length : 0);
    const options = assist.names(), have = new Set(options.map((o) => o.label));
    for (const m of ctx.state.doc.toString().matchAll(/[A-Za-z_]\w+/g)) {   // names written in this box, not yet run
      if (m.index === from || have.has(m[0])) continue;
      have.add(m[0]);
      options.push({ label: m[0], boost: 1 });
    }
    return { from, options, validFor: /^\w*$/ };
  };
}

// Tab right after a name, a "." or inside df["...: complete, as in JupyterLab; elsewhere: indent.
function wantsCompletion(state) {
  const pos = state.selection.main.head, line = state.doc.lineAt(pos);
  const before = line.text.slice(0, pos - line.from);
  return state.selection.main.empty && (/[\w.]$/.test(before) || KEY_BEFORE.test(before));
}

// The call the cursor is in: the innermost "(" not closed before pos, and the expression before it.
function openCall(doc, pos) {
  const from = Math.max(0, pos - 3000), text = doc.sliceString(from, pos);
  let depth = 0;
  for (let i = text.length - 1; i >= 0; i--) {
    const ch = text[i];
    if (ch === ')' || ch === ']' || ch === '}') depth++;
    else if (ch === '(' || ch === '[' || ch === '{') {
      if (depth > 0) { depth--; continue; }
      if (ch !== '(') return null;
      const m = new RegExp('(' + EXPR + String.raw`)\s*$`).exec(text.slice(Math.max(0, i - 300), i));
      return m ? { open: from + i, callee: m[1] } : null;
    }
  }
  return null;
}

// Which argument the cursor is in, from the text between "(" and the cursor; null once the call is closed.
function currentArg(inside) {
  let depth = 0, index = 0, start = 0, keywords = false, quote = null;
  for (let i = 0; i < inside.length; i++) {
    const ch = inside[i];
    if (quote) { if (ch === quote) quote = null; continue; }
    if (ch === '"' || ch === "'") quote = ch;
    else if (ch === '(' || ch === '[' || ch === '{') depth++;
    else if (ch === ')' || ch === ']' || ch === '}') { if (depth === 0) return null; depth--; }
    else if (ch === ',' && depth === 0) {
      if (/^\s*[A-Za-z_]\w*\s*=(?!=)/.test(inside.slice(start, i))) keywords = true;
      index++; start = i + 1;
    }
  }
  const kw = /^\s*([A-Za-z_]\w*)\s*=(?!=)/.exec(inside.slice(start));
  return { index, keyword: kw ? kw[1] : null, keywordsBefore: keywords };
}

function currentParam(params, arg) {
  if (arg.keyword) {
    const i = params.findIndex((p) => p[0] === arg.keyword);
    return i >= 0 ? i : params.findIndex((p) => p[1] === 'VAR_KEYWORD');
  }
  if (arg.keywordsBefore) return -1;
  let k = arg.index;
  for (let i = 0; i < params.length; i++) {
    const kind = params[i][1];
    if (kind === 'POSITIONAL_ONLY' || kind === 'POSITIONAL_OR_KEYWORD') { if (k === 0) return i; k--; }
    else if (kind === 'VAR_POSITIONAL') return i;
    else break;
  }
  return -1;
}

function sigTooltip(open, info, current) {
  return {
    pos: open, above: true, strictSide: false, arrow: false,
    create: () => {
      const dom = document.createElement('div');
      dom.className = 'py-sig';
      const head = document.createElement('div');
      head.className = 'py-sig-call';
      head.append(info.name + '(');
      info.parts.forEach(([text, i], k) => {
        if (k) head.append(', ');
        if (i >= 0 && i === current) { const b = document.createElement('b'); b.textContent = text; head.append(b); }
        else head.append(text);
      });
      head.append(info.name ? ')' : '');
      dom.append(head);
      if (info.doc) { const d = document.createElement('div'); d.className = 'py-sig-doc'; d.textContent = info.doc; dom.append(d); }
      return { dom };
    },
  };
}

const setSig = StateEffect.define();
const sigField = StateField.define({
  create: () => null,
  update(value, tr) {
    let fresh = false;
    for (const e of tr.effects) if (e.is(setSig)) { value = e.value; fresh = true; }
    if (!value) return null;
    let open = value.open;
    if (tr.docChanged) open = tr.changes.mapPos(open, 1);
    const head = tr.state.selection.main.head;
    if (tr.state.doc.sliceString(open, open + 1) !== '(' || head <= open) return null;
    const arg = currentArg(tr.state.doc.sliceString(open + 1, head));
    if (!arg) return null;
    const current = currentParam(value.info.params, arg);
    if (!fresh && open === value.open && current === value.current) return value;
    return { open, info: value.info, current, tooltip: sigTooltip(open, value.info, current) };
  },
  provide: (f) => showTooltip.from(f, (v) => (v ? v.tooltip : null)),
});

// After "(" or "," is typed, ask the running Python for the signature of the function being called.
function sigTrigger(assist) {
  return EditorView.updateListener.of((u) => {
    if (!u.docChanged || !u.transactions.some((tr) => tr.isUserEvent('input'))) return;
    let typed = '';
    u.changes.iterChanges((fa, ta, fb, tb, text) => { typed += text.toString(); });
    if (!/[(,]/.test(typed)) return;
    const view = u.view, call = openCall(view.state.doc, view.state.selection.main.head);
    if (!call) return;
    const current = view.state.field(sigField, false);
    if (current && current.open === call.open) return;               // already showing this call
    const info = assist.signature(call.callee);
    if (info) setTimeout(() => view.dispatch({ effects: setSig.of({ open: call.open, info, current: -2 }) }), 0);
  });
}

function assistExtensions(assist) {
  return [
    keymap.of([
      { key: 'Tab', run: (v) => (completionStatus(v.state) === 'active' ? acceptCompletion(v)
        : wantsCompletion(v.state) ? startCompletion(v) : tabIndent(v)), shift: indentLess },
      { key: 'Enter', run: (v) => !TOUCH && completionStatus(v.state) === 'active' && acceptCompletion(v) },
      { key: 'ArrowDown', run: (v) => completionStatus(v.state) === 'active' && moveCompletionSelection(true)(v) },
      { key: 'ArrowUp', run: (v) => completionStatus(v.state) === 'active' && moveCompletionSelection(false)(v) },
      { key: 'Ctrl-Space', run: startCompletion },
      { key: 'Escape', run: (v) => {
        if (completionStatus(v.state)) return closeCompletion(v);
        if (v.state.field(sigField, false)) { v.dispatch({ effects: setSig.of(null) }); return true; }
        return false;
      } },
    ]),
    autocompletion({
      override: [completionSource(assist)], activateOnTyping: TOUCH, defaultKeymap: false, icons: false, filterStrict: true,
      maxRenderedOptions: 60, closeOnBlur: true, tooltipClass: () => 'py-complete',
    }),
    sigField,
    sigTrigger(assist),
    // popups go on <body>, so the code card does not clip them; on a phone they stay above the keyboard
    tooltips({
      parent: document.body,
      tooltipSpace: () => {
        const vv = window.visualViewport;
        return vv ? { top: vv.offsetTop, left: vv.offsetLeft, bottom: vv.offsetTop + vv.height, right: vv.offsetLeft + vv.width }
          : { top: 0, left: 0, bottom: window.innerHeight, right: window.innerWidth };
      },
    }),
  ];
}

function create(pre, text, opts) {
  opts = opts || {};
  const run = () => { if (opts.onRun) opts.onRun(); return true; };
  const view = new EditorView({
    state: EditorState.create({
      doc: text,
      extensions: [
        keymap.of([{ key: 'Shift-Enter', run }, { key: 'Mod-Enter', run }]),
        opts.assist ? assistExtensions(opts.assist) : keymap.of([{ key: 'Tab', run: tabIndent, shift: indentLess }]),
        history(),
        closeBrackets(),
        indentOnInput(),
        bracketMatching(),
        keymap.of([...closeBracketsKeymap, ...historyKeymap, ...defaultKeymap]),
        indentUnit.of('    '),
        EditorState.tabSize.of(4),
        new LanguageSupport(pythonLanguage),
        syntaxHighlighting(chroma),
        builtinNames,
        straightQuotes,
        EditorView.contentAttributes.of({ class: 'chroma', 'aria-label': opts.label || 'Python 代码,可以直接修改' }),
        EditorView.editorAttributes.of({ class: opts.exercise ? 'py-editor py-editor-exercise' : 'py-editor' }),
        EditorView.updateListener.of((u) => { if (u.docChanged && opts.onChange) opts.onChange(); }),
      ],
    }),
  });
  pre.replaceWith(view.dom);
  return {
    view,
    get: () => view.state.doc.toString(),
    set: (s) => view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: s }, selection: { anchor: 0 } }),
    focus: () => view.focus(),
  };
}

window.PyEditor = { create };
