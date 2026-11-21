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
//   (not inside a comment), since Python only accepts straight ones.
import { EditorState, EditorSelection, RangeSetBuilder } from '@codemirror/state';
import { EditorView, keymap, ViewPlugin, Decoration } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentMore, indentLess } from '@codemirror/commands';
import { LanguageSupport, indentUnit, indentOnInput, bracketMatching, syntaxHighlighting, HighlightStyle, syntaxTree } from '@codemirror/language';
import { pythonLanguage } from '@codemirror/lang-python';
import { closeBrackets, closeBracketsKeymap, insertBracket } from '@codemirror/autocomplete';
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

function create(pre, text, opts) {
  opts = opts || {};
  const run = () => { if (opts.onRun) opts.onRun(); return true; };
  const view = new EditorView({
    state: EditorState.create({
      doc: text,
      extensions: [
        keymap.of([{ key: 'Shift-Enter', run }, { key: 'Mod-Enter', run }, { key: 'Tab', run: tabIndent, shift: indentLess }]),
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
