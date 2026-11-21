#!/usr/bin/env bash
# Builds assets/js/vendor/py-editor.js, the code editor of the course code boxes, from editor.js.
# The versions are pinned in package-lock.json. Run it after changing editor.js or a version:
#     cd tools/editor && ./build.sh
set -euo pipefail
cd "$(dirname "$0")"
npm ci --no-audit --no-fund
BANNER="/*! py-editor: the code editor of the course code boxes (source: tools/editor/editor.js).
Bundles CodeMirror 6 and Lezer (https://codemirror.net), MIT License:
Copyright (C) 2018-2021 by Marijn Haverbeke <marijn@haverbeke.berlin> and others
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */"
npx esbuild editor.js --bundle --minify --format=iife --target=es2020 --banner:js="$BANNER" --outfile=../../assets/js/vendor/py-editor.js
