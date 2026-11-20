"""Headless check of the course lessons (/python/, /stat-ml/) at phone width.

For each lesson it taps every "想一想" option, opens the folded hints and answers, runs every
Python block in page order (Pyodide is downloaded from the CDN, so this needs network access),
and drives every interactive figure. It prints what a reader would see, so the numbers quoted
in the lesson text can be compared with the real output before publishing.

    hugo --minify -d /tmp/public
    python tools/check_course.py /tmp/public python/01-first-steps stat-ml/04-cross-validation
    python tools/check_course.py LIVE stat-ml/04-cross-validation        # the published site

The built site is served to the browser by request interception: no HTTP port is opened.
Needs playwright with Chromium. Exit status 1 if a reference answer fails its check, a block
outside an exercise raises (lessons may list blocks that are meant to fail with --expect-errors),
the page scrolls sideways, or the page logs a JavaScript error.
"""
import argparse
import mimetypes
import pathlib
import sys
import time

from playwright.sync_api import sync_playwright

SITE = 'https://ym-xu.github.io/'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('public', help='directory built by hugo, or LIVE')
    ap.add_argument('lessons', nargs='+', help='paths such as stat-ml/04-cross-validation')
    ap.add_argument('--expect-errors', type=int, default=None,
                    help='number of blocks outside exercises that are meant to raise (default: do not check)')
    ap.add_argument('--full', action='store_true', help='print the output of every block')
    args = ap.parse_args()
    live = args.public == 'LIVE'
    root = None if live else pathlib.Path(args.public)

    def serve(route):
        path = route.request.url.split('#')[0].split('?')[0].replace(SITE, '')
        f = root / path
        if f.is_dir() or path.endswith('/') or path == '':
            f = f / 'index.html'
        if not f.exists():
            return route.fulfill(status=404, body='not found')
        ctype = 'text/plain; charset=utf-8' if f.suffix == '.py' else (mimetypes.guess_type(str(f))[0] or 'application/octet-stream')
        route.fulfill(status=200, body=f.read_bytes(), headers={'content-type': ctype})

    failed = False
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        if not live:
            ctx.route(SITE + '**', serve)
        for lesson in args.lessons:
            page = ctx.new_page()
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(f'{SITE}{lesson.strip("/")}/?t={time.time()}', wait_until='networkidle')
            page.evaluate("document.documentElement.classList.add('steps-all')")     # themed practice: show every step
            width = page.evaluate('document.documentElement.scrollWidth')
            print(f'== {lesson}: page width {width}')

            for qi, quiz in enumerate(page.locator('.quiz').all()):
                for opt in quiz.locator('.quiz-opt').all():
                    opt.scroll_into_view_if_needed()
                    opt.tap()
                    mark = '✓' if opt.get_attribute('data-correct') is not None else ' '
                    print(f'  想一想 {qi + 1} {mark} [{opt.inner_text()[:26]}] {quiz.locator(".quiz-why").inner_text()[:70]}')

            for wi, w in enumerate(page.locator('.widget[data-widget]').all()):
                w.scroll_into_view_if_needed()
                name = w.get_attribute('data-widget')
                rng = w.locator('input[type=range]')
                if rng.count():
                    for v in (rng.get_attribute('min'), rng.get_attribute('max')):
                        rng.evaluate("(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); }", v)
                        print(f'  图 {wi + 1} ({name}) 滑块 {v}:', ' | '.join(x.inner_text().replace('\n', ' ') for x in w.locator('.w-bar').all()))
                first = w.locator('.w-btn').first
                if first.count():
                    for k in range(7):
                        if first.is_disabled():
                            break
                        first.tap()
                        page.wait_for_timeout(60)
                    print(f'  图 {wi + 1} ({name}) 点 7 次以后:', w.locator('.w-status').inner_text()[:160])

            page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
            cards = page.locator('.py-runnable')
            raised, t0 = 0, time.time()
            for i in range(cards.count()):
                card = cards.nth(i)
                button = card.locator('.py-run')
                button.scroll_into_view_if_needed()
                button.tap()
                page.wait_for_function('(el) => !el.querySelector(".py-run").disabled', arg=card.element_handle(), timeout=300000)
                out = card.locator('.code-output')
                in_exercise = card.evaluate('el => !!el.closest(".exercise")')
                in_answer = card.evaluate('el => !!el.closest("details")')
                check = out.locator('.code-output-check')
                has_error = out.locator('.code-output-err').count() > 0
                if in_exercise and in_answer:
                    verdict = check.inner_text() if check.count() else '(no check ran)'
                    ok = verdict.startswith('✓')
                    failed = failed or not ok
                    print(f'  [{i}] 参考答案 {"" if ok else "FAILED "}{verdict[:110]}')
                elif not in_exercise:
                    raised += has_error
                    if args.full or has_error:
                        print(f'  [{i}] {"报错" if has_error else "输出"}: {out.inner_text()[:400 if args.full else 120]!r}')
            print(f'  {cards.count()} blocks in {time.time() - t0:.0f} s; blocks outside exercises that raised: {raised}; JavaScript errors: {errors}')
            if width > 390 or errors or (args.expect_errors is not None and raised != args.expect_errors):
                failed = True
            page.close()
        browser.close()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
