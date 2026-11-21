"""Several sittings on one device, then a second device: what is kept and shown again.

Usage (the built site is served to a headless browser by route interception, no port is opened):

    python tools/check_progress.py <public dir> <dir for screenshots>

Checks the panel "我的进度" on the course page, the code and answers put back on a lesson, the
blocks run again when Python starts empty, the link that carries progress to another device,
and that the course page, the lessons and the practice share one page width.
"""
import sys, re, mimetypes, pathlib
from playwright.sync_api import sync_playwright
PUBLIC = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); out.mkdir(exist_ok=True)
BASE = 'https://ym-xu.github.io'
def serve(route):
    path = route.request.url.split('#')[0].split('?')[0].replace(BASE + '/', '')
    f = PUBLIC / path
    if f.is_dir() or path.endswith('/') or path == '': f = f / 'index.html'
    if not f.exists(): return route.fulfill(status=404, body='nf')
    ctype = 'text/plain; charset=utf-8' if f.suffix in ('.py', '.csv') else (mimetypes.guess_type(str(f))[0] or 'application/octet-stream')
    route.fulfill(status=200, body=f.read_bytes(), headers={'content-type': ctype})
ok = True
def check(name, cond, detail=''):
    global ok
    ok = ok and bool(cond)
    print(('  ok   ' if cond else '  FAIL ') + name + (' | ' + str(detail) if detail != '' else ''))
def run(pg, card, timeout=300000):
    card.locator('.py-run').scroll_into_view_if_needed(); card.locator('.py-run').click()
    pg.wait_for_function('(el) => !el.querySelector(".py-run").disabled', arg=card.element_handle(), timeout=timeout)
    return card.locator('.code-output').inner_text().strip()
def type_code(pg, card, code):
    pre = card.locator('pre').first; pre.scroll_into_view_if_needed(); pre.click(); pg.keyboard.press('Control+A'); pg.keyboard.insert_text(code)
def answer_of(ex):
    return ex.locator('details .py-runnable code').last.evaluate('e => e.textContent')
def panel(pg):
    return dict(btn=pg.locator('[data-continue]').inner_text(), href=pg.locator('[data-continue]').get_attribute('href'), note=pg.locator('[data-continue-note]').inner_text(),
                stats=[d.inner_text() for d in pg.locator('.course-stats dd').all() if d.is_visible()], tags=pg.evaluate("[...document.querySelectorAll('.lesson-row-tag')].map(e => e.textContent)"))

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={'width': 390, 'height': 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
    ctx.route(BASE + '/**', serve)
    errs = []
    def page():
        pg = ctx.new_page(); pg.on('pageerror', lambda e: errs.append(str(e))); return pg

    print('== day 0: nothing recorded')
    pg = page(); pg.goto(BASE + '/python/', wait_until='networkidle')
    s = panel(pg); check('panel starts empty', s['btn'].startswith('从第 1 课开始') and s['stats'] == ['0 / 7 课', '0 / 34 题', '0 / 6 部分'] and not s['tags'], s)
    order = pg.evaluate("['.course-intro', '.course-side', '.syllabus'].map(q => Math.round(document.querySelector(q).getBoundingClientRect().top + scrollY))")
    check('phone: text, then panel, then syllabus', order == sorted(order), order)
    g = pg.evaluate("(() => { const p = document.querySelector('.pair'); const a = p.querySelector('.lesson-row').getBoundingClientRect(), b = p.querySelector('.practice-row').getBoundingClientRect(); return [Math.round(a.bottom), Math.round(b.top), Math.round(b.left - a.left), getComputedStyle(p.querySelector('.practice-kicker')).display]; })()")
    check('phone: the practice part is under its lesson, indented, and says 课后练习', g[1] >= g[0] - 8 and g[2] > 20 and g[3] != 'none', g)
    check('phone: no sideways scroll', pg.evaluate('document.documentElement.scrollWidth') == 390)
    w = pg.evaluate("['.course-intro', '.course-side', '.course-body'].map(q => Math.round(document.querySelector(q).getBoundingClientRect().width))")
    check('phone: one column, each block as wide as the page', min(w) >= 340 and len(set(w)) == 1, w)
    pg.locator('.course-side').screenshot(path=str(out / 'panel-empty-phone.png'))

    print('== day 1: lesson 2, one exercise passed, one half written, one question answered, stopped in section 4')
    pg.goto(BASE + '/python/02-pandas/', wait_until='networkidle')
    exs = pg.locator('.exercise'); ex1, ex2 = exs.nth(0), exs.nth(1)
    card1, card2 = ex1.locator('.py-runnable').first, ex2.locator('.py-runnable').first
    type_code(pg, card1, answer_of(ex1))
    o = run(pg, card1)
    check('exercise 1 passes although nothing above it was run', '✓' in o and '已先运行本页前面的' in o, o.splitlines()[0][:60])
    type_code(pg, card2, 'half = "写到一半"\n')
    pg.wait_for_timeout(900)
    quiz = pg.locator('.quiz').first; quiz.locator('.quiz-opt[data-correct]').scroll_into_view_if_needed(); quiz.locator('.quiz-opt[data-correct]').click()
    heads = pg.locator('.lesson-main .content h2[id]'); h4 = heads.nth(3); h4_id, h4_text = h4.get_attribute('id'), h4.inner_text()
    h4.evaluate('e => e.scrollIntoView({block: "start"})'); pg.wait_for_timeout(300); pg.mouse.wheel(0, 160); pg.wait_for_timeout(1400)
    pg.close()

    print('== day 2: the course page')
    pg = page(); pg.goto(BASE + '/python/', wait_until='networkidle')
    s = panel(pg)
    check('continue points at lesson 2', s['btn'] == '继续: 第 2 课 · 表格数据 pandas →', s['btn'])
    check('... at the section read last', s['href'].endswith('#' + h4_id) or re.sub('%..', '', s['href']).startswith('/python/02-pandas/#'), s['href'][:60])
    check('note names the section and the day', h4_text in s['note'] and '今天' in s['note'], s['note'])
    check('counts and the row tag', s['stats'] == ['0 / 7 课', '1 / 34 题', '0 / 6 部分'] and s['tags'] == ['在学 · 练习 1 / 5'], (s['stats'], s['tags']))
    pg.locator('.course-side').screenshot(path=str(out / 'panel-day2-phone.png'))
    pg.locator('[data-continue]').click(); pg.wait_for_load_state('networkidle'); pg.wait_for_timeout(500)
    check('continue opens the lesson at that section', abs(pg.evaluate("id => document.getElementById(id).getBoundingClientRect().top", h4_id)) < 140)
    exs = pg.locator('.exercise'); ex1, ex2 = exs.nth(0), exs.nth(1)
    card1, card2 = ex1.locator('.py-runnable').first, ex2.locator('.py-runnable').first
    check('exercise 1 still marked as passed', 'passed' in (ex1.get_attribute('class') or ''))
    check('the code of exercise 1 is back', card1.locator('code').first.evaluate('e => e.textContent').strip() == answer_of(ex1).strip())
    check('the half-written code of exercise 2 is back', '写到一半' in card2.locator('code').first.evaluate('e => e.textContent'))
    check('the answered question is shown again', pg.locator('.quiz').first.locator('.quiz-opt.right').count() == 1 and pg.locator('.quiz').first.locator('.quiz-why').count() == 1)
    o = run(pg, card1)
    check('a new day: Python is empty, earlier blocks are run first', '✓' in o and '已先运行本页前面的' in o, o.splitlines()[0][:60])
    card2.locator('.code-reset').click(); pg.wait_for_timeout(200)
    check('重置 puts the starter code back', '写到一半' not in card2.locator('code').first.evaluate('e => e.textContent') and not card2.locator('.code-reset').is_visible())
    pg.goto(BASE + '/python/02-pandas/', wait_until='networkidle'); pg.wait_for_timeout(300)
    bar = pg.locator('.resume-bar')
    check('opened without a place: the bar offers the section read last', bar.count() == 1 and h4_text in bar.inner_text(), bar.inner_text() if bar.count() else '')
    bar.screenshot(path=str(out / 'resume-bar-phone.png'))
    bar.locator('.resume-go').click(); pg.wait_for_timeout(700)
    check('... and jumps there', abs(pg.evaluate("id => document.getElementById(id).getBoundingClientRect().top", h4_id)) < 140 and pg.locator('.resume-bar').count() == 0)

    print('== day 3: the practice, continued after a break')
    pg.goto(BASE + '/python/practice/02-clean/', wait_until='networkidle')
    setup = pg.locator('.py-runnable').first
    steps = pg.locator('.steps > .step')
    type_code(pg, steps.nth(0).locator('.py-runnable').first, answer_of(steps.nth(0)))
    o = run(pg, steps.nth(0).locator('.py-runnable').first)
    check('step 1 passes; the preparation block above was run for it', '✓' in o and '已先运行本页前面的 1 个代码块' in o, o.splitlines()[0][:50])
    pg.locator('.steps-next').click(); pg.wait_for_timeout(300)
    pg.close(); pg = page(); pg.goto(BASE + '/python/practice/02-clean/', wait_until='networkidle'); pg.wait_for_timeout(300)
    steps = pg.locator('.steps > .step')
    check('back on step 2', pg.locator('.steps-label').inner_text() == '第 2 / 6 步', pg.locator('.steps-label').inner_text())
    type_code(pg, steps.nth(1).locator('.py-runnable').first, answer_of(steps.nth(1)))
    o = run(pg, steps.nth(1).locator('.py-runnable').first)
    check('step 2 passes: preparation and my step 1 were run first', '✓' in o and '已先运行本页前面的 2 个代码块' in o, o.splitlines()[0][:50])
    pg.goto(BASE + '/python/', wait_until='networkidle')
    s = panel(pg)
    check('course page: the practice is where to go on', s['btn'] == '继续: 课后练习 2 · 读入和清理审计数据 →' and '已通过 2 / 5' in s['tags'], (s['btn'], s['tags']))
    check('the head of the practice column counts the parts', pg.locator('[data-practice-count]').inner_text() == '已完成 0 / 6 部分', pg.locator('[data-practice-count]').inner_text())
    pg.goto(BASE + '/python/practice/', wait_until='networkidle')
    s = panel(pg); check('practice page: its own counts', s['stats'] == ['0 / 6 部分', '2 / 25 题'] and s['btn'].startswith('继续: 练习 2'), s)

    print('== another device: the progress travels as a link')
    pg.goto(BASE + '/python/', wait_until='networkidle')
    pg.locator('[data-progress-export]').click(); pg.wait_for_function("document.querySelector('[data-progress-box] textarea').value.length > 0")
    link = pg.locator('[data-progress-box] textarea').input_value()
    check('a link to the course page with the progress in it', link.startswith(BASE + '/python/#progress=') and len(link) < 2500, f'{len(link)} characters')
    pg.locator('.course-side').screenshot(path=str(out / 'panel-export-phone.png'))
    ctx2 = b.new_context(viewport={'width': 1280, 'height': 900}); ctx2.route(BASE + '/**', serve)
    d = ctx2.new_page(); d.on('pageerror', lambda e: errs.append(str(e)))
    d.goto(link, wait_until='networkidle'); d.wait_for_function("document.querySelector('[data-progress-msg]').textContent.length > 0")
    s2 = panel(d)
    check('opened there: merged, same continue button and tags', '已把链接里的进度合并' in d.locator('[data-progress-msg]').inner_text() and s2['btn'] == s['btn'].replace('练习 2 ·', '课后练习 2 ·') or s2['btn'].startswith('继续: 课后练习 2'), (s2['btn'], s2['tags']))
    check('the link is taken out of the address', '#' not in d.url, d.url)
    d.goto(BASE + '/python/02-pandas/', wait_until='networkidle')
    check('there, exercise 1 counts as passed; code does not travel', 'passed' in (d.locator('.exercise').nth(0).get_attribute('class') or '') and d.locator('.exercise').nth(0).locator('.py-runnable').first.locator('.code-reset').is_visible() is False)

    print('== widths on a wide screen')
    def box(pg_, sel):
        return pg_.evaluate("q => { const r = document.querySelector(q).getBoundingClientRect(); return [Math.round(r.left), Math.round(r.width)]; }", sel)
    lesson_sheet, lesson_main, lesson_toc = box(d, '.sheet'), box(d, '.lesson-main'), box(d, '.lesson-toc')
    d.goto(BASE + '/python/', wait_until='networkidle'); d.wait_for_timeout(500)
    course_sheet, course_main, course_side = box(d, '.sheet'), box(d, '.course-body'), box(d, '.course-side')
    check('course page and lesson: same sheet', course_sheet == lesson_sheet, (course_sheet, lesson_sheet))
    check('same text column', course_main == lesson_main, (course_main, lesson_main))
    check('panel where the lesson has its contents', course_side == lesson_toc, (course_side, lesson_toc))
    g = d.evaluate("(() => { const p = document.querySelector('.pair'); const a = p.querySelector('.lesson-row').getBoundingClientRect(), b = p.querySelector('.practice-row').getBoundingClientRect(); return [Math.round(a.top), Math.round(b.top), Math.round(a.right), Math.round(b.left), getComputedStyle(p.querySelector('.practice-kicker')).display]; })()")
    check('wide screen: lesson and practice side by side on one row', abs(g[0] - g[1]) <= 2 and g[3] >= g[2] and g[4] == 'none', g)
    d.goto(BASE + '/python/practice/', wait_until='networkidle'); check('practice page: same sheet', box(d, '.sheet') == lesson_sheet, box(d, '.sheet'))
    d.goto(BASE + '/stat-ml/', wait_until='networkidle'); check('stat-ml course page: same sheet', box(d, '.sheet') == lesson_sheet, box(d, '.sheet'))
    d.goto(BASE + '/python/', wait_until='networkidle'); d.wait_for_timeout(400)
    h = d.evaluate('document.documentElement.scrollHeight')
    for yy in range(0, h, 500):
        d.evaluate('window.scrollTo(0, %d)' % yy); d.wait_for_timeout(100)
    d.evaluate('window.scrollTo(0, 0)'); d.wait_for_timeout(500); d.screenshot(path=str(out / 'course-desktop.png'), full_page=True)
    d.evaluate('window.scrollTo(0, 900)'); d.wait_for_timeout(300)
    check('the panel stays in view while the syllabus scrolls', 0 <= d.evaluate("document.querySelector('.course-side-inner').getBoundingClientRect().top") < 60)
    d.goto(BASE + '/python/02-pandas/', wait_until='networkidle'); d.wait_for_timeout(400); d.screenshot(path=str(out / 'lesson-desktop.png'))
    check('no JavaScript errors', not errs, errs)
    b.close()
print('ALL OK' if ok else 'SOME CHECKS FAILED')
