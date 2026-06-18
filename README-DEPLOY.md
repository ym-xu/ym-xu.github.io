# ym-xu.github.io — complete redesign drop-in

Everything in one package, organized exactly like your repo. Copy the folders into
your `ym-xu.github.io/` repo root (merging, not replacing the repo) and push.

## File map

```
config/_default/config.toml                       REPLACE  (new nav + home params)
content/_index.md                                 NEW      (home bio)
content/projects/_index.md  + 3 project .md        NEW      (Projects section)
content/publications/icml-madqa/index.md           NEW      (ICML 2026, Oral — MADQA)
content/publications/icdar-adanav/index.md          NEW      (ICDAR 2026 — AdaNav)
content/publications/lrec-coling-2024-kg-qa/index.md  REPLACE front matter + body
content/publications/iccs-fglab/index.md            REPLACE front matter + body
data/awards.yaml                                   NEW      (home Awards list)
layouts/index.html                                 OVERRIDE (new home)
layouts/projects/list.html, single.html            NEW
layouts/publications/list.html, single.html        OVERRIDE (badge, bold author, pills)
layouts/partials/publication_cards.html            OVERRIDE
layouts/partials/social.html                       NEW
static/css/custom.css                              REPLACE  (combined stylesheet)
```

Nothing here touches the `themes/mu-min/` folder — these are site-root overrides,
which Hugo prefers over the theme.

---

## Option A — local Git (recommended)

```bash
cd path/to/ym-xu.github.io
# copy the folders from this package into the repo, merging by path
hugo server -D            # preview at http://localhost:1313
git add -A
git commit -m "Redesign: home/projects/nav + publications (MADQA, AdaNav, Oral badges)"
git push
```
GitHub Pages rebuilds automatically (watch the Actions tab). Live in ~1–2 min.

## Option B — GitHub web upload (no terminal)

1. Open https://github.com/ym-xu/ym-xu.github.io
2. For each file: navigate to its folder (or use **Add file → Upload files** and
   drag the folder), or **Add file → Create new file** and paste contents, setting
   the path (e.g. `content/publications/icml-madqa/index.md`).
3. For the 4 REPLACE/OVERRIDE files, open the existing file → pencil (Edit) →
   paste the new contents.
4. Commit to `main`. Pages rebuilds automatically.

> Web upload is fine but tedious for many files; Option A is much faster.

---

## After deploying — finish these

- **CV PDF**: add `static/cv.pdf` (or change `params.home.cvUrl` in config). The
  Download-CV button 404s until then.
- **AdaNav PDF**: `links.pdf` is empty in `content/publications/icdar-adanav/index.md` —
  fill it when the link exists.
- **Hierarchical Planner** project blurb is placeholder — edit to taste.
- **About page**: removed from the nav but `content/about/` still exists (reachable by
  URL). Delete it if unwanted. The home hero reuses `content/about/avatar.jpg`, so if
  you delete About, move the avatar (e.g. `static/avatar.jpg`) and update the `$avatar`
  lookup in `layouts/index.html`.
