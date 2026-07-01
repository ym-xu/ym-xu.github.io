// Lesson reading UI: a top reading-progress bar + scroll-spy that highlights
// the current section in the sidebar TOC. Self-guards: does nothing unless a
// lesson page ([data-lesson]) is present, so it's safe to load site-wide.
(function () {
  function init() {
    const article = document.querySelector('[data-lesson]');
    if (!article) return;

    const main = article.querySelector('.lesson-main') || article;
    const bar = document.querySelector('.reading-progress > span');

    // Map TOC links → their target headings (skip links whose anchor is missing).
    const links = Array.from(document.querySelectorAll('.lesson-toc #TableOfContents a'));
    const items = links
      .map((a) => {
        const id = decodeURIComponent((a.getAttribute('href') || '').replace(/^#/, ''));
        return { a, el: id ? document.getElementById(id) : null };
      })
      .filter((x) => x.el);

    const OFFSET = 90; // px below the sticky header counts as "current"

    function updateProgress() {
      if (!bar) return;
      const rect = main.getBoundingClientRect();
      const total = main.offsetHeight - window.innerHeight;
      const scrolled = Math.min(Math.max(-rect.top, 0), Math.max(total, 0));
      const pct = total > 0 ? (scrolled / total) * 100 : 0;
      bar.style.width = pct.toFixed(2) + '%';
    }

    function updateSpy() {
      if (!items.length) return;
      let current = null;
      for (const item of items) {
        if (item.el.getBoundingClientRect().top - OFFSET <= 0) current = item;
        else break;
      }
      if (!current) current = items[0];
      for (const { a } of items) a.classList.toggle('active', a === current.a);
    }

    let ticking = false;
    function onScroll() {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(() => {
        updateProgress();
        updateSpy();
        ticking = false;
      });
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    onScroll();
  }

  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
