(function(){
  const input = document.getElementById('search-input');
  const results = document.getElementById('results');
  const stats = document.getElementById('search-stats');
  let data = [];

  function render(items){
    results.innerHTML = '';
    if (!items.length){ stats.textContent = 'No results found'; return; }
    stats.textContent = `Found ${items.length} results`;
    const frag = document.createDocumentFragment();
    items.slice(0, 50).forEach(p => {
      const li = document.createElement('li');
      li.innerHTML = `<a href="${p.permalink}">${p.title}</a> <span class="muted">${p.date}</span><p class="summary">${p.summary || ''}</p>`;
      frag.appendChild(li);
    });
    results.appendChild(frag);
  }

  function search(q){
    const term = q.trim().toLowerCase();
    if (!term){ stats.textContent = 'Type to search'; results.innerHTML = ''; return; }
    const hits = data.filter(p => (
      (p.title || '').toLowerCase().includes(term) ||
      (p.summary || '').toLowerCase().includes(term) ||
      (p.tags || []).join(' ').toLowerCase().includes(term)
    ));
    render(hits);
  }

  fetch(new URL('/index.json', window.location)).then(r => r.json()).then(j => {
    data = j.pages || [];
  }).catch(() => { stats.textContent = 'Failed to load index'; });

  input && input.addEventListener('input', (e) => search(e.target.value));
})();
