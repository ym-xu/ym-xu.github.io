---
title: Reading List
date: 2025-11-11
lastmod: 2025-11-11T10:06:17+02:00
slug:
summary: ""
tags:
  - ""
series: ""
draft: false
toc: true
comments: true
key: value
---

<div class="reading-list-controls">
  <label for="reading-list-sort-column">Sort by:</label>
  <select id="reading-list-sort-column">
    <option value="0">Paper</option>
    <option value="1">Paper Link</option>
    <option value="2">Code / Model</option>
    <option value="3">Venue</option>
    <option value="4">Brand</option>
  </select>
  <label for="reading-list-sort-direction">Order:</label>
  <select id="reading-list-sort-direction">
    <option value="asc">Ascending</option>
    <option value="desc">Descending</option>
  </select>
</div>

<table id="reading-list-table">
  <thead>
    <tr>
      <th>Paper</th>
      <th>Paper Link</th>
      <th>Code / Model</th>
      <th>Venue</th>
      <th>Brand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Visual Document Understanding and Question Answering: A Multi-Agent Collaboration Framework with Test-Time Scaling</td>
      <td><a href="https://arxiv.org/pdf/2508.03404">Link</a></td>
      <td><a href="https://github.com/YU-deep/MACT">Code</a></td>
      <td>arXiv</td>
      <td>Qwen</td>
    </tr>
    <tr>
      <td>DeepSeek-OCR: Contexts Optical Compression</td>
      <td><a href="https://arxiv.org/pdf/2510.18234">Link</a></td>
      <td><a href="https://github.com/deepseek-ai/DeepSeek-OCR">Code</a></td>
      <td>arXiv</td>
      <td>DeepSeek</td>
    </tr>
    <tr>
      <td>DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding</td>
      <td><a href="https://arxiv.org/pdf/2508.08589">Link</a></td>
      <td><a href="https://github.com/wenwenyu/DocThinker">Code</a></td>
      <td>arXiv</td>
      <td>Alibaba</td>
    </tr>
  </tbody>
</table>

<script>
  document.addEventListener('DOMContentLoaded', function () {
    const table = document.getElementById('reading-list-table');
    const tbody = table.querySelector('tbody');
    const columnSelect = document.getElementById('reading-list-sort-column');
    const directionSelect = document.getElementById('reading-list-sort-direction');

    function sortTable() {
      const columnIndex = parseInt(columnSelect.value, 10);
      const direction = directionSelect.value;
      const multiplier = direction === 'asc' ? 1 : -1;

      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((rowA, rowB) => {
        const cellA = rowA.children[columnIndex].textContent.trim().toLowerCase();
        const cellB = rowB.children[columnIndex].textContent.trim().toLowerCase();
        return cellA.localeCompare(cellB) * multiplier;
      });

      rows.forEach((row) => tbody.appendChild(row));
    }

    columnSelect.addEventListener('change', sortTable);
    directionSelect.addEventListener('change', sortTable);
    sortTable();
  });
</script>
