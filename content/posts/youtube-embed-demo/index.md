---
title: "YouTube Embed Demo"
date: 2025-09-30T12:20:00+08:00
lastmod: 2025-09-30T12:20:00+08:00
slug: "youtube-embed-demo"
summary: "Two ways to embed a YouTube video: Hugo shortcode and manual iframe (privacy-friendly)."
tags: ["video", "embed", "how-to"]
draft: false
toc: true
comments: true
---

This post shows two simple ways to embed a YouTube video. The site config enables YouTube’s privacy-enhanced mode, so embeds use `youtube-nocookie.com`.

## 1) Hugo shortcode (recommended)

Use the built-in shortcode with the video ID. For `https://www.youtube.com/watch?v=w7Ft2ymGmfc`, the ID is `w7Ft2ymGmfc`.
For reliability during your current setup, the live shortcode embed is omitted here to avoid parser issues.

Notes:
- It’s responsive and lazy by default in modern Hugo.
- To set a start time with the shortcode, the syntax is: youtube id="w7Ft2ymGmfc" start="60" (shown without braces to avoid rendering).

## 2) Manual iframe (more control)

If you need custom attributes, you can embed an iframe directly. The theme includes a small `.video` class to keep a 16:9 aspect ratio.

<div class="video">
  <iframe
    src="https://www.youtube-nocookie.com/embed/w7Ft2ymGmfc"
    title="YouTube video"
    loading="lazy"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    referrerpolicy="strict-origin-when-cross-origin"
    allowfullscreen>
  </iframe>
  
</div>

Use the shortcode for most cases; switch to the iframe when you need full control over attributes or layout.
