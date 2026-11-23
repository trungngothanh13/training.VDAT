#!/usr/bin/env python3
"""
Lightweight side-by-side web app comparing Stock YOLO26 vs Custom Trained YOLO26
on random or specific frames from test videos (zero web-framework dependencies).

Usage:
    python compare_app.py --port 7860
    # On ThunderCompute, open locally via: ssh -L 7860:localhost:7860 tnr-0
"""

import argparse
import base64
import datetime
import json
import random
import threading
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from model_utils import (
    ROOT_DIR,
    draw_detections_bgr,
    get_device_info,
    get_precision_kwargs,
    list_test_videos,
    make_side_by_side_bgr,
    parse_names_dict,
    resolve_trained_weights,
    run_model_detections,
)
from evaluate_compare import match_between_models


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>VDAT Model Comparator — Stock YOLO26 vs Trained YOLO26</title>
  <style>
    :root {
      color-scheme: light dark;

      --bg-light: #f4f6fa;
      --bg-dark: #0c111d;
      --surface-light: #ffffff;
      --surface-dark: #151c2c;
      --surface-alt-light: #eef2f7;
      --surface-alt-dark: #1c2538;
      --border-light: #d5dde8;
      --border-dark: #28344d;
      --text-light: #0f172a;
      --text-dark: #eef2ff;
      --muted-light: #475569;
      --muted-dark: #94a3b8;
      --brand-light: #1d4ed8;
      --brand-dark: #60a5fa;
      --brand-text-light: #ffffff;
      --brand-text-dark: #071326;
      --good-light: #15803d;
      --good-dark: #4ade80;
      --warn-light: #b45309;
      --warn-dark: #fbbf24;

      --bg: var(--bg-light);
      --surface: var(--surface-light);
      --surface-alt: var(--surface-alt-light);
      --border: var(--border-light);
      --text: var(--text-light);
      --muted: var(--muted-light);
      --brand: var(--brand-light);
      --brand-text: var(--brand-text-light);
      --good: var(--good-light);
      --warn: var(--warn-light);

      @media (prefers-color-scheme: dark) {
        --bg: var(--bg-dark);
        --surface: var(--surface-dark);
        --surface-alt: var(--surface-alt-dark);
        --border: var(--border-dark);
        --text: var(--text-dark);
        --muted: var(--muted-dark);
        --brand: var(--brand-dark);
        --brand-text: var(--brand-text-dark);
        --good: var(--good-dark);
        --warn: var(--warn-dark);
      }

      @supports (color: light-dark(white, black)) {
        --bg: light-dark(var(--bg-light), var(--bg-dark));
        --surface: light-dark(var(--surface-light), var(--surface-dark));
        --surface-alt: light-dark(var(--surface-alt-light), var(--surface-alt-dark));
        --border: light-dark(var(--border-light), var(--border-dark));
        --text: light-dark(var(--text-light), var(--text-dark));
        --muted: light-dark(var(--muted-light), var(--muted-dark));
        --brand: light-dark(var(--brand-light), var(--brand-dark));
        --brand-text: light-dark(var(--brand-text-light), var(--brand-text-dark));
        --good: light-dark(var(--good-light), var(--good-dark));
        --warn: light-dark(var(--warn-light), var(--warn-dark));
      }

      accent-color: var(--brand);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
      accent-color: var(--brand);
    }

    .container {
      max-inline-size: 1640px;
      margin-inline: auto;
      padding: 1rem 1.25rem 2.5rem;
      display: grid;
      gap: 1rem;
    }

    header.topbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      padding: 0.85rem 1.1rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
    }

    .title-group h1 {
      margin: 0;
      font-size: 1.15rem;
      font-weight: 700;
    }

    .subtitle {
      font-size: 0.8rem;
      color: var(--muted);
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.75rem 1rem;
      padding: 0.85rem 1.1rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
    }

    .field {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      font-size: 0.84rem;
    }

    .field label {
      font-weight: 600;
      color: var(--muted);
    }

    select, input[type="number"], button {
      font: inherit;
      border-radius: 7px;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      color: var(--text);
      padding: 0.42rem 0.7rem;
    }

    input[type="number"] {
      inline-size: 5.8rem;
    }

    button {
      cursor: pointer;
      font-weight: 600;
      transition: opacity 0.15s ease;
    }

    button:hover:not(:disabled) { opacity: 0.88; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }

    button.primary {
      background: var(--brand);
      color: var(--brand-text);
      border-color: var(--brand);
    }

    .check-label {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
    }

    /* Brand-consistent form fallback when accent-color is unavailable */
    @supports not (accent-color: #1d4ed8) {
      .check-label {
        position: relative;
        padding-inline-start: 1.6rem;
      }
      .check-label input[type="checkbox"] {
        position: absolute !important;
        clip-path: inset(50%) !important;
        overflow: hidden !important;
        width: 1px !important;
        height: 1px !important;
        margin: -1px !important;
        padding: 0 !important;
        border: 0 !important;
        white-space: nowrap !important;
      }
      .check-label input[type="checkbox"] + span::before {
        content: "";
        position: absolute;
        inset-inline-start: 0;
        inset-block-start: 50%;
        transform: translateY(-50%);
        inline-size: 1.1rem;
        block-size: 1.1rem;
        border: 2px solid var(--border);
        border-radius: 4px;
        background: var(--surface);
      }
      .check-label input[type="checkbox"]:checked + span::before {
        background: var(--brand);
        border-color: var(--brand);
      }
      .check-label input[type="checkbox"]:checked + span::after {
        content: "✓";
        position: absolute;
        inset-inline-start: 0.22rem;
        inset-block-start: 50%;
        transform: translateY(-50%);
        color: var(--brand-text);
        font-size: 0.8rem;
        font-weight: 700;
      }
      input[type="range"] {
        -webkit-appearance: none;
        appearance: none;
        background: transparent;
      }
      input[type="range"]::-webkit-slider-runnable-track {
        inline-size: 100%;
        block-size: 6px;
        border-radius: 4px;
        background: linear-gradient(to right, var(--brand) var(--progress, 25%), var(--border) var(--progress, 25%));
      }
      input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        inline-size: 15px;
        block-size: 15px;
        border-radius: 50%;
        background: var(--brand);
        margin-block-start: -4.5px;
      }
    }

    .status-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      font-size: 0.84rem;
      color: var(--muted);
    }

    .badge {
      display: inline-flex;
      align-items: center;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 600;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      color: var(--text);
    }

    .badge-good { color: var(--good); }
    .badge-warn { color: var(--warn); }

    .compare-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr));
      gap: 1rem;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      padding: 0.7rem 0.95rem;
      border-block-end: 1px solid var(--border);
      background: var(--surface-alt);
    }

    .panel-header h2 {
      margin: 0;
      font-size: 0.95rem;
      font-weight: 700;
    }

    .frame-box {
      background: #05070c;
      aspect-ratio: 16 / 9;
      display: grid;
      place-items: center;
      overflow: hidden;
    }

    .frame-box img {
      inline-size: 100%;
      block-size: 100%;
      object-fit: contain;
      display: block;
    }

    .panel-metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.5rem;
      padding: 0.7rem 0.95rem;
      border-block-start: 1px solid var(--border);
      font-size: 0.8rem;
    }

    .metric-cell strong {
      display: block;
      font-size: 1rem;
    }

    .metric-cell span {
      color: var(--muted);
      font-size: 0.74rem;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.9rem 1.05rem;
    }

    /* Responsive table (modern-web-guidance: sticky headers + @container stacked cards) */
    .table-wrapper {
      overflow: auto;
      max-inline-size: 100%;
      max-block-size: min(460px, 65vh);
      container-type: inline-size;
    }

    table {
      inline-size: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.85rem;
    }

    caption {
      text-align: start;
      font-weight: 700;
      font-size: 0.95rem;
      margin-block-end: 0.6rem;
    }

    thead {
      position: sticky;
      inset-block-start: 0;
      z-index: 1;
      background: var(--surface-alt);
    }

    th, td {
      padding: 0.5rem 0.75rem;
      border-block-end: 1px solid var(--border);
      text-align: start;
      white-space: nowrap;
    }

    tbody th {
      position: sticky;
      inset-inline-start: 0;
      background: var(--surface);
      font-weight: 600;
    }

    @container (width < 560px) {
      table, thead, tbody, tr, th, td { display: block; }
      thead { display: none; }
      tr {
        margin-block-end: 0.8rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
      }
      tbody th {
        position: sticky;
        inset-block-start: 0;
        z-index: 2;
        background: var(--surface-alt);
      }
      table {
        --label-2: "Stock YOLO26";
        --label-3: "Trained YOLO26";
        --label-4: "Delta";
      }
      td:nth-child(2)::before { content: var(--label-2) ": "; font-weight: 600; color: var(--muted); }
      td:nth-child(3)::before { content: var(--label-3) ": "; font-weight: 600; color: var(--muted); }
      td:nth-child(4)::before { content: var(--label-4) ": "; font-weight: 600; color: var(--muted); }
    }

    pre.report-box {
      margin: 0;
      padding: 0.85rem;
      background: var(--surface-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: auto;
      max-block-size: 420px;
      font-size: 0.8rem;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="topbar">
      <div class="title-group">
        <h1>VDAT Model Comparator — Stock YOLO26 vs Trained YOLO26</h1>
        <div class="subtitle" id="meta-line">Loading model info...</div>
      </div>
      <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;">
        <span class="badge" id="device-badge">Device: ...</span>
        <button type="button" id="btn-save">Save Side-by-Side PNG</button>
        <button type="button" id="btn-report">Toggle Report Log</button>
      </div>
    </header>

    <section class="controls" aria-label="Frame comparison controls">
      <div class="field">
        <label for="video-select">Video</label>
        <select id="video-select"></select>
      </div>

      <button type="button" class="primary" id="btn-random">Random Frame</button>

      <div class="field">
        <label for="frame-input">Frame #</label>
        <button type="button" id="btn-prev30">-30</button>
        <button type="button" id="btn-prev1">-1</button>
        <input type="number" id="frame-input" min="0" value="0">
        <button type="button" id="btn-next1">+1</button>
        <button type="button" id="btn-next30">+30</button>
        <button type="button" id="btn-go">Go</button>
      </div>

      <div class="field">
        <label for="conf-range">Conf: <span id="conf-val">0.30</span></label>
        <input type="range" id="conf-range" min="0.10" max="0.90" step="0.05" value="0.30">
      </div>

      <div class="field">
        <label for="imgsz-select">ImgSz</label>
        <select id="imgsz-select">
          <option value="640">640</option>
          <option value="864" selected>864</option>
          <option value="1280">1280</option>
        </select>
      </div>

      <label class="check-label" for="vehicle-only">
        <input type="checkbox" id="vehicle-only" checked>
        <span>Filter Stock to Vehicles Only</span>
      </label>
    </section>

    <div class="status-bar">
      <span id="frame-status">Ready.</span>
      <span id="agreement-status"></span>
    </div>

    <main class="compare-grid">
      <article class="panel">
        <div class="panel-header">
          <h2 id="stock-title">Stock YOLO26</h2>
          <span class="badge" id="stock-ms">-- ms</span>
        </div>
        <div class="frame-box">
          <img id="stock-img" alt="Stock YOLO26 detection output">
        </div>
        <div class="panel-metrics">
          <div class="metric-cell"><strong id="stock-count">-</strong><span>Detections</span></div>
          <div class="metric-cell"><strong id="stock-conf">-</strong><span>Mean Conf</span></div>
          <div class="metric-cell"><strong id="stock-small">-</strong><span>Small (&lt;32²)</span></div>
          <div class="metric-cell"><strong id="stock-hc">-</strong><span>Conf ≥ 0.50</span></div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-header">
          <h2 id="trained-title">Trained YOLO26</h2>
          <span class="badge" id="trained-ms">-- ms</span>
        </div>
        <div class="frame-box">
          <img id="trained-img" alt="Trained YOLO26 detection output">
        </div>
        <div class="panel-metrics">
          <div class="metric-cell"><strong id="trained-count">-</strong><span>Detections</span></div>
          <div class="metric-cell"><strong id="trained-conf">-</strong><span>Mean Conf</span></div>
          <div class="metric-cell"><strong id="trained-small">-</strong><span>Small (&lt;32²)</span></div>
          <div class="metric-cell"><strong id="trained-hc">-</strong><span>Conf ≥ 0.50</span></div>
        </div>
      </article>
    </main>

    <section class="card">
      <div tabindex="0" role="region" aria-labelledby="class-table-caption" class="table-wrapper">
        <table>
          <caption id="class-table-caption">Per-Class Frame Detection Breakdown</caption>
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col">Stock YOLO26</th>
              <th scope="col">Trained YOLO26</th>
              <th scope="col">Delta (Trained − Stock)</th>
            </tr>
          </thead>
          <tbody id="class-tbody"></tbody>
        </table>
      </div>
    </section>

    <section class="card" id="report-section" hidden>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-block-end:0.6rem;">
        <strong>Latest Benchmark Report (reports/comparison_report.md)</strong>
        <span class="subtitle">Run <code>python evaluate_compare.py</code> to regenerate</span>
      </div>
      <pre class="report-box" id="report-content">No report generated yet.</pre>
    </section>
  </div>

  <script>
    const els = {
      metaLine: document.getElementById('meta-line'),
      deviceBadge: document.getElementById('device-badge'),
      videoSelect: document.getElementById('video-select'),
      frameInput: document.getElementById('frame-input'),
      confRange: document.getElementById('conf-range'),
      confVal: document.getElementById('conf-val'),
      imgszSelect: document.getElementById('imgsz-select'),
      vehicleOnly: document.getElementById('vehicle-only'),
      frameStatus: document.getElementById('frame-status'),
      agreementStatus: document.getElementById('agreement-status'),
      stockTitle: document.getElementById('stock-title'),
      trainedTitle: document.getElementById('trained-title'),
      stockMs: document.getElementById('stock-ms'),
      trainedMs: document.getElementById('trained-ms'),
      stockImg: document.getElementById('stock-img'),
      trainedImg: document.getElementById('trained-img'),
      stockCount: document.getElementById('stock-count'),
      trainedCount: document.getElementById('trained-count'),
      stockConf: document.getElementById('stock-conf'),
      trainedConf: document.getElementById('trained-conf'),
      stockSmall: document.getElementById('stock-small'),
      trainedSmall: document.getElementById('trained-small'),
      stockHc: document.getElementById('stock-hc'),
      trainedHc: document.getElementById('trained-hc'),
      classTbody: document.getElementById('class-tbody'),
      reportSection: document.getElementById('report-section'),
      reportContent: document.getElementById('report-content'),
    };

    let videosMeta = {};
    let currentFrame = 0;

    function syncRangeFallback() {
      const min = parseFloat(els.confRange.min);
      const max = parseFloat(els.confRange.max);
      const val = parseFloat(els.confRange.value);
      const pct = ((val - min) / (max - min)) * 100;
      els.confRange.style.setProperty('--progress', pct.toFixed(1) + '%');
    }

    async function init() {
      const res = await fetch('/api/info');
      const info = await res.json();
      els.deviceBadge.textContent = info.device;
      els.stockTitle.textContent = 'Stock: ' + info.stock_name;
      els.trainedTitle.textContent = info.trained_available
        ? 'Trained: ' + info.trained_name
        : 'Trained: (Not found — run train.py)';
      els.metaLine.textContent = `Stock=${info.stock_model} | Trained=${info.trained_model || 'missing'}`;

      els.videoSelect.innerHTML = '';
      for (const v of info.videos) {
        videosMeta[v.name] = v;
        const opt = document.createElement('option');
        opt.value = v.name;
        opt.textContent = `${v.name} (${v.width}x${v.height}, ${v.frames} frames, ${v.duration_s}s)`;
        els.videoSelect.appendChild(opt);
      }
      syncRangeFallback();
      if (info.videos.length > 0) {
        await loadFrame('random');
      } else {
        els.frameStatus.textContent = 'No .mp4 videos found in training.VDAT.';
      }
    }

    async function loadFrame(frameArg) {
      const video = els.videoSelect.value;
      if (!video) return;
      const conf = els.confRange.value;
      const imgsz = els.imgszSelect.value;
      const vehicleOnly = els.vehicleOnly.checked ? '1' : '0';

      els.frameStatus.textContent = `Running inference on ${video} (frame=${frameArg})...`;
      const q = new URLSearchParams({ video, frame: String(frameArg), conf, imgsz, vehicle_only: vehicleOnly });
      const res = await fetch('/api/compare?' + q.toString());
      const data = await res.json();
      if (data.error) {
        els.frameStatus.textContent = 'Error: ' + data.error;
        return;
      }

      currentFrame = data.frame_index;
      els.frameInput.value = currentFrame;
      els.frameInput.max = Math.max(0, data.total_frames - 1);

      els.stockImg.src = 'data:image/jpeg;base64,' + data.stock.image_b64;
      els.trainedImg.src = 'data:image/jpeg;base64,' + data.trained.image_b64;

      els.stockMs.textContent = data.stock.infer_ms.toFixed(1) + ' ms';
      els.trainedMs.textContent = data.trained.available ? data.trained.infer_ms.toFixed(1) + ' ms' : 'N/A';

      els.stockCount.textContent = data.stock.count;
      els.trainedCount.textContent = data.trained.available ? data.trained.count : '-';
      els.stockConf.textContent = data.stock.mean_conf.toFixed(3);
      els.trainedConf.textContent = data.trained.available ? data.trained.mean_conf.toFixed(3) : '-';
      els.stockSmall.textContent = data.stock.small_count;
      els.trainedSmall.textContent = data.trained.available ? data.trained.small_count : '-';
      els.stockHc.textContent = data.stock.high_conf_count;
      els.trainedHc.textContent = data.trained.available ? data.trained.high_conf_count : '-';

      els.frameStatus.textContent =
        `${data.video} — Frame #${data.frame_index} / ${data.total_frames - 1} (${data.timestamp_s}s)`;
      if (data.agreement) {
        els.agreementStatus.textContent =
          `Agreement (IoU≥0.50): Matched=${data.agreement.matched_both} | Only Trained=${data.agreement.only_trained} | Only Stock=${data.agreement.only_stock}`;
      } else {
        els.agreementStatus.textContent = '';
      }

      const classes = Array.from(new Set([
        ...Object.keys(data.stock.class_counts || {}),
        ...Object.keys(data.trained.class_counts || {}),
      ])).sort();

      els.classTbody.innerHTML = '';
      if (classes.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<th scope="row">None</th><td>0</td><td>0</td><td>0</td>';
        els.classTbody.appendChild(tr);
      } else {
        for (const cls of classes) {
          const sc = (data.stock.class_counts || {})[cls] || 0;
          const tc = (data.trained.class_counts || {})[cls] || 0;
          const delta = tc - sc;
          const tr = document.createElement('tr');
          const dStr = delta > 0 ? '+' + delta : String(delta);
          tr.innerHTML = `<th scope="row">${cls}</th><td>${sc}</td><td>${tc}</td><td>${dStr}</td>`;
          els.classTbody.appendChild(tr);
        }
      }
    }

    function stepFrame(delta) {
      const meta = videosMeta[els.videoSelect.value];
      const maxF = meta ? Math.max(0, meta.frames - 1) : 0;
      const target = Math.max(0, Math.min(maxF, currentFrame + delta));
      loadFrame(target);
    }

    document.getElementById('btn-random').addEventListener('click', () => loadFrame('random'));
    document.getElementById('btn-prev30').addEventListener('click', () => stepFrame(-30));
    document.getElementById('btn-prev1').addEventListener('click', () => stepFrame(-1));
    document.getElementById('btn-next1').addEventListener('click', () => stepFrame(1));
    document.getElementById('btn-next30').addEventListener('click', () => stepFrame(30));
    document.getElementById('btn-go').addEventListener('click', () => loadFrame(parseInt(els.frameInput.value || '0', 10)));
    els.videoSelect.addEventListener('change', () => loadFrame('random'));
    els.imgszSelect.addEventListener('change', () => loadFrame(currentFrame));
    els.vehicleOnly.addEventListener('change', () => loadFrame(currentFrame));
    els.confRange.addEventListener('input', () => {
      els.confVal.textContent = parseFloat(els.confRange.value).toFixed(2);
      syncRangeFallback();
    });
    els.confRange.addEventListener('change', () => loadFrame(currentFrame));

    document.getElementById('btn-save').addEventListener('click', async () => {
      const res = await fetch('/api/save-snapshot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video: els.videoSelect.value,
          frame: currentFrame,
          conf: parseFloat(els.confRange.value),
          imgsz: parseInt(els.imgszSelect.value, 10),
          vehicle_only: els.vehicleOnly.checked,
        }),
      });
      const data = await res.json();
      els.frameStatus.textContent = data.saved_path
        ? `Saved snapshot -> ${data.saved_path}`
        : `Snapshot error: ${data.error || 'unknown'}`;
    });

    document.getElementById('btn-report').addEventListener('click', async () => {
      els.reportSection.hidden = !els.reportSection.hidden;
      if (!els.reportSection.hidden) {
        const res = await fetch('/api/report');
        const data = await res.json();
        els.reportContent.textContent = data.content || 'No report found yet. Run: python evaluate_compare.py';
      }
    });

    init();
  </script>
</body>
</html>
"""


class ComparatorState:
    def __init__(self, stock_path: str, trained_path: Optional[Path]):
        self.lock = threading.Lock()
        self.device_arg, self.device_desc = get_device_info()
        self.precision_kwargs = get_precision_kwargs(use_cuda=(self.device_arg != "cpu"))
        self.stock_path = stock_path
        self.trained_path = trained_path

        print(f"[CompareApp] Loading Stock model: {stock_path} ({self.device_desc})")
        self.stock_model = YOLO(stock_path)
        self.trained_model = None
        if trained_path and trained_path.exists():
            print(f"[CompareApp] Loading Trained model: {trained_path}")
            self.trained_model = YOLO(str(trained_path))
        else:
            print("[CompareApp] Trained weights not found yet — running with Stock only until train.py completes.")

        # Warm up
        dummy = np.zeros((480, 854, 3), dtype=np.uint8)
        run_model_detections(self.stock_model, dummy, conf=0.30, imgsz=864, vehicle_only_coco=True, precision_kwargs=self.precision_kwargs)
        if self.trained_model is not None:
            run_model_detections(self.trained_model, dummy, conf=0.30, imgsz=864, vehicle_only_coco=False, precision_kwargs=self.precision_kwargs)

        self.videos: Dict[str, Dict[str, Any]] = {}
        for vp in list_test_videos(ROOT_DIR):
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.videos[vp.name] = {
                "name": vp.name,
                "path": vp,
                "width": w,
                "height": h,
                "fps": round(fps, 2),
                "frames": frames,
                "duration_s": round(frames / fps, 1) if fps > 0 else 0.0,
            }

    def read_video_frame(self, video_name: str, frame_arg: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        meta = self.videos.get(video_name)
        if not meta:
            raise ValueError(f"Unknown video: {video_name}")
        total = max(1, meta["frames"])
        if frame_arg == "random":
            idx = random.randint(0, total - 1)
        else:
            idx = max(0, min(total - 1, int(frame_arg)))

        cap = cv2.VideoCapture(str(meta["path"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {video_name}")
        return frame, idx, meta

    def run_comparison(
        self,
        video_name: str,
        frame_arg: str,
        conf: float,
        imgsz: int,
        vehicle_only: bool,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        frame, idx, meta = self.read_video_frame(video_name, frame_arg)

        with self.lock:
            s_dets, s_ms = run_model_detections(
                self.stock_model,
                frame,
                conf=conf,
                imgsz=imgsz,
                vehicle_only_coco=vehicle_only,
                precision_kwargs=self.precision_kwargs,
            )
            if self.trained_model is not None:
                t_dets, t_ms = run_model_detections(
                    self.trained_model,
                    frame,
                    conf=conf,
                    imgsz=imgsz,
                    vehicle_only_coco=False,
                    precision_kwargs=self.precision_kwargs,
                )
            else:
                t_dets, t_ms = [], 0.0

        left = draw_detections_bgr(frame, s_dets, title=f"Stock YOLO26 | {len(s_dets)} det | {s_ms:.1f}ms")
        if self.trained_model is not None:
            right = draw_detections_bgr(frame, t_dets, title=f"Trained YOLO26 | {len(t_dets)} det | {t_ms:.1f}ms")
        else:
            right = draw_detections_bgr(frame, [], title="Trained YOLO26 not found (run python train.py)")

        def summarize(dets: List[Dict[str, Any]], ms: float, img_bgr: np.ndarray, available: bool) -> Dict[str, Any]:
            _, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            confs = [d["conf"] for d in dets]
            return {
                "available": available,
                "count": len(dets),
                "mean_conf": float(np.mean(confs)) if confs else 0.0,
                "small_count": sum(1 for d in dets if d["area"] < 32 * 32),
                "high_conf_count": sum(1 for c in confs if c >= 0.50),
                "class_counts": dict(Counter(d["canon_name"] for d in dets)),
                "infer_ms": ms,
                "image_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
            }

        agreement = None
        if self.trained_model is not None:
            m, os_n, ot_n, ious = match_between_models(s_dets, t_dets)
            agreement = {
                "matched_both": m,
                "only_stock": os_n,
                "only_trained": ot_n,
                "mean_pair_iou": float(np.mean(ious)) if ious else 0.0,
            }

        payload = {
            "video": video_name,
            "frame_index": idx,
            "total_frames": meta["frames"],
            "timestamp_s": round(idx / meta["fps"], 2) if meta["fps"] > 0 else 0.0,
            "stock": summarize(s_dets, s_ms, left, True),
            "trained": summarize(t_dets, t_ms, right, self.trained_model is not None),
            "agreement": agreement,
        }
        return payload, left, right


def make_handler(state: ComparatorState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return  # Keep console clean

        def _send_json(self, obj: Any, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/info":
                self._send_json({
                    "device": state.device_desc,
                    "stock_model": state.stock_path,
                    "stock_name": Path(state.stock_path).name,
                    "trained_model": str(state.trained_path) if state.trained_path else None,
                    "trained_name": state.trained_path.name if state.trained_path else None,
                    "trained_available": state.trained_model is not None,
                    "trained_classes": parse_names_dict(getattr(state.trained_model, "names", {})) if state.trained_model else {},
                    "videos": [
                        {k: v for k, v in meta.items() if k != "path"}
                        for meta in state.videos.values()
                    ],
                })
                return

            if parsed.path == "/api/compare":
                qs = urllib.parse.parse_qs(parsed.query)
                video = qs.get("video", [next(iter(state.videos.keys()), "")])[0]
                frame_arg = qs.get("frame", ["random"])[0]
                conf = float(qs.get("conf", ["0.30"])[0])
                imgsz = int(qs.get("imgsz", ["864"])[0])
                vehicle_only = qs.get("vehicle_only", ["1"])[0] not in ("0", "false", "False")
                try:
                    payload, _, _ = state.run_comparison(video, frame_arg, conf, imgsz, vehicle_only)
                    self._send_json(payload)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                return

            if parsed.path == "/api/report":
                rep = ROOT_DIR / "reports" / "comparison_report.md"
                content = rep.read_text(encoding="utf-8") if rep.exists() else ""
                self._send_json({"exists": rep.exists(), "path": str(rep), "content": content})
                return

            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/save-snapshot":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    req = json.loads(raw.decode("utf-8"))
                    video = req.get("video", "")
                    frame = str(req.get("frame", 0))
                    conf = float(req.get("conf", 0.30))
                    imgsz = int(req.get("imgsz", 864))
                    vehicle_only = bool(req.get("vehicle_only", True))
                    _, left, right = state.run_comparison(video, frame, conf, imgsz, vehicle_only)
                    combo = make_side_by_side_bgr(left, right)
                    snap_dir = ROOT_DIR / "reports" / "snapshots"
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_file = snap_dir / f"{Path(video).stem}_f{frame}_{ts}.png"
                    cv2.imwrite(str(out_file), combo)
                    self._send_json({"saved_path": str(out_file.relative_to(ROOT_DIR))})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                return

            self.send_response(404)
            self.end_headers()

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive side-by-side Stock vs Trained YOLO26 comparator")
    parser.add_argument("--stock", type=str, default="yolo26l.pt", help="Stock YOLO26 weights (default: yolo26l.pt)")
    parser.add_argument("--trained", type=str, default=None, help="Trained weights path (default: auto-find weights/best.pt)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port (default: 7860)")
    return parser.parse_args()


def main():
    args = parse_args()
    trained_path = resolve_trained_weights(args.trained)
    state = ComparatorState(args.stock, trained_path)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print("=" * 68)
    print(f"VDAT Side-by-Side Model Comparator running at: http://localhost:{args.port}")
    print(f"  If running on ThunderCompute, forward port:  ssh -L {args.port}:localhost:{args.port} tnr-0")
    print("=" * 68)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down comparator.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
