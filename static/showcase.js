(() => {
  const tabs = [...document.querySelectorAll('[role="tab"][data-path]')];
  const outputImage = document.getElementById('output-image');
  const inputImage = document.getElementById('input-image');
  const outputPanel = document.getElementById('output-panel');
  const outputName = document.getElementById('output-name');
  const outputSubtitle = document.getElementById('output-subtitle');
  const fileInput = document.getElementById('image-file');
  const processButton = document.getElementById('process-image');
  const restoreButton = document.getElementById('restore-demo');
  const status = document.getElementById('browser-status');
  const referenceTab = document.getElementById('path-reference');
  const localConsoleLink = document.getElementById('local-console-link');
  const setupLink = document.getElementById('setup-link');
  const metrics = {
    psnr: document.getElementById('metric-psnr'),
    ssim: document.getElementById('metric-ssim'),
    delta: document.getElementById('metric-delta-e'),
    lpips: document.getElementById('metric-lpips'),
    scope: document.getElementById('metric-scope'),
  };
  const benchmarkLabels = {
    classical: ['Classical output', 'Bilateral denoise · exposure and tone finish', '/static/showcase/classical.jpg'],
    learned: ['Learned output', 'SIDD-CC sRGB denoiser · tone finish', '/static/showcase/learned.jpg'],
    hybrid: ['Hybrid output', 'Classical and learned adaptive blend', '/static/showcase/hybrid.jpg'],
    reference: ['Aligned reference', 'Published clean benchmark target', '/static/showcase/reference.png'],
  };
  let labels = benchmarkLabels;
  let resultData = null;
  let customMetrics = false;
  let inferenceSession = null;
  const localApp = ['localhost', '127.0.0.1'].includes(location.hostname) && location.port === '5050';
  if (localApp) {
    localConsoleLink.hidden = false;
    setupLink.hidden = true;
  }

  function setMetrics(path) {
    if (customMetrics) {
      [metrics.psnr, metrics.ssim, metrics.delta, metrics.lpips].forEach(node => { node.textContent = '—'; });
      metrics.scope.textContent = 'No aligned reference was supplied; quality metrics are not available for this image.';
      return;
    }
    if (path === 'reference') {
      [metrics.psnr, metrics.ssim, metrics.delta, metrics.lpips].forEach(node => { node.textContent = '—'; });
      metrics.scope.textContent = 'Ground-truth reference for this aligned test block.';
      return;
    }
    const values = resultData?.paths?.[path]?.metrics;
    metrics.psnr.textContent = values?.psnr == null ? '—' : Number(values.psnr).toFixed(2);
    metrics.ssim.textContent = values?.ssim == null ? '—' : Number(values.ssim).toFixed(4);
    metrics.delta.textContent = values?.delta_e == null ? '—' : Number(values.delta_e).toFixed(2);
    metrics.lpips.textContent = values?.lpips == null ? '—' : Number(values.lpips).toFixed(4);
    metrics.scope.textContent = 'Scores are computed against the aligned reference for this one block.';
  }

  function activate(tab, moveFocus = false) {
    const path = tab.dataset.path;
    tabs.forEach(item => {
      const active = item === tab;
      item.setAttribute('aria-selected', String(active));
      item.tabIndex = active ? 0 : -1;
    });
    const [title, subtitle, image] = labels[path];
    outputPanel.setAttribute('aria-labelledby', tab.id);
    outputImage.src = image;
    outputImage.alt = title + ' image';
    outputName.textContent = title.toUpperCase();
    outputSubtitle.textContent = subtitle;
    setMetrics(path);
    if (moveFocus) tab.focus();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', event => {
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else return;
      event.preventDefault();
      activate(tabs[next], true);
    });
  });

  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    processButton.disabled = !file;
    if (!file) return;
    status.textContent = file.size > 30 * 1024 * 1024
      ? 'This file is larger than 30 MB. Choose a smaller image.'
      : file.name + ' · ' + (file.size / 1048576).toFixed(2) + ' MB · ready to process in this browser.';
  });
  processButton.addEventListener('click', processUploadedImage);
  restoreButton.addEventListener('click', restoreBenchmark);

  function percentile(values, fraction) {
    if (!values.length) return 0;
    values.sort((a, b) => a - b);
    return values[Math.min(values.length - 1, Math.floor((values.length - 1) * fraction))];
  }
  function clamp(value) { return Math.max(0, Math.min(1, value)); }
  function linear(value) { return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4; }
  function gamma(value) { return value <= 0.0031308 ? value * 12.92 : 1.055 * Math.max(value, 0) ** (1 / 2.4) - 0.055; }

  function toneMap(source, width, height, sharpen = false) {
    const count = width * height;
    const luminance = new Float32Array(count);
    const sample = [];
    for (let i = 0; i < count; i++) {
      const j = i * 3;
      luminance[i] = 0.2126 * linear(source[j]) + 0.7152 * linear(source[j + 1]) + 0.0722 * linear(source[j + 2]);
      if (i % 16 === 0) sample.push(luminance[i]);
    }
    const measured = Math.max(percentile(sample.slice(), 0.55), 1e-5);
    const gain = 2 ** Math.max(-1, Math.min(4.5, Math.log2(0.05 / measured)));
    const exposedY = sample.map(y => y * gain);
    const white = Math.max(percentile(exposedY, 0.997), 0.12);
    const out = new Float32Array(source.length);
    for (let i = 0; i < count; i++) {
      const j = i * 3;
      const y = luminance[i] * gain;
      const yd = clamp(y * (1 + y / (white * white)) / (1 + y));
      const ratio = yd / Math.max(y, 1e-6);
      for (let c = 0; c < 3; c++) out[j + c] = clamp(gamma(linear(source[j + c]) * gain * ratio));
    }
    if (!sharpen) return out;
    const sharp = new Float32Array(out.length);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      const up = (Math.max(0, y - 1) * width + x) * 3;
      const down = (Math.min(height - 1, y + 1) * width + x) * 3;
      const left = (y * width + Math.max(0, x - 1)) * 3;
      const right = (y * width + Math.min(width - 1, x + 1)) * 3;
      for (let c = 0; c < 3; c++) sharp[i + c] = clamp(out[i + c] + 0.18 * (out[i + c] - 0.25 * (out[up + c] + out[down + c] + out[left + c] + out[right + c])));
    }
    return sharp;
  }

  function classicalPath(source, width, height) {
    const tone = toneMap(source, width, height, false);
    const out = new Float32Array(tone.length);
    const spatial = [1, 2, 1];
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const center = (y * width + x) * 3;
      const centerY = 0.2126 * tone[center] + 0.7152 * tone[center + 1] + 0.0722 * tone[center + 2];
      let rSum = 0, gSum = 0, bSum = 0;
      let weights = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = Math.max(0, Math.min(height - 1, y + dy));
        const xx = Math.max(0, Math.min(width - 1, x + dx));
        const pos = (yy * width + xx) * 3;
        const neighborY = 0.2126 * tone[pos] + 0.7152 * tone[pos + 1] + 0.0722 * tone[pos + 2];
        const diff = neighborY - centerY;
        const w = spatial[dy + 1] * spatial[dx + 1] * Math.exp(-(diff * diff) / 0.0128);
        weights += w;
        rSum += tone[pos] * w;
        gSum += tone[pos + 1] * w;
        bSum += tone[pos + 2] * w;
      }
      out[center] = clamp(tone[center] * 0.72 + (rSum / weights) * 0.28);
      out[center + 1] = clamp(tone[center + 1] * 0.72 + (gSum / weights) * 0.28);
      out[center + 2] = clamp(tone[center + 2] * 0.72 + (bSum / weights) * 0.28);
    }
    const sharp = new Float32Array(out.length);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      const up = (Math.max(0, y - 1) * width + x) * 3;
      const down = (Math.min(height - 1, y + 1) * width + x) * 3;
      const left = (y * width + Math.max(0, x - 1)) * 3;
      const right = (y * width + Math.min(width - 1, x + 1)) * 3;
      for (let c = 0; c < 3; c++) sharp[i + c] = clamp(out[i + c] + 0.16 * (out[i + c] - 0.25 * (out[up + c] + out[down + c] + out[left + c] + out[right + c])));
    }
    return sharp;
  }

  function tileStarts(length, tile = 256, overlap = 32) {
    if (length <= tile) return [0];
    const starts = [];
    for (let p = 0; p < length - tile; p += tile - overlap) starts.push(p);
    const last = length - tile;
    if (starts[starts.length - 1] !== last) starts.push(last);
    return starts;
  }
  function windowWeight(local, start, extent, full, overlap = 32) {
    let value = 1;
    if (start > 0 && local < overlap) value *= Math.sin(((local + 0.5) / overlap) * Math.PI / 2) ** 2;
    if (start + extent < full && local >= extent - overlap) value *= Math.sin(((extent - local - 0.5) / overlap) * Math.PI / 2) ** 2;
    return value;
  }
  function tileNoise(source, width, y, x, height, extent) {
    let total = 0, count = 0;
    const step = 4;
    const luma = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
    for (let yy = 1; yy < height - 1; yy += step) for (let xx = 1; xx < extent - 1; xx += step) {
      const at = ((y + yy) * width + x + xx) * 3;
      const here = luma(source[at], source[at + 1], source[at + 2]);
      const left = at - 3, right = at + 3, up = at - width * 3, down = at + width * 3;
      const blur = (luma(source[left], source[left + 1], source[left + 2]) + luma(source[right], source[right + 1], source[right + 2]) + luma(source[up], source[up + 1], source[up + 2]) + luma(source[down], source[down + 1], source[down + 2])) / 4;
      total += Math.abs(here - blur); count++;
    }
    return count ? total / count : 0;
  }

  async function getSession() {
    if (inferenceSession) return inferenceSession;
    if (!window.ort) throw new Error('The browser inference runtime could not be loaded. Check the connection and reload this page.');
    window.ort.env.wasm.numThreads = 1;
    window.ort.env.wasm.wasmPaths = {
      mjs: new URL('/static/vendor/ort/ort-wasm-simd-threaded.mjs', location.href).href,
      wasm: new URL('/static/vendor/ort/ort-wasm-simd-threaded.wasm', location.href).href,
    };
    inferenceSession = await window.ort.InferenceSession.create('/static/showcase/nafnet-srgb.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return inferenceSession;
  }

  async function learnedPath(source, width, height, progress) {
    const session = await getSession();
    const ys = tileStarts(height), xs = tileStarts(width);
    const tiles = [];
    ys.forEach(y => xs.forEach(x => {
      const h = Math.min(256, height - y), w = Math.min(256, width - x);
      tiles.push({ y, x, h, w, score: tileNoise(source, width, y, x, h, w) });
    }));
    const accumulated = new Float32Array(source.length);
    const weights = new Float32Array(width * height);
    for (let index = 0; index < tiles.length; index++) {
      const tile = tiles[index];
      const input = new Float32Array(3 * 256 * 256);
      for (let py = 0; py < 256; py++) {
        const sy = Math.min(height - 1, tile.y + py);
        for (let px = 0; px < 256; px++) {
          const sx = Math.min(width - 1, tile.x + px);
          const sourceAt = (sy * width + sx) * 3;
          const at = py * 256 + px;
          input[at] = source[sourceAt];
          input[256 * 256 + at] = source[sourceAt + 1];
          input[2 * 256 * 256 + at] = source[sourceAt + 2];
        }
      }
      const tensor = new window.ort.Tensor('float32', input, [1, 3, 256, 256]);
      const output = (await session.run({ srgb: tensor })).restored_srgb.data;
      for (let py = 0; py < tile.h; py++) {
        const wy = windowWeight(py, tile.y, tile.h, height);
        for (let px = 0; px < tile.w; px++) {
          const weight = wy * windowWeight(px, tile.x, tile.w, width);
          const dest = (tile.y + py) * width + tile.x + px;
          const tileAt = py * 256 + px;
          weights[dest] += weight;
          for (let c = 0; c < 3; c++) accumulated[dest * 3 + c] += clamp(output[c * 256 * 256 + tileAt]) * weight;
        }
      }
      progress(index + 1, tiles.length);
    }
    const restored = new Float32Array(source.length);
    for (let i = 0; i < weights.length; i++) {
      const scale = 1 / Math.max(weights[i], 1e-8);
      restored[i * 3] = accumulated[i * 3] * scale;
      restored[i * 3 + 1] = accumulated[i * 3 + 1] * scale;
      restored[i * 3 + 2] = accumulated[i * 3 + 2] * scale;
    }
    return { image: toneMap(restored, width, height, true), tiles };
  }

  function hybridPath(classical, learned, tiles, width, height) {
    const selected = new Set(tiles.slice().sort((a, b) => b.score - a.score).slice(0, Math.ceil(tiles.length * 0.5)));
    const alphaSum = new Float32Array(width * height);
    const weightSum = new Float32Array(width * height);
    for (const tile of tiles) {
      for (let py = 0; py < tile.h; py++) for (let px = 0; px < tile.w; px++) {
        const weight = windowWeight(py, tile.y, tile.h, height) * windowWeight(px, tile.x, tile.w, width);
        const i = (tile.y + py) * width + tile.x + px;
        weightSum[i] += weight;
        if (selected.has(tile)) alphaSum[i] += weight;
      }
    }
    const output = new Float32Array(classical.length);
    for (let i = 0; i < weightSum.length; i++) {
      const alpha = weightSum[i] ? alphaSum[i] / weightSum[i] : 0;
      for (let c = 0; c < 3; c++) output[i * 3 + c] = classical[i * 3 + c] * (1 - alpha) + learned[i * 3 + c] * alpha;
    }
    return { image: output, routed: Math.round(100 * selected.size / Math.max(tiles.length, 1)) };
  }

  function asDataUrl(pixels, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d', { alpha: false });
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      rgba[i * 4] = Math.round(clamp(pixels[i * 3]) * 255);
      rgba[i * 4 + 1] = Math.round(clamp(pixels[i * 3 + 1]) * 255);
      rgba[i * 4 + 2] = Math.round(clamp(pixels[i * 3 + 2]) * 255);
      rgba[i * 4 + 3] = 255;
    }
    ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
    return canvas.toDataURL('image/jpeg', 0.94);
  }

  async function processUploadedImage() {
    const file = fileInput.files?.[0];
    if (!file) return;
    if (file.size > 30 * 1024 * 1024) {
      status.textContent = 'This file is larger than 30 MB. Choose a smaller image.';
      return;
    }
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      status.textContent = 'Choose a JPEG, PNG or WebP image. Camera RAW is supported in the local app.';
      return;
    }
    processButton.disabled = true;
    fileInput.disabled = true;
    restoreButton.disabled = true;
    status.textContent = 'Decoding image locally…';
    try {
      const bitmap = await createImageBitmap(file);
      if (bitmap.width * bitmap.height > 50_000_000) {
        bitmap.close();
        throw new Error('This image has more than 50 megapixels. Choose a smaller export.');
      }
      const scale = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
      const width = Math.max(1, Math.round(bitmap.width * scale));
      const height = Math.max(1, Math.round(bitmap.height * scale));
      const canvas = document.createElement('canvas');
      canvas.width = width; canvas.height = height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(bitmap, 0, 0, width, height);
      bitmap.close();
      const rgba = ctx.getImageData(0, 0, width, height).data;
      const source = new Float32Array(width * height * 3);
      for (let i = 0; i < width * height; i++) {
        source[i * 3] = rgba[i * 4] / 255;
        source[i * 3 + 1] = rgba[i * 4 + 1] / 255;
        source[i * 3 + 2] = rgba[i * 4 + 2] / 255;
      }
      status.textContent = 'Running the classical path…';
      await new Promise(resolve => requestAnimationFrame(resolve));
      const classical = classicalPath(source, width, height);
      status.textContent = 'Loading the restoration model in this browser…';
      await new Promise(resolve => requestAnimationFrame(resolve));
      const start = performance.now();
      const result = await learnedPath(source, width, height, (done, total) => {
        status.textContent = 'Running learned restoration · tile ' + done + ' of ' + total + '…';
      });
      const learnedElapsed = performance.now() - start;
      status.textContent = 'Composing the hybrid comparison…';
      const hybrid = hybridPath(classical, result.image, result.tiles, width, height);
      inputImage.src = canvas.toDataURL('image/jpeg', 0.94);
      inputImage.alt = 'Uploaded RGB image resized to ' + width + ' by ' + height + ' pixels';
      document.getElementById('context-title').textContent = 'Uploaded sRGB input';
      document.getElementById('context-detail').textContent = 'Processed in this browser';
      document.getElementById('context-size').textContent = width + ' × ' + height + ' px';
      document.getElementById('comparison-context').textContent = 'User-supplied image · rendered sRGB processing';
      document.getElementById('input-subtitle').textContent = file.type.replace('image/', '').toUpperCase() + ' · local input';
      labels = {
        classical: ['Classical output', 'Exposure · bilateral filter · tone and detail', asDataUrl(classical, width, height)],
        learned: ['Learned output', 'SIDD-CC sRGB model · ' + learnedElapsed.toFixed(0) + ' ms inference', asDataUrl(result.image, width, height)],
        hybrid: ['Hybrid output', 'Adaptive tile blend · ' + hybrid.routed + '% selected · no compute savings', asDataUrl(hybrid.image, width, height)],
        reference: benchmarkLabels.reference,
      };
      customMetrics = true;
      referenceTab.hidden = true;
      document.getElementById('integrity-note').innerHTML = '<strong>Scope:</strong> trained for SIDD-CC smartphone sRGB denoising. Results may vary on other image sources. Hybrid is composed after full learned inference and does not save compute. No aligned reference was supplied, so no quality metrics are shown.';
      activate(tabs.find(tab => tab.dataset.path === 'learned'));
      status.textContent = 'Processed ' + width + ' × ' + height + ' sRGB in ' + (performance.now() - start).toFixed(0) + ' ms. Learned inference ' + learnedElapsed.toFixed(0) + ' ms · ' + result.tiles.length + ' tiles · ' + hybrid.routed + '% routed in Hybrid. Image remained in this browser.';
    } catch (error) {
      console.error(error);
      status.textContent = error?.message || 'Could not process this image in the browser. Try a smaller JPEG, PNG or WebP.';
    } finally {
      fileInput.disabled = false;
      restoreButton.disabled = false;
      processButton.disabled = !fileInput.files?.[0];
    }
  }

  function restoreBenchmark() {
    customMetrics = false;
    labels = benchmarkLabels;
    inputImage.src = '/static/showcase/input.jpg';
    inputImage.alt = 'Noisy 256 by 256 SIDD-CC test image';
    document.getElementById('context-title').textContent = 'SIDD-CC sRGB test';
    document.getElementById('context-detail').textContent = 'Scene 0164 · block 1 of 928';
    document.getElementById('context-size').textContent = 'Aligned 256 × 256 crop';
    document.getElementById('comparison-context').textContent = 'Fixed held-out test example · rendered sRGB denoising';
    document.getElementById('input-subtitle').textContent = 'Noisy sRGB';
    document.getElementById('integrity-note').innerHTML = '<strong>Scope:</strong> these are measurements for one held-out block after the workbench finishing path. They are not the 928-block benchmark average. Hybrid reuses Learned predictions for this RGB comparison and does not claim compute savings.';
    referenceTab.hidden = false;
    fileInput.value = '';
    processButton.disabled = true;
    activate(tabs.find(tab => tab.dataset.path === 'learned'));
    status.textContent = 'Benchmark sample restored. Choose an RGB image to run the model in this browser; no image is uploaded.';
  }

  fetch('/static/showcase/results.json')
    .then(response => {
      if (!response.ok) throw new Error('Showcase measurements could not be loaded.');
      return response.json();
    })
    .then(data => {
      resultData = data;
      setMetrics(document.querySelector('[role="tab"][aria-selected="true"]')?.dataset.path || 'learned');
    })
    .catch(() => { metrics.scope.textContent = 'Images are available; this browser could not load the measurement record.'; });
})();
