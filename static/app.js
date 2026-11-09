const $ = selector => document.querySelector(selector);
let mode = 'human';
let data = null;
let view = 'hybrid';
let inputPreviewUrl = null;

const image = $('#image');
const reference = $('#reference');
const run = $('#run');
const demoRun = $('#demo-run');
const rows = [...document.querySelectorAll('.method-row')];

function clearOutput() {
  data = null;
  $('#raw-integrity').hidden = true;
  $('#output-img').removeAttribute('src');
  $('#output-frame').classList.remove('has-image');
  $('#empty-output').hidden = false;
  $('#download').classList.add('disabled');
  $('#download').setAttribute('aria-disabled', 'true');
  $('#source-summary').textContent = 'Select an image to begin.';
  $('#dimensions').textContent = '—';
  $('#selected-method').textContent = '—';
  $('#latency').textContent = '—';
  $('#memory').textContent = '—';
  $('#parameters').textContent = '—';
  $('#model-size').textContent = '—';
  $('#quality-state').textContent = 'Reference required';
  ['psnr', 'ssim', 'delta-e', 'lpips'].forEach(id => $(`#${id}`).textContent = '—');
  $('#path-status').textContent = 'Run a process to compare';
  $('#measurement-state').textContent = 'Actual local execution';
  rows.forEach(row => {
    row.disabled = true;
    row.setAttribute('aria-selected', 'false');
    row.querySelector('[data-runtime]').textContent = '—';
    row.querySelector('[data-state]').textContent = 'Waiting';
  });
}

image.addEventListener('change', () => {
  const file = image.files[0];
  $('#file-name').textContent = file?.name || 'Choose a file';
  run.disabled = !file;
  $('#error').textContent = '';
  clearOutput();
  if (!file) {
    $('#input-img').removeAttribute('src');
    $('#input-frame').classList.remove('has-image');
    $('#empty-input').hidden = false;
    $('#input-meta').textContent = 'Waiting for image';
    return;
  }
  if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
  inputPreviewUrl = URL.createObjectURL(file);
  $('#input-img').src = inputPreviewUrl;
  $('#input-frame').classList.add('has-image');
  $('#empty-input').hidden = true;
  $('#input-meta').textContent = `${file.type || 'Image file'} · ready to process`;
  const probe = new Image();
  probe.onload = () => {
    if (image.files[0] === file) {
      $('#input-meta').textContent = `${probe.naturalWidth} × ${probe.naturalHeight} · ${file.type || 'Image file'}`;
      $('#source-summary').textContent = `${file.name} · ${probe.naturalWidth} × ${probe.naturalHeight}`;
    }
  };
  probe.src = inputPreviewUrl;
});

reference.addEventListener('change', () => {
  $('#ref-name').textContent = reference.files[0]?.name || 'Optional';
});

const drop = $('#drop');
['dragenter', 'dragover'].forEach(type => drop.addEventListener(type, event => {
  event.preventDefault();
  drop.classList.add('drag-active');
}));
['dragleave', 'drop'].forEach(type => drop.addEventListener(type, event => {
  event.preventDefault();
  drop.classList.remove('drag-active');
}));
drop.addEventListener('drop', event => {
  const files = event.dataTransfer?.files;
  if (!files?.length) return;
  image.files = files;
  image.dispatchEvent(new Event('change', { bubbles: true }));
});

document.querySelectorAll('.mode').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.mode').forEach(option => {
    const selected = option === button;
    option.classList.toggle('active', selected);
    option.setAttribute('aria-pressed', String(selected));
  });
  mode = button.dataset.mode;
}));

$('#strength').addEventListener('input', event => {
  $('#strength-value').textContent = `${Number(event.target.value).toFixed(1)}×`;
});

function setMetric(id, value, suffix = '') {
  $(`#${id}`).textContent = value == null ? '—' : `${value}${suffix}`;
}

function describePath(name, output) {
  if (name === 'classical') return 'Classical baseline';
  if (data.camera_managed_raw) {
    return name === 'hybrid' ? `Camera RAW · ${output.routed}% sRGB tiles routed` : 'Camera RAW rendered by LibRaw, then SIDD-CC sRGB denoising';
  }
  if (data.input_type === 'rgb') {
    return name === 'hybrid' ? `RGB hybrid · ${output.routed}% routed` : 'SIDD-CC sRGB denoiser';
  }
  return name === 'hybrid' ? `Bayer hybrid · ${output.routed}% routed` : 'Bayer restoration model';
}

function render() {
  if (!data) return;
  const output = data.outputs[view];
  if (!output) return;
  const raw = data.raw_integrity;
  const rawPanel = $('#raw-integrity');
  rawPanel.hidden = !raw;
  if (raw) {
    const values = {
      'raw-camera': raw.camera || 'Not reported',
      'raw-cfa': raw.cfa || 'Not reported',
      'raw-black': Array.isArray(raw.black_level_rggb) ? raw.black_level_rggb.join(' / ') : (raw.black_level_rggb ?? 'Not reported'),
      'raw-white': raw.white_level ?? 'Not reported',
      'raw-wb': Array.isArray(raw.white_balance_rggb) ? raw.white_balance_rggb.map(value => Number(value).toFixed(3)).join(' / ') : (raw.white_balance_rggb ?? 'Not reported'),
      'raw-matrix': raw.color_matrix || 'Not reported',
      'raw-renderer': raw.renderer || 'Not reported',
      'raw-shadow-clip': `${Number(raw.shadow_clip_percent || 0).toFixed(3)}%`,
      'raw-highlight-clip': `${Number(raw.highlight_clip_percent || 0).toFixed(3)}%`,
      'raw-noise': raw.noise_assessment || 'Not calibrated'
    };
    Object.entries(values).forEach(([id, value]) => { $(`#${id}`).textContent = value; });
    $('#raw-integrity-status').textContent = raw.metadata_status || 'RAW metadata parsed';
    const clipping = Math.max(Number(raw.shadow_clip_percent || 0), Number(raw.highlight_clip_percent || 0));
    $('#raw-integrity-note').textContent = `${clipping >= 1 ? 'Clipping check: at least 1% of sensor samples meet a black/white boundary.' : 'Clipping check: less than 1% of sensor samples meet either boundary.'} This is a boundary-count heuristic; noise is not camera-calibrated.`;
  }
  const outputFrame = $('#output-frame');
  outputFrame.classList.add('is-changing');
  window.setTimeout(() => outputFrame.classList.remove('is-changing'), 220);
  $('#input-img').src = data.input;
  $('#input-frame').classList.add('has-image');
  $('#empty-input').hidden = true;
  $('#output-img').src = output.image;
  $('#output-frame').classList.add('has-image');
  $('#empty-output').hidden = true;
  const routeName = data.camera_managed_raw ? 'CAMERA RAW · RAW→RGB' : data.input_type === 'rgb' ? 'RENDERED RGB · S-RGB' : 'BAYER SENSOR';
  $('#output-label').textContent = `${view.toUpperCase()} · ${routeName}`;
  $('#download').href = output.image;
  $('#download').download = `mod-isp-${view}.jpg`;
  $('#download').classList.remove('disabled');
  $('#download').setAttribute('aria-disabled', 'false');
  $('#selected-method').textContent = `${view[0].toUpperCase()}${view.slice(1)}`;
  $('#latency').textContent = `${output.latency_ms} ms`;
  $('#memory').textContent = `${output.memory_mb} MB`;
  $('#memory').title = output.reused_inference
    ? 'Incremental Python allocations; shared learned tile predictions are not attributed again, and native/accelerator memory is excluded.'
    : 'Python allocations; native/accelerator memory is excluded.';
  const isModelPath = view !== 'classical';
  $('#parameters').textContent = isModelPath && data.model.available ? Number(data.model.parameters).toLocaleString() : '—';
  $('#model-size').textContent = isModelPath && data.model.available ? `${data.model.size_kb} KB` : '—';
  const metrics = output.metrics;
  setMetric('psnr', metrics?.psnr, typeof metrics?.psnr === 'number' || metrics?.psnr === '∞' ? ' dB' : '');
  setMetric('ssim', metrics?.ssim);
  setMetric('delta-e', metrics?.delta_e);
  setMetric('lpips', metrics?.lpips);
  $('#quality-state').textContent = data.demo ? 'Single held-out test block' : metrics ? 'Reference attached' : 'Reference required';
  $('#source-summary').textContent = `${data.input_kind} · ${data.processing_note || 'Standard pipeline'} · ${data.environment}`;
  $('#dimensions').textContent = `${data.width} × ${data.height}`;
  $('#input-meta').textContent = `${data.width} × ${data.height} · ${data.camera_managed_raw ? 'Camera RAW' : data.input_type === 'rgb' ? 'Rendered RGB' : 'Bayer sensor'}`;
  $('#path-status').textContent = 'Select a path to update the output';
  $('#measurement-state').textContent = 'Measured on this machine';
  const qualityNote = metrics
    ? 'Quality scores compare this output with the supplied reference.'
    : 'Attach a matching reference to calculate PSNR, SSIM, Delta E, and LPIPS.';
  const reuseNote = output.reused_inference
    ? ' Hybrid time and Python allocation are incremental comparison costs: selected tiles reuse Learned-path predictions, so these are not standalone Hybrid costs.'
    : '';
  const gateNote = output.gate
    ? ` Hybrid routing: ${output.gate.status}.${output.gate.validation_route_percent != null ? ` On the SIDD-CC validation scene it invoked the Bayer model on ${output.gate.validation_route_percent}% of tiles (${Number(output.gate.validation_tile_call_reduction_percent).toFixed(1)}% fewer tile calls) and changed mean PSNR by ${Number(output.gate.validation_psnr_delta_db).toFixed(2)} dB versus all-Learned.` : ''}`
    : '';
  const demoNote = data.demo ? ` ${data.demo.note}` : '';
  $('#measurement-note').textContent = `${describePath(view, output)}.${reuseNote}${gateNote} Memory excludes native and accelerator allocations. ${qualityNote}${demoNote}`;
  rows.forEach(row => {
    const name = row.dataset.view;
    const result = data.outputs[name];
    row.disabled = !result;
    row.setAttribute('aria-selected', String(name === view));
    row.querySelector('[data-runtime]').textContent = result ? `${result.latency_ms} ms` : 'Unavailable';
    row.querySelector('[data-state]').textContent = result ? (name === 'classical' ? 'Baseline' : `${result.routed != null ? `${result.routed}% routed` : 'Available'}${result.reused_inference ? ' · shared tiles' : ''}${result.gate?.deployment_eligible === false ? ' · experimental' : ''}`) : 'Model unavailable';
    const description = row.querySelector('[data-description]');
    if (description && (data.input_type === 'rgb' || data.camera_managed_raw)) description.textContent = name === 'hybrid' ? 'Variance-gated sRGB denoising' : 'SIDD-CC sRGB denoising';
    if (description && data.camera_managed_raw) description.textContent = name === 'hybrid' ? 'Variance-gated sRGB denoising after LibRaw rendering' : 'LibRaw camera render + sRGB denoising';
    else if (description && data.input_type === 'bayer') description.textContent = name === 'hybrid' ? 'RAW-statistics tile routing · experimental' : 'Joint Bayer restoration';
  });
}

rows.forEach(row => row.addEventListener('click', () => {
  if (row.disabled || !data?.outputs[row.dataset.view]) return;
  view = row.dataset.view;
  render();
}));

rows.forEach(row => row.addEventListener('keydown', event => {
  if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
  const available = rows.filter(item => !item.disabled);
  if (!available.length) return;
  event.preventDefault();
  const current = available.indexOf(row);
  const nextIndex = event.key === 'Home' ? 0 : event.key === 'End' ? available.length - 1 : (current + (event.key === 'ArrowDown' ? 1 : -1) + available.length) % available.length;
  available[nextIndex].focus();
  available[nextIndex].click();
}));

async function processCurrentImage() {
  if (!image.files.length) return;
  const form = new FormData();
  form.append('image', image.files[0]);
  if (reference.files[0]) form.append('reference', reference.files[0]);
  form.append('mode', mode);
  form.append('style', $('#style').value);
  form.append('strength', $('#strength').value);
  run.disabled = true;
  run.setAttribute('aria-busy', 'true');
  run.innerHTML = '<span>Processing image</span><span aria-hidden="true">↻</span>';
  $('#error').textContent = '';
  try {
    const response = await fetch('/api/process', { method: 'POST', body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Processing failed. Check the input and try again.');
    data = result;
    view = result.outputs.hybrid ? 'hybrid' : 'classical';
    $('#empty-output').hidden = false;
    render();
  } catch (error) {
    $('#error').textContent = error.message;
  } finally {
    run.disabled = !image.files.length;
    run.removeAttribute('aria-busy');
    run.innerHTML = '<span>Process image</span><span aria-hidden="true">↗</span>';
  }
}

run.addEventListener('click', processCurrentImage);

demoRun.addEventListener('click', async () => {
  demoRun.disabled = true;
  demoRun.setAttribute('aria-busy', 'true');
  demoRun.innerHTML = '<span>Running benchmark example</span><b aria-hidden="true">↻</b>';
  run.disabled = true;
  $('#error').textContent = '';
  try {
    const response = await fetch('/api/demo');
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Benchmark example could not be loaded.');
    image.value = '';
    reference.value = '';
    $('#file-name').textContent = 'SIDD-CC test · scene 0164 · block 1';
    $('#ref-name').textContent = 'Published aligned clean reference';
    data = result;
    view = result.outputs.hybrid ? 'hybrid' : 'classical';
    render();
    $('#comparison').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    $('#error').textContent = error.message;
  } finally {
    run.disabled = !image.files.length;
    demoRun.disabled = false;
    demoRun.removeAttribute('aria-busy');
    demoRun.innerHTML = '<span>Run held-out SIDD-CC example</span><b aria-hidden="true">↗</b>';
  }
});

fetch('/api/status').then(response => response.json()).then(status => {
  const ready = [status.rgb?.available, status.bayer?.available].filter(Boolean).length;
  $('#model-status').textContent = ready === 2 ? 'Learned models ready' : ready === 1 ? 'One learned model ready' : 'Classical processing ready';
  if (ready === 0) $('.system-status').classList.add('error');
}).catch(() => {
  $('#model-status').textContent = 'Local processing service unavailable';
  $('.system-status').classList.add('error');
});
