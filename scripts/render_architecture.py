"""Render the repository architecture diagram to docs/architecture.png."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "architecture.png"
W, H, SCALE = 1800, 1630, 2
img = Image.new("RGB", (W * SCALE, H * SCALE), "#f7f8fa")
draw = ImageDraw.Draw(img)
font_path = ROOT / "static" / "fonts" / "manrope-variable.ttf"
mono_path = ROOT / "static" / "fonts" / "dm-mono-regular.ttf"

def font(size, mono=False):
    return ImageFont.truetype(str(mono_path if mono else font_path), size * SCALE)

def box(x, y, w, h, title, body, tint="#ffffff", accent="#2056ed"):
    bounds = tuple(v * SCALE for v in (x, y, x + w, y + h))
    draw.rounded_rectangle(bounds, radius=14 * SCALE, fill=tint, outline="#dfe3e8", width=2 * SCALE)
    draw.rounded_rectangle((bounds[0], bounds[1], bounds[0] + 7 * SCALE, bounds[3]), radius=3 * SCALE, fill=accent)
    draw.text(((x + 22) * SCALE, (y + 17) * SCALE), title, font=font(21), fill="#17191e")
    lines = body.split("\n")
    for i, line in enumerate(lines):
        draw.text(((x + 22) * SCALE, (y + 54 + i * 27) * SCALE), line, font=font(14), fill="#626873")

def arrow(points, color="#738094", width=3):
    pts = [(x * SCALE, y * SCALE) for x, y in points]
    draw.line(pts, fill=color, width=width * SCALE, joint="curve")
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 12 * SCALE
    left = (x2 - length * math.cos(angle - 0.48), y2 - length * math.sin(angle - 0.48))
    right = (x2 - length * math.cos(angle + 0.48), y2 - length * math.sin(angle + 0.48))
    draw.polygon([(x2, y2), left, right], fill=color)

def label(x, y, text):
    draw.text((x * SCALE, y * SCALE), text.upper(), font=font(11, mono=True), fill="#687386")

# Header
draw.text((72 * SCALE, 42 * SCALE), "MOD-ISP  /  SYSTEM ARCHITECTURE", font=font(16, mono=True), fill="#2056ed")
draw.text((72 * SCALE, 72 * SCALE), "Two image domains, one local workbench", font=font(34), fill="#17191e")
draw.text((72 * SCALE, 123 * SCALE), "Camera RAW and rendered sRGB follow separate restoration paths. Results return to the local browser.", font=font(16), fill="#626873")

# Request flow
label(80, 184, "01  /  Local request")
box(80, 210, 530, 112, "Browser workbench", "Select image · profile · optional aligned reference", "#ffffff", "#2056ed")
box(1190, 210, 530, 112, "Flask API · /api/process", "Decode → route by input domain → encode available outputs", "#ffffff", "#17191e")
arrow([(610, 266), (1190, 266)], "#2056ed")
label(800, 235, "multipart upload")

# Input domain
box(650, 342, 500, 68, "Input decoder and router", "Validate file type · preserve RAW / RGB / Bayer semantics", "#ffffff", "#17191e")
label(80, 438, "02  /  Input domain")
box(80, 464, 500, 132, "Camera RAW", "LibRaw / rawpy render with camera metadata\nThen enter the rendered sRGB path", "#eef3ff", "#2056ed")
box(650, 464, 500, 132, "Rendered RGB / sRGB", "JPEG · PNG · WebP · TIFF\nPreserve RGB; do not convert to Bayer", "#ffffff", "#737d8d")
box(1220, 464, 500, 132, "Normalized Bayer NPY", "Finite RGGB mosaic · black-subtracted + WB\nSensor-white normalized · keeps sub-black/headroom", "#eef3ff", "#2056ed")
arrow([(1455, 322), (1455, 330), (900, 330), (900, 342)])
arrow([(900, 410), (900, 438), (330, 438), (330, 464)])
arrow([(900, 410), (900, 464)])
arrow([(900, 410), (900, 438), (1470, 438), (1470, 464)])

# Domain groups
label(80, 640, "03  /  Restoration paths")
draw.rounded_rectangle((80 * SCALE, 666 * SCALE, 865 * SCALE, 1072 * SCALE), radius=18 * SCALE, fill="#ffffff", outline="#dfe3e8", width=2 * SCALE)
draw.rounded_rectangle((935 * SCALE, 666 * SCALE, 1720 * SCALE, 1072 * SCALE), radius=18 * SCALE, fill="#ffffff", outline="#dfe3e8", width=2 * SCALE)
draw.text((106 * SCALE, 685 * SCALE), "RENDERED RGB DOMAIN", font=font(12, True), fill="#2056ed")
draw.text((961 * SCALE, 685 * SCALE), "BAYER DOMAIN", font=font(12, True), fill="#2056ed")

box(105, 722, 715, 90, "Classical RGB baseline", "Exposure + tone finishing · conservative bilateral denoise · detail", "#f7f8fa", "#737d8d")
box(105, 833, 715, 90, "SIDD-CC sRGB NAFNet", "Fine-tuned denoiser for rendered RGB; separate from the Bayer model", "#eef3ff", "#2056ed")
box(105, 944, 715, 90, "RGB hybrid", "High-pass tile ranking · selected learned tiles · smooth blend", "#f7f8fa", "#7384a0")

box(960, 722, 715, 90, "Classical Bayer baseline", "Bilinear demosaic → RGB finishing", "#f7f8fa", "#737d8d")
box(960, 833, 715, 90, "Compact Bayer-v1 student", "Packed RGGB · batched RGB reconstruction · 29,980 parameters", "#eef3ff", "#2056ed")
box(960, 944, 715, 90, "Bayer hybrid", "Relative variance / brightness tile rank · blend Classical + Learned", "#f7f8fa", "#7384a0")

arrow([(330, 596), (330, 635), (465, 635), (465, 666)], "#2056ed")
arrow([(900, 596), (900, 635), (465, 635), (465, 666)], "#738094")
arrow([(1470, 596), (1470, 666)], "#2056ed")

# Result and evaluation
label(80, 1112, "04  /  Compare and measure")
box(80, 1138, 760, 125, "Output comparison", "Available Classical · Learned · Hybrid images\nActual per-run timing and memory scope are labeled", "#ffffff", "#17191e")
box(960, 1138, 760, 125, "Optional reference metrics", "Aligned reference → PSNR · SSIM · Delta E 2000 · LPIPS\nNo reference → no per-upload quality score", "#ffffff", "#2056ed")
arrow([(465, 1072), (465, 1138)])
arrow([(1325, 1072), (1325, 1138)])
arrow([(840, 1200), (960, 1200)], "#2056ed")
arrow([(80, 1200), (40, 1200), (40, 266), (80, 266)], "#738094")

# Research workflow
label(80, 1305, "05  /  Research workflow")
box(80, 1330, 500, 150, "Registered data splits", "Train · calibration · held-out test\nDatasets are acquired separately", "#ffffff", "#737d8d")
box(650, 1330, 500, 150, "Training and gate calibration", "train.py · train_sidd_srgb.py\ncalibrate_gate.py", "#ffffff", "#2056ed")
box(1220, 1330, 500, 150, "Evaluation and export", "evaluate.py · verify_gate.py\nexport_litert.py · verification reports", "#ffffff", "#737d8d")
arrow([(580, 1405), (650, 1405)])
arrow([(1150, 1405), (1220, 1405)])

draw.text((80 * SCALE, 1540 * SCALE), "EXPERIMENTAL · Local inference · Bayer and sRGB metrics are not pooled", font=font(12, True), fill="#687386")

img = img.resize((W, H), Image.Resampling.LANCZOS)
img.save(OUT, optimize=True)
