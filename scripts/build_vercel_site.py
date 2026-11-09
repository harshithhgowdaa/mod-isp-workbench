"""Build the static judge-facing comparison page for Vercel."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
dist = root / "dist"
showcase = root / "static" / "showcase"
required = (
    showcase / "input.jpg",
    showcase / "classical.jpg",
    showcase / "learned.jpg",
    showcase / "hybrid.jpg",
    showcase / "reference.png",
    showcase / "results.json",
    showcase / "nafnet-srgb.onnx",
    root / "static" / "vendor" / "ort" / "ort.wasm.min.js",
    root / "static" / "vendor" / "ort" / "ort-wasm-simd-threaded.mjs",
    root / "static" / "vendor" / "ort" / "ort-wasm-simd-threaded.wasm",
    showcase / "MODEL-LICENSE.md",
    showcase / "MODEL-NOTICE.md",
)
missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
if missing:
    raise SystemExit(
        "Static showcase assets are missing: " + ", ".join(missing)
        + ". Rebuild them with .venv312/bin/python scripts/build_static_showcase_assets.py"
    )

if dist.exists():
    shutil.rmtree(dist)
dist.mkdir()
shutil.copy2(root / "templates" / "showcase.html", dist / "index.html")
shutil.copytree(
    root / "static",
    dist / "static",
    ignore=shutil.ignore_patterns("app.js", "demo"),
)
