"""Create a no-copy paired RAW/sRGB SIDD-CC training view for Bayer scripts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def add_link(source: Path, link: Path) -> None:
    if link.is_symlink():
        if link.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to replace unrelated symlink: {link}")
    if link.exists():
        raise FileExistsError(f"Refusing to replace existing file: {link}")
    relative_target = os.path.relpath(source.resolve(), link.parent.resolve())
    link.symlink_to(relative_target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, default=Path("data/siddcc_rawsource/Raw/train/Data"))
    parser.add_argument("--srgb-data", type=Path, default=Path("data/siddcc_srgb/SRGB/train/Data"))
    parser.add_argument("--out", type=Path, default=Path("work/siddcc_bayer_all_train"))
    args = parser.parse_args()

    raw_root = args.raw_data.resolve()
    srgb_root = args.srgb_data.resolve()
    output = args.out
    for name in ("input", "target", "metadata"):
        (output / name).mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_root.rglob("*_NOISY_RAW_*.MAT"))
    if not raw_files:
        raise FileNotFoundError(f"No SIDD-CC RAW training captures found below {raw_root}")

    count = 0
    for raw in raw_files:
        stem = raw.stem
        gt_name = stem.replace("_NOISY_RAW_", "_GT_SRGB_") + ".PNG"
        metadata_name = stem.replace("_NOISY_RAW_", "_METADATA_RAW_") + ".MAT"
        matches = list(srgb_root.rglob(gt_name))
        metadata = raw.with_name(metadata_name)
        if len(matches) != 1 or not metadata.is_file():
            raise FileNotFoundError(
                f"Expected one RGB target and one metadata file for {raw}: "
                f"targets={matches}, metadata={metadata}"
            )
        add_link(raw, output / "input" / raw.name)
        add_link(matches[0], output / "target" / f"{stem}.PNG")
        # The loader seeks metadata/<input-stem>.MAT for MAT Bayer inputs.
        add_link(metadata, output / "metadata" / f"{stem}.MAT")
        count += 1

    print(f"Staged {count} paired SIDD-CC RAW/sRGB captures at {output} (symlinks only).")


if __name__ == "__main__":
    main()
