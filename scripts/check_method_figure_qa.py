from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).absolute().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="QA the vector-first method workflow export bundle.")
    parser.add_argument(
        "--figure-base",
        default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence" / "figures" / "fig_method_architecture"),
    )
    args = parser.parse_args()
    base = Path(args.figure_base)
    required = [base.with_suffix(ext) for ext in (".svg", ".pdf", ".tiff", ".png")]
    missing = [path for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing method-figure exports: {missing}")

    svg = base.with_suffix(".svg").read_text(encoding="utf-8")
    if "<text" not in svg:
        raise ValueError("SVG text is not editable.")
    if re.search(r"font-size:\s*([0-4](?:\.|px))", svg):
        raise ValueError("SVG contains text below the conservative final-size threshold.")
    for phrase in ("Study design", "Model estimation", "Fixed test evaluation", "test boundary"):
        if phrase not in svg:
            raise ValueError(f"Required scientific-flow label missing from SVG: {phrase}")

    with Image.open(base.with_suffix(".tiff")) as image:
        dpi = image.info.get("dpi", (0, 0))
        if min(dpi) < 590:
            raise ValueError(f"TIFF resolution is below 600 dpi tolerance: {dpi}")
        if image.width < 4000:
            raise ValueError(f"TIFF is unexpectedly narrow: {image.width}px")
    with Image.open(base.with_suffix(".png")) as image:
        if image.width < 2500 or image.height < 1400:
            raise ValueError(f"PNG preview is too small: {image.size}")
    print("METHOD_FIGURE_QA_PASS editable_svg=1 pdf=1 tiff_600dpi=1 png_preview=1")


if __name__ == "__main__":
    main()
