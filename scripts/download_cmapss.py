from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NASA_CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
FALLBACK_URLS = [
    NASA_CMAPSS_URL,
    "https://zenodo.org/records/15346912/files/CMAPSSData.zip?download=1",
]


def download_one(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 C-MAPSS-RUL-Research/0.1",
            "Accept": "application/zip,application/octet-stream,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"Saved to {target}")


def download(urls: list[str], target: Path) -> None:
    last_error: Exception | None = None
    for url in urls:
        try:
            download_one(url, target)
            return
        except Exception as exc:
            last_error = exc
            print(f"Download failed from {url}: {exc}")
    raise RuntimeError(f"All download URLs failed. Last error: {last_error}")


def extract(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"Extracted to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract NASA C-MAPSS data.")
    parser.add_argument("--url", default=None, help="Override download URL. Default tries NASA first, then fallback mirrors.")
    parser.add_argument("--zip-path", default=str(PROJECT_ROOT / "data" / "raw" / "CMAPSSData.zip"))
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not args.skip_download:
        urls = [args.url] if args.url else FALLBACK_URLS
        download(urls, zip_path)
    if args.extract:
        extract(zip_path, PROJECT_ROOT / "data" / "raw")


if __name__ == "__main__":
    main()
