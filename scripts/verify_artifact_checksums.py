from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_delivery_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "checksums.sha256").is_file():
            return candidate
    return None


def main() -> None:
    root = Path(".").resolve()
    delivery = find_delivery_root(root)
    if delivery is None:
        raise SystemExit(
            "ARTIFACT_CHECKSUM_FAIL no delivery-root checksums.sha256; "
            "build or extract a submission delivery before checksum verification"
        )
    prefix = root.relative_to(delivery).as_posix()
    prefix = "" if prefix == "." else prefix.rstrip("/") + "/"
    rows = 0
    issues: list[str] = []
    for raw_line in (delivery / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        expected, relative = raw_line.split("  ", 1)
        if prefix and not relative.startswith(prefix):
            continue
        rows += 1
        target = delivery / Path(relative)
        if not target.is_file():
            issues.append(f"missing {relative}")
        elif sha256_file(target) != expected:
            issues.append(f"hash mismatch {relative}")
    if rows == 0:
        issues.append(f"checksum authority contains no entries for prefix {prefix!r}")
    if issues:
        print(f"ARTIFACT_CHECKSUM_FAIL authority=checksums.sha256 rows={rows} issues={len(issues)}")
        for issue in issues[:30]:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"ARTIFACT_CHECKSUM_PASS authority=checksums.sha256 rows={rows}")


if __name__ == "__main__":
    main()
