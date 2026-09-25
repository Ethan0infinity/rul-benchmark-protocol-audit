from __future__ import annotations

from pathlib import Path
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "scripts", "tests", "benchmark", "eval_harness")


def main() -> None:
    checked = 0
    for directory in SOURCE_DIRS:
        root = PROJECT_ROOT / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8-sig")
            with warnings.catch_warnings():
                warnings.simplefilter("error", SyntaxWarning)
                compile(source, str(path), "exec", dont_inherit=True)
            checked += 1
    print(f"PYTHON_SYNTAX_PASS files={checked}")


if __name__ == "__main__":
    main()
