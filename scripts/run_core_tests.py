from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    path = PROJECT_ROOT / "tests" / "test_core.py"
    spec = importlib.util.spec_from_file_location("test_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = sorted(name for name in dir(module) if name.startswith("test_"))
    for name in tests:
        getattr(module, name)()
        print(f"PASS {name}")
    print(f"CORE_TESTS_PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
