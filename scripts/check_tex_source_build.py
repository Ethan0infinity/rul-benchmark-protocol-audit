"""Compile the release-facing TeX sources in an isolated temporary directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_sources(package_root: Path | None) -> list[Path]:
    if package_root is not None:
        return [
            package_root / "manuscript" / "main_revised.tex",
            package_root / "supplementary" / "manuscript" / "supplement_en.tex",
        ]
    manuscript = next(
        (
            candidate / "main_revised.tex"
            for candidate in PROJECT_ROOT.parent.iterdir()
            if candidate.is_dir() and (candidate / "main_revised.tex").exists()
        ),
        None,
    )
    if manuscript is None:
        manuscript = PROJECT_ROOT.parent / "正文" / "main_revised.tex"
    return [manuscript, PROJECT_ROOT / "supplementary" / "supplement_en.tex"]


def required_tex_tools(sources: list[Path]) -> list[str]:
    tools = ["latexmk", "xelatex"]
    text = "\n".join(
        source.read_text(encoding="utf-8-sig", errors="replace")
        for source in sources
        if source.exists()
    )
    if "\\bibliography{" in text:
        tools.append("bibtex")
    if "\\addbibresource{" in text or "\\printbibliography" in text:
        tools.append("biber")
    return tools


def version_line(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    lines = [
        line.strip()
        for line in (completed.stdout + completed.stderr).splitlines()
        if line.strip()
    ]
    filtered = [
        line
        for line in lines
        if not line.startswith(("Initial Win CP", "I changed", "Reverting Windows"))
    ]
    return (filtered or lines or [""])[0]


def compile_source(source: Path, output_root: Path) -> dict[str, object]:
    output = output_root / source.stem
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "latexmk",
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={output}",
        source.name,
    ]
    env = os.environ.copy()
    env["max_print_line"] = "1000"
    completed = subprocess.run(
        command,
        cwd=source.parent,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    pdf = output / source.with_suffix(".pdf").name
    combined = (completed.stdout + completed.stderr).strip()
    combined = combined.replace(str(source.parent), ".")
    combined = combined.replace(source.parent.as_posix(), ".")
    combined = combined.replace(str(output_root), "<temporary>")
    combined = combined.replace(output_root.as_posix(), "<temporary>")
    return {
        "source": source.name,
        "command": [
            "latexmk",
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-outdir=<temporary>",
            source.name,
        ],
        "exit_code": completed.returncode,
        "pdf_created": pdf.exists() and pdf.stat().st_size >= 10000,
        "pdf_size_bytes": pdf.stat().st_size if pdf.exists() else 0,
        "output_tail": combined.splitlines()[-20:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild main and Supplementary PDFs from source."
    )
    parser.add_argument("--package-root", default="")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check the complete TeX toolchain before any compilation starts.",
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "reports" / "tex_source_build_gate.json"),
    )
    args = parser.parse_args()
    package_root = Path(args.package_root).resolve() if args.package_root else None
    sources = resolve_sources(package_root)
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise SystemExit("TEX_SOURCE_BUILD_FAIL missing=" + ", ".join(missing))
    required = required_tex_tools(sources)
    resolved_tools = {tool: shutil.which(tool) for tool in required}
    tool_availability = {tool: path is not None for tool, path in resolved_tools.items()}
    tool_records = {
        tool: {
            "available": path is not None,
            "executable": Path(path).name if path else "",
            "version": version_line([path, "--version"]) if path else "",
        }
        for tool, path in resolved_tools.items()
    }
    unavailable = [tool for tool, path in resolved_tools.items() if path is None]
    if unavailable:
        report = {
            "schema_version": "1.2",
            "status": "DEPENDENCY_MISSING",
            "mode": "preflight" if args.preflight_only else "build",
            "required_tools": required,
            "tool_availability": tool_availability,
            "tools": tool_records,
            "missing_tools": unavailable,
            "source_count": len(sources),
            "records": [],
            "issues": [f"missing executable: {tool}" for tool in unavailable],
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("TEX_DEPENDENCY_MISSING " + ", ".join(unavailable))
        raise SystemExit(2)

    if args.preflight_only:
        report = {
            "schema_version": "1.2",
            "status": "PASS",
            "mode": "preflight",
            "required_tools": required,
            "tool_availability": tool_availability,
            "tools": tool_records,
            "source_count": len(sources),
            "sources": [source.name for source in sources],
            "issues": [],
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            "TEX_TOOLCHAIN_PREFLIGHT_PASS "
            + " ".join(f"{tool}={tool_records[tool]['version']}" for tool in required)
        )
        return

    with tempfile.TemporaryDirectory(prefix="rul_tex_build_") as temporary:
        records = [compile_source(source, Path(temporary)) for source in sources]
    issues = [
        str(record["source"])
        for record in records
        if record["exit_code"] != 0 or not record["pdf_created"]
    ]
    report = {
        "schema_version": "1.2",
        "mode": "build",
        "required_tools": required,
        "tool_availability": tool_availability,
        "tools": tool_records,
        "latexmk": version_line(["latexmk", "--version"]),
        "xelatex": version_line(["xelatex", "--version"]),
        "source_count": len(sources),
        "records": records,
        "issues": issues,
        "status": "PASS" if not issues else "FAIL",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if issues:
        print("TEX_SOURCE_BUILD_FAIL " + ", ".join(issues))
        raise SystemExit(1)
    print(
        f"TEX_SOURCE_BUILD_PASS sources={len(sources)} "
        f"latexmk={report['latexmk']} xelatex={report['xelatex']}"
    )


if __name__ == "__main__":
    main()
