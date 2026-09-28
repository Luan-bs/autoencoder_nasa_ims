"""
src/data/extractor.py

Extracts IMS Bearing RAR archives to data/interim/ and validates the result.

Usage (CLI):
    uv run python -m data.extractor --set 2nd_test
    uv run python -m data.extractor --set 2nd_test --dry-run

Usage (import):
    from data.extractor import extract_set
    result = extract_set("2nd_test")

Rules:
  - data/raw/ is NEVER modified (read-only).
  - Output always goes to data/interim/<set_name>/.
  - __MACOSX dirs and ._* files are removed after extraction.
  - File count is compared against expected_file_count in configs/data.yaml.
  - A divergence is logged as WARNING but does not abort — the caller decides.
  - Log is written to data/interim/<set_name>_extraction.log.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# ── Project paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "configs" / "data.yaml"

# WinRAR confirmed at this path on the target machine
DEFAULT_WINRAR = Path("C:/Program Files/WinRAR/WinRAR.exe")

# IMS data files are named YYYY.MM.DD.HH.MM.SS (no extension)
_IMS_FILENAME_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Structured result returned by extract_set()."""

    set_name: str
    rar_path: Path
    output_dir: Path
    expected_count: int
    actual_count: int
    count_ok: bool
    skipped_files: list[str] = field(default_factory=list)   # non-data files found
    warnings: list[str]      = field(default_factory=list)
    log_path: Optional[Path] = None
    success: bool = False


# ── Logger ────────────────────────────────────────────────────────────────────

def _setup_logger(log_path: Path) -> logging.Logger:
    """Logger that writes to both stdout and a UTF-8 log file."""
    name   = f"extractor.{log_path.stem}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load configs/data.yaml. Raises FileNotFoundError if missing."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_set_config(cfg: dict, set_name: str) -> dict:
    """Return the config sub-dict for a named test set."""
    sets = cfg.get("sets", {})
    if set_name not in sets:
        raise ValueError(
            f"Unknown set '{set_name}'. Available: {sorted(sets)}"
        )
    return sets[set_name]


# ── Filesystem helpers ────────────────────────────────────────────────────────

def _find_data_files(directory: Path) -> tuple[list[Path], list[Path]]:
    """
    Walk *directory* recursively and separate files into:
      - data_files  : IMS snapshot files (name matches YYYY.MM.DD.HH.MM.SS)
      - skipped     : everything else (macOS artifacts, READMEs, etc.)

    Returns (data_files, skipped).
    """
    data_files: list[Path] = []
    skipped:    list[Path] = []

    for p in sorted(directory.rglob("*")):
        if p.is_dir():
            continue
        if _IMS_FILENAME_RE.match(p.name):
            data_files.append(p)
        else:
            skipped.append(p)

    return data_files, skipped


def _remove_macos_artifacts(directory: Path, logger: logging.Logger) -> int:
    """
    Delete __MACOSX directories and ._* files created by macOS archiving tools.
    Returns the number of items removed.
    """
    removed = 0

    for macos_dir in list(directory.rglob("__MACOSX")):
        if macos_dir.is_dir():
            shutil.rmtree(macos_dir)
            logger.info(f"  Removed __MACOSX/  →  {macos_dir.relative_to(directory)}")
            removed += 1

    for dot_under in list(directory.rglob("._*")):
        if dot_under.is_file():
            dot_under.unlink()
            logger.info(f"  Removed ._* file   →  {dot_under.relative_to(directory)}")
            removed += 1

    return removed


# ── WinRAR wrapper ────────────────────────────────────────────────────────────

def _run_winrar(
    winrar_exe: Path,
    rar_path: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Extract *rar_path* into *output_dir* using WinRAR's CLI.

    WinRAR flags used:
        x      – extract with full paths
        -y     – answer Yes to all prompts
        -ibck  – suppress GUI window (background mode)

    Exit codes:
        0 – success
        1 – warning (non-fatal; logged, execution continues)
        2+ – error (raises RuntimeError)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        str(winrar_exe),
        "x",
        str(rar_path),
        str(output_dir) + "\\",
        "-y",
        "-ibck",
    ]

    logger.info(f"CMD: {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=900,   # 15 min; 3rd_test is ~581 MB
    )

    if proc.stdout.strip():
        logger.debug(f"WinRAR stdout:\n{proc.stdout.strip()}")
    if proc.stderr.strip():
        logger.warning(f"WinRAR stderr:\n{proc.stderr.strip()}")

    if proc.returncode >= 2:
        raise RuntimeError(
            f"WinRAR failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    if proc.returncode == 1:
        logger.warning("WinRAR exited with code 1 (warnings). Continuing.")


# ── Public API ────────────────────────────────────────────────────────────────

def extract_set(
    set_name: str,
    winrar_exe: Path = DEFAULT_WINRAR,
    dry_run: bool = False,
) -> ExtractionResult:
    """
    Extract one IMS test set and validate the result.

    Parameters
    ----------
    set_name : str
        Key in configs/data.yaml → sets  (e.g. "2nd_test").
    winrar_exe : Path
        Path to WinRAR.exe.
    dry_run : bool
        Skip extraction; only count and validate whatever is already in
        data/interim/<set_name>/. Useful for re-validating without re-extracting.

    Returns
    -------
    ExtractionResult
        Structured outcome. result.success is True unless a fatal error occurred.
        A file-count mismatch sets result.count_ok = False and adds a warning,
        but does NOT set success = False — the caller decides how to handle it.
    """
    cfg     = _load_config()
    set_cfg = _get_set_config(cfg, set_name)

    rar_path       = PROJECT_ROOT / set_cfg["rar_file"]
    output_dir     = PROJECT_ROOT / set_cfg["interim_dir"]
    expected_count: int = set_cfg["expected_file_count"]

    # Log file lands next to the set folder, inside data/interim/
    interim_root = output_dir.parent
    interim_root.mkdir(parents=True, exist_ok=True)
    log_path = interim_root / f"{set_name}_extraction.log"

    logger = _setup_logger(log_path)

    # ── Header ───────────────────────────────────────────────────────────
    sep = "=" * 68
    logger.info(sep)
    logger.info(f"  IMS Extractor  |  set: {set_name}")
    logger.info(f"  Started : {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"  RAR     : {rar_path}")
    logger.info(f"  Output  : {output_dir}")
    logger.info(f"  Expected: {expected_count} data files")
    logger.info(f"  Dry run : {dry_run}")
    logger.info(sep)

    result = ExtractionResult(
        set_name       = set_name,
        rar_path       = rar_path,
        output_dir     = output_dir,
        expected_count = expected_count,
        actual_count   = 0,
        count_ok       = False,
        log_path       = log_path,
    )

    # ── Pre-flight ────────────────────────────────────────────────────────
    if not rar_path.exists():
        msg = f"RAR not found: {rar_path}"
        logger.error(msg)
        result.warnings.append(msg)
        return result   # success stays False

    if not dry_run and not winrar_exe.exists():
        msg = f"WinRAR.exe not found: {winrar_exe}"
        logger.error(msg)
        result.warnings.append(msg)
        return result

    # ── Extraction (or skip) ──────────────────────────────────────────────
    if dry_run:
        logger.info("DRY RUN — skipping extraction, validating existing files.")

    elif output_dir.exists() and any(output_dir.iterdir()):
        msg = (
            f"Output dir already exists and is non-empty: {output_dir}  "
            "→ skipping extraction. Delete the folder to re-extract."
        )
        logger.warning(msg)
        result.warnings.append(msg)

    else:
        logger.info("Extracting…")
        _run_winrar(winrar_exe, rar_path, output_dir, logger)
        logger.info("Extraction finished.")

    # ── Cleanup macOS artifacts ───────────────────────────────────────────
    n_removed = _remove_macos_artifacts(output_dir, logger)
    if n_removed:
        logger.info(f"Removed {n_removed} macOS artifact(s).")
    else:
        logger.info("No macOS artifacts found.")

    # ── Count data files ──────────────────────────────────────────────────
    data_files, skipped = _find_data_files(output_dir)
    result.actual_count = len(data_files)
    result.skipped_files = [str(p.relative_to(output_dir)) for p in skipped]

    logger.info(f"Data files found  : {result.actual_count}")
    logger.info(f"Non-data files    : {len(skipped)}")
    if skipped:
        logger.info("  Non-data files (skipped, not counted):")
        for p in skipped:
            logger.info(f"    {p.relative_to(output_dir)}")

    # ── Count comparison ──────────────────────────────────────────────────
    if result.actual_count == expected_count:
        result.count_ok = True
        logger.info(f"✓  Count matches expected ({expected_count}).")
    else:
        diff = result.actual_count - expected_count
        sign = "+" if diff > 0 else ""
        msg = (
            f"⚠  COUNT MISMATCH — expected {expected_count}, "
            f"got {result.actual_count} ({sign}{diff}). "
            "Verify the archive integrity or update expected_file_count in data.yaml."
        )
        logger.warning(msg)
        result.warnings.append(msg)

    # ── Summary ───────────────────────────────────────────────────────────
    result.success = True
    logger.info("-" * 68)
    logger.info("  SUMMARY")
    logger.info(f"  Set name     : {set_name}")
    logger.info(f"  Expected     : {expected_count}")
    logger.info(f"  Actual       : {result.actual_count}")
    logger.info(f"  Count OK     : {result.count_ok}")
    logger.info(f"  Warnings     : {len(result.warnings)}")
    logger.info(f"  Log file     : {log_path}")
    logger.info(f"  Finished     : {datetime.now().isoformat(timespec='seconds')}")
    logger.info(sep)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract an IMS Bearing RAR archive to data/interim/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--set",
        required=True,
        dest="set_name",
        metavar="SET_NAME",
        help="Test set key in configs/data.yaml (e.g. 2nd_test, 1st_test, 3rd_test).",
    )
    p.add_argument(
        "--winrar",
        default=str(DEFAULT_WINRAR),
        metavar="PATH",
        help="Path to WinRAR.exe.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip extraction; only count and validate files already in interim/.",
    )
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    result = extract_set(
        set_name   = args.set_name,
        winrar_exe = Path(args.winrar),
        dry_run    = args.dry_run,
    )

    if result.warnings:
        print("\n⚠  Warnings:")
        for w in result.warnings:
            print(f"   - {w}")

    sys.exit(0 if (result.success and result.count_ok) else 1)


if __name__ == "__main__":
    main()
