"""Shared utilities: logging, pickle I/O, console input validation."""
from __future__ import annotations
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Optional


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.propagate = False
    return logger


logger = get_logger(__name__)


def save_pickle(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved: {path}")


def load_pickle(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Run `python -m src.predictor` to train it first."
        )
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise IOError(f"Model file at '{path}' appears corrupted.") from e


def prompt_float(text: str, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    while True:
        raw = input(text).strip()
        try:
            v = float(raw)
        except ValueError:
            print(f"  '{raw}' is not a number. Try again."); continue
        if lo is not None and v < lo: print(f"  Must be >= {lo}."); continue
        if hi is not None and v > hi: print(f"  Must be <= {hi}."); continue
        return v


def prompt_int(text: str, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    while True:
        raw = input(text).strip()
        try:
            v = int(raw)
        except ValueError:
            print(f"  '{raw}' is not an integer. Try again."); continue
        if lo is not None and v < lo: print(f"  Must be >= {lo}."); continue
        if hi is not None and v > hi: print(f"  Must be <= {hi}."); continue
        return v


def prompt_choice(text: str, choices: list) -> str:
    cmap = {c.lower(): c for c in choices}
    while True:
        raw = input(f"{text} [{', '.join(choices)}]: ").strip()
        if raw.lower() in cmap:
            return cmap[raw.lower()]
        print(f"  Invalid choice. Pick one of: {', '.join(choices)}")
