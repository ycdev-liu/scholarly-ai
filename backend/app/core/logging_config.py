"""Logging setup used by the command-line research examples."""

import logging
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str | None = None, log_dir: str = "./logs") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(directory / log_file, encoding="utf-8"))
    logging.basicConfig(level=log_level, handlers=handlers, force=True)
