"""Performance utilities."""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(name: str = "Operation") -> Generator[None, None, None]:
    """Context manager to measure execution time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} completed in {elapsed:.4f} seconds")


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"
