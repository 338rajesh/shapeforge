"""
Profiling utilities for ShapeForge.

Provides deterministic cProfile and pstats instrumentation, formatted terminal
hotspot summaries, and binary profile exports compatible with visualization tools
such as SnakeViz, Tuna, and VS Code Profile Viewer.
"""

from __future__ import annotations

import cProfile
import io
import pstats
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from .utils import get_logger

logger = get_logger(__name__)


class Profiler:
    """
    Context manager and controller for profiling code blocks using cProfile.

    Parameters
    ----------
    output_path : Path | str | None, optional
        Path where the binary `.prof` file will be saved upon exit.
    top_n : int, optional
        Number of top hotspot entries to display in the summary table (default: 25).
    sort_by : str, optional
        Metric used to sort the statistics. Options include 'cumulative' (default),
        'time' (self/internal time), 'calls', etc.
    print_summary : bool, optional
        Whether to automatically log/print the hotspot summary on exit (default: True).
    text_report_path : Path | str | None, optional
        Optional path to write a full human-readable text report.
    """

    def __init__(
        self,
        output_path: Path | str | None = None,
        *,
        top_n: int = 25,
        sort_by: str = "cumulative",
        print_summary: bool = True,
        text_report_path: Path | str | None = None,
    ):
        self.output_path = Path(output_path) if output_path else None
        self.top_n = top_n
        self.sort_by = sort_by
        self.print_summary = print_summary
        self.text_report_path = (
            Path(text_report_path) if text_report_path else None
        )
        self._profiler = cProfile.Profile()
        self._stats: pstats.Stats | None = None
        self._is_running = False

    def enable(self) -> None:
        """Start or resume profiling."""
        if not self._is_running:
            self._profiler.enable()
            self._is_running = True

    def disable(self) -> None:
        """Stop profiling."""
        if self._is_running:
            self._profiler.disable()
            self._is_running = False

    def start(self) -> None:
        """Alias for enable()."""
        self.enable()

    def stop(self) -> None:
        """Alias for disable()."""
        self.disable()

    def __enter__(self) -> Profiler:
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disable()
        self._build_stats()

        if self.output_path:
            self.dump_stats(self.output_path)

        if self.text_report_path:
            self.dump_text_report(self.text_report_path)

        if self.print_summary:
            print(self.get_summary(top_n=self.top_n, sort_by=self.sort_by))

    def _build_stats(self) -> pstats.Stats:
        if self._stats is None:
            self._stats = pstats.Stats(self._profiler)
        return self._stats

    @property
    def stats(self) -> pstats.Stats:
        """Returns the underlying pstats.Stats instance."""
        return self._build_stats()

    def dump_stats(self, path: Path | str) -> Path:
        """
        Save binary profiling data in standard pstats format.
        Compatible with SnakeViz, Tuna, Speedscope, and VS Code.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use fresh stats instance without strip_dirs to retain absolute file paths for IDEs
        pstats.Stats(self._profiler).dump_stats(str(path))
        return path

    def dump_text_report(
        self,
        path: Path | str,
        top_n: int = 50,
        sort_by: str = "cumulative",
    ) -> Path:
        """Save a human-readable text summary of profiling statistics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary(top_n=top_n, sort_by=sort_by)
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary)
        return path

    def get_summary(
        self,
        top_n: int | None = None,
        sort_by: str | None = None,
    ) -> str:
        """
        Format a clean, readable summary of top performance bottlenecks.
        """
        top_n = self.top_n if top_n is None else top_n
        sort_by = self.sort_by if sort_by is None else sort_by

        stream = io.StringIO()
        stats_copy = pstats.Stats(self._profiler, stream=stream)
        stats_copy.strip_dirs()
        stats_copy.sort_stats(sort_by)

        total_tt = getattr(stats_copy, "total_tt", 0.0)
        total_calls = getattr(stats_copy, "total_calls", 0)

        header = [
            "=" * 80,
            f"{'ShapeForge Performance Profile':^80}",
            "=" * 80,
            f" Total Duration : {total_tt:8.4f} seconds",
            f" Function Calls : {total_calls:,} calls",
            f" Sort Criterion : {sort_by}",
        ]
        if self.output_path:
            header.append(f" Output File    : {self.output_path}")

        header.extend([
            "-" * 80,
            f"Top {top_n} Hotspots:",
            "-" * 80,
        ])

        stats_copy.print_stats(top_n)
        raw_stats = stream.getvalue()

        footer = [
            "-" * 80,
            "Interactive Visualization Tips:",
            "  * SnakeViz (web GUI):    pip install snakeviz && snakeviz <profile.prof>",
            "  * Tuna (icicle viewer):  pip install tuna && tuna <profile.prof>",
            "  * VS Code:               Install 'Python Profile Viewer' extension and open <profile.prof>",
            "=" * 80,
        ]

        return (
            "\n"
            + "\n".join(header)
            + "\n"
            + raw_stats.strip()
            + "\n"
            + "\n".join(footer)
            + "\n"
        )


def profile_function(
    output_path: Path | str | None = None,
    *,
    top_n: int = 25,
    sort_by: str = "cumulative",
    print_summary: bool = True,
) -> Callable:
    """
    Decorator to profile individual function calls.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Profiler(
                output_path=output_path,
                top_n=top_n,
                sort_by=sort_by,
                print_summary=print_summary,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator

