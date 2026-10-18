"""
Profiling utilities for ShapeForge.
===================================

Provides deterministic ``cProfile`` and ``pstats`` instrumentation, formatted terminal
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
    Context manager and controller for profiling execution blocks using ``cProfile``.

    Captures low-overhead execution metrics, formats terminal hotspot tables,
    and exports binary ``.prof`` files for interactive web/GUI visualizers.

    Parameters
    ----------
    output_path : Path or str, optional
        Filesystem destination where the binary ``.prof`` file will be saved upon exit.
    top_n : int, optional
        Number of top bottleneck function entries to show in the summary table
        (default: 25).
    sort_by : str, optional
        Sorting metric for statistical ranking. Options include ``'cumulative'``
        (default), ``'time'`` (internal execution time), or ``'calls'``.
    print_summary : bool, optional
        Whether to print the formatted summary report to stdout on exit
        (default: True).
    text_report_path : Path or str, optional
        Optional path to write a plain-text hotspot summary report.

    Attributes
    ----------
    output_path : Path or None
        Path to output binary profile.
    top_n : int
        Number of hotspots to display.
    sort_by : str
        Active sort metric.
    print_summary : bool
        Flag indicating if stdout report is enabled.
    text_report_path : Path or None
        Path to output plain-text report.

    Examples
    --------
    >>> from shapeforge.profiler import Profiler
    >>> with Profiler(top_n=5, print_summary=False) as prof:
    ...     total = sum(i**2 for i in range(10000))
    >>> prof.stats is not None
    True
    """

    def __init__(
        self,
        output_path: Path | str | None = None,
        *,
        top_n: int = 25,
        sort_by: str = "cumulative",
        print_summary: bool = True,
        text_report_path: Path | str | None = None,
    ) -> None:
        self.output_path: Path | None = Path(output_path) if output_path else None
        self.top_n: int = top_n
        self.sort_by: str = sort_by
        self.print_summary: bool = print_summary
        self.text_report_path: Path | None = (
            Path(text_report_path) if text_report_path else None
        )
        self._profiler = cProfile.Profile()
        self._stats: pstats.Stats | None = None
        self._is_running: bool = False

    def enable(self) -> None:
        """Start or resume profiling execution."""
        if not self._is_running:
            self._profiler.enable()
            self._is_running = True

    def disable(self) -> None:
        """Pause or stop profiling execution."""
        if self._is_running:
            self._profiler.disable()
            self._is_running = False

    def start(self) -> None:
        """Alias for :meth:`enable`."""
        self.enable()

    def stop(self) -> None:
        """Alias for :meth:`disable`."""
        self.disable()

    def __enter__(self) -> Profiler:
        """Enter the runtime context and activate profiling."""
        self.enable()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the runtime context, stop profiling, and export reports."""
        self.disable()
        self._build_stats()

        if self.output_path:
            self.dump_stats(self.output_path)

        if self.text_report_path:
            self.dump_text_report(self.text_report_path)

        if self.print_summary:
            print(self.get_summary(top_n=self.top_n, sort_by=self.sort_by))

    def _build_stats(self) -> pstats.Stats:
        """Build and cache the pstats.Stats instance."""
        if self._stats is None:
            self._stats = pstats.Stats(self._profiler)
        return self._stats

    @property
    def stats(self) -> pstats.Stats:
        """
        Underlying :class:`pstats.Stats` instance.

        Returns
        -------
        pstats.Stats
            Calculated profiling statistics.
        """
        return self._build_stats()

    def dump_stats(self, path: Path | str) -> Path:
        """
        Save binary profiling data in standard pstats format.

        Compatible with SnakeViz, Tuna, Speedscope, and VS Code Profile Viewer.

        Parameters
        ----------
        path : Path or str
            Destination file path for the binary profile (e.g. ``'profile.prof'``).

        Returns
        -------
        Path
            Absolute or resolved path to the saved file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pstats.Stats(self._profiler).dump_stats(str(p))
        return p

    def dump_text_report(
        self,
        path: Path | str,
        top_n: int = 50,
        sort_by: str = "cumulative",
    ) -> Path:
        """
        Save a human-readable text summary of profiling statistics to disk.

        Parameters
        ----------
        path : Path or str
            Destination file path for the text report.
        top_n : int, optional
            Number of top lines to include (default: 50).
        sort_by : str, optional
            Sorting metric (default: ``'cumulative'``).

        Returns
        -------
        Path
            Path to the saved report file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary(top_n=top_n, sort_by=sort_by)
        with open(p, "w", encoding="utf-8") as f:
            f.write(summary)
        return p

    def get_summary(
        self,
        top_n: int | None = None,
        sort_by: str | None = None,
    ) -> str:
        """
        Format a clean, human-readable terminal summary of top performance bottlenecks.

        Parameters
        ----------
        top_n : int, optional
            Number of hotspot functions to display. If None, uses ``self.top_n``.
        sort_by : str, optional
            Sort key. If None, uses ``self.sort_by``.

        Returns
        -------
        str
            Formatted multi-line report string with headers, statistics, and tips.
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
) -> Callable[..., Any]:
    """
    Decorator to profile individual function or method invocations.

    Parameters
    ----------
    output_path : Path or str, optional
        Filesystem destination to write the binary ``.prof`` file.
    top_n : int, optional
        Number of hotspot entries to display in the console summary (default: 25).
    sort_by : str, optional
        Sorting metric (default: ``'cumulative'``).
    print_summary : bool, optional
        Whether to log the formatted hotspot table to stdout (default: True).

    Returns
    -------
    Callable
        Decorated function that executes within a :class:`Profiler` context.

    Examples
    --------
    >>> from shapeforge.profiler import profile_function
    >>> @profile_function(top_n=5, print_summary=False)
    ... def compute():
    ...     return sum(i for i in range(1000))
    >>> compute()
    499500
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
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
