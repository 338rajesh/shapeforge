import pstats
from pathlib import Path

import pytest
from shapeforge.main import build_parser
from shapeforge.profiler import Profiler, profile_function


def test_profiler_context_manager(tmp_path: Path):
    prof_file = tmp_path / "test.prof"
    text_file = tmp_path / "test_report.txt"

    with Profiler(
        output_path=prof_file,
        top_n=10,
        sort_by="cumulative",
        print_summary=False,
        text_report_path=text_file,
    ) as prof:
        # Perform some work to generate profile stats
        total = sum(i * i for i in range(10000))
        assert total > 0

    assert prof_file.exists()
    assert prof_file.stat().st_size > 0
    assert text_file.exists()
    assert "ShapeForge Performance Profile" in text_file.read_text()

    # Verify that the dumped stats can be loaded by standard pstats
    loaded_stats = pstats.Stats(str(prof_file))
    assert loaded_stats.total_calls > 0


def test_profiler_get_summary():
    prof = Profiler(print_summary=False)
    with prof:
        _ = [x**2 for x in range(5000)]

    summary = prof.get_summary(top_n=5, sort_by="time")
    assert "ShapeForge Performance Profile" in summary
    assert "Interactive Visualization Tips" in summary
    assert "SnakeViz" in summary


def test_profile_function_decorator(tmp_path: Path):
    prof_file = tmp_path / "decorated.prof"

    @profile_function(output_path=prof_file, print_summary=False)
    def compute_sum(n: int) -> int:
        return sum(x for x in range(n))

    result = compute_sum(1000)
    assert result == 499500
    assert prof_file.exists()


def test_cli_parser_profile_flags():
    parser = build_parser()

    # Default arguments
    args = parser.parse_args(["demo/config.yaml"])
    assert args.config_file == "demo/config.yaml"
    assert args.profile is False
    assert args.profile_output is None
    assert args.profile_lines == 25
    assert args.profile_sort == "cumulative"

    # Profile enabled with custom options
    custom_args = parser.parse_args([
        "demo/config.yaml",
        "--profile",
        "--profile-output",
        "custom_profile.prof",
        "--profile-lines",
        "50",
        "--profile-sort",
        "time",
    ])
    assert custom_args.profile is True
    assert custom_args.profile_output == "custom_profile.prof"
    assert custom_args.profile_lines == 50
    assert custom_args.profile_sort == "time"

