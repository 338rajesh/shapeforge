import json

import numpy as np
import pytest
import yaml

from shapeforge.optim import OptimisationResult
from shapeforge.utils import (
    DistributionSampler,
    DistributionSpec,
    _load_dict,
    plot_f_and_g_norms,
)


def test_distribution_spec_valid():
    spec = DistributionSpec("uniform", loc=1.0, scale=0.5)
    assert spec.distribution == "uniform"
    assert spec.loc == 1.0
    assert spec.scale == 0.5

    rvs = spec.rvs_partial()
    val = rvs()
    assert 1.0 <= val <= 1.5

    # from_signature
    sig_spec = DistributionSpec.from_signature("uniform(2.0, 0.4)")
    assert sig_spec.distribution == "uniform"
    assert sig_spec.loc == 2.0
    assert sig_spec.scale == 0.4

    # Idempotence with existing instance
    same_spec = DistributionSpec.from_signature(sig_spec)
    assert same_spec is sig_spec


def test_distribution_spec_invalid():
    # Unsupported distribution
    with pytest.raises(ValueError, match="Unsupported distribution"):
        DistributionSpec("non_existent_dist_xyz", loc=0.0, scale=1.0)

    # Negative scale
    with pytest.raises(Exception):
        DistributionSpec("norm", loc=0.0, scale=-1.0)

    # Malformed signature string
    with pytest.raises(ValueError, match="Invalid signature"):
        DistributionSpec.from_signature("not_a_function_call")

    # Non-string signature
    with pytest.raises(TypeError):
        DistributionSpec.from_signature(12345)


def test_distribution_sampler():
    rng = np.random.default_rng(42)
    spec = DistributionSpec("norm", loc=10.0, scale=1.0)
    sampler = DistributionSampler(spec, rng=rng)

    samples = sampler.sample(size=100)
    assert len(samples) == 100
    np.testing.assert_allclose(np.mean(samples), 10.0, atol=0.5)

    # from_signature
    sig_sampler = DistributionSampler.from_signature(
        "uniform(5.0, 2.0)", rng=rng
    )
    sample_val = sig_sampler.sample()
    assert 5.0 <= sample_val <= 7.0


def test_load_dict(tmp_path):
    # Pass dict directly
    d = {"hello": "world", "num": 42}
    assert _load_dict(d) is d

    # Load JSON
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump(d, f)
    assert _load_dict(json_path) == d

    # Load YAML
    yaml_path = tmp_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(d, f)
    assert _load_dict(yaml_path) == d

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        _load_dict(tmp_path / "missing.json")


def test_plot_f_and_g_norms(tmp_path):
    result = OptimisationResult(
        x_optimal=np.array([1.0, 2.0]),
        status="success",
        f_history=[10.0, 5.0, 1.0],
        g_norm_history=[5.0, 2.0, 0.1],
        iter_count=2,
    )
    out_file = tmp_path / "convergence.png"
    plot_f_and_g_norms(result, f_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
