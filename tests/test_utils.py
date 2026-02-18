"""Tests of miscellaneous utility functions not part of the main 21cmFAST simulation modules."""

import numpy as np
import pytest

from py21cmfast import InputParameters
from py21cmfast.utils import recursive_difference, show_references


def test_ref_printing():
    inputs = InputParameters.from_template("latest", random_seed=1234)
    ref_str = show_references(inputs, lightcone=True, print_to_stdout=False)

    assert "2011MNRAS.411..955M" in ref_str  # 21cmFAST first paper
    assert "10.21105/joss.02582" in ref_str  # v3 (wrapper)
    assert "10.1093/mnras/stu377" in ref_str  # INHOMO_RECO
    assert "10.1093/mnras/sty796" in ref_str  # LIGHTCONE + RSD
    assert "10.1093/mnras/stz032" in ref_str  # USE_MASS_DEPENDENT_ZETA
    assert "10.1093/mnras/staa1131" not in ref_str  # USE_MINI_HALOS
    assert "10.1093/mnras/stac185" not in ref_str  # USE_RELATIVE_VELOCITIES
    assert "10.1093/mnras/stac2756" not in ref_str  # PHOTON_CONS
    assert "10.1051/0004-6361/202554951" not in ref_str  # DISCRETE_HALOS
    assert "10.48550/arXiv.2601.14360" not in ref_str  # LYA_MULTIPLE_SCATTERING

    inputs = InputParameters.from_template(
        "mini-dhalos", random_seed=1234, K_MAX_FOR_CLASS=1.0
    )
    ref_str = show_references(inputs, lightcone=True, print_to_stdout=False)

    assert "2011MNRAS.411..955M" in ref_str  # 21cmFAST first paper
    assert "10.21105/joss.02582" in ref_str  # v3 (wrapper)
    assert "10.1093/mnras/stu377" in ref_str  # INHOMO_RECO
    assert "10.1093/mnras/sty796" in ref_str  # LIGHTCONE + RSD
    assert "10.1093/mnras/stz032" in ref_str  # USE_MASS_DEPENDENT_ZETA
    assert "10.1093/mnras/staa1131" in ref_str  # USE_MINI_HALOS
    assert "10.1093/mnras/stac185" in ref_str  # USE_RELATIVE_VELOCITIES
    assert "10.1093/mnras/stac2756" not in ref_str  # PHOTON_CONS
    assert "10.1051/0004-6361/202554951" in ref_str  # DISCRETE_HALOS
    assert "10.48550/arXiv.2601.14360" not in ref_str  # LYA_MULTIPLE_SCATTERING

    inputs = InputParameters.from_template("const-zeta", random_seed=1234)
    ref_str = show_references(inputs, lightcone=True, print_to_stdout=True)

    assert "2011MNRAS.411..955M" in ref_str  # 21cmFAST first paper
    assert "10.21105/joss.02582" in ref_str  # v3 (wrapper)
    assert "10.1093/mnras/stu377" not in ref_str  # INHOMO_RECO
    assert "10.1093/mnras/sty796" in ref_str  # LIGHTCONE + RSD
    assert "10.1093/mnras/stz032" not in ref_str  # USE_MASS_DEPENDENT_ZETA
    assert "10.1093/mnras/staa1131" not in ref_str  # USE_MINI_HALOS
    assert "10.1093/mnras/stac185" not in ref_str  # USE_RELATIVE_VELOCITIES
    assert "10.1093/mnras/stac2756" not in ref_str  # PHOTON_CONS
    assert "10.1051/0004-6361/202554951" not in ref_str  # DISCRETE_HALOS
    assert "10.48550/arXiv.2601.14360" not in ref_str  # LYA_MULTIPLE_SCATTERING


class TestRecursiveDifference:
    """Tests of the recursive_difference function."""

    def test_b_empty(self):
        """Test if b is empty in a - b gives a."""
        a = {"a": 1, "b": 2}
        aa = recursive_difference(a, {})

        assert a == aa

    def test_a_empty(self):
        """Test that {} - b == {}."""
        a = {}
        b = {"a": 1, "b": 2}
        aa = recursive_difference(a, b)
        assert len(aa) == 0

    def test_disjoint(self):
        """Test that a - b == a if b is disjoint with a."""
        a = {"a": 1}
        b = {"b": 2}
        assert recursive_difference(a, b) == a

    def test_a_recursive_b_not_recursive(self):
        """Test where a is recursive but b is not."""
        a = {"a": {"a": 1, "b": 2}, "b": 1}
        b = {"a": 1, "b": 2}

        assert recursive_difference(a, b) == a

    def test_a_and_b_recurse(self):
        """Test where both a and b are recursive."""
        a = {"a": {"a": 1, "b": 2}, "b": 1}
        b = a

        assert recursive_difference(a, b) == {}

    def test_comparison_rules(self):
        """Test that passing cmprules works as expected."""
        a = {"a": np.zeros(10)}
        b = {"a": np.zeros(10)}

        with pytest.raises(ValueError, match="The truth value of an array"):
            recursive_difference(a, b)

        cmprules = {np.ndarray: lambda x, y: np.allclose(x, y)}
        assert recursive_difference(a, b, cmprules=cmprules) == {}
