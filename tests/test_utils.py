"""Tests of miscellaneous utility functions not part of the main 21cmFAST simulation modules."""

from py21cmfast import InputParameters
from py21cmfast.utils import show_references


def test_ref_printing():
    inputs = InputParameters.from_template("latest", random_seed=1234)
    ref_str = show_references(inputs, print_to_stdout=False)

    assert "2011MNRAS.411..955M" in ref_str
    assert "10.21105/joss.02582" in ref_str
    assert "10.1093/mnras/stz032" in ref_str
    assert "10.1093/mnras/staa1131" not in ref_str
    assert "10.1093/mnras/stac185" not in ref_str
    assert "10.48550/arXiv.2504.17254" not in ref_str

    inputs = InputParameters.from_template("mini-dhalos", random_seed=1234)
    ref_str = show_references(inputs, print_to_stdout=False)

    assert "2011MNRAS.411..955M" in ref_str
    assert "10.21105/joss.02582" in ref_str
    assert "10.1093/mnras/stz032" in ref_str
    assert "10.1093/mnras/staa1131" in ref_str
    assert "10.1093/mnras/stac185" in ref_str
    assert "10.48550/arXiv.2504.17254" in ref_str

    inputs = InputParameters.from_template("const-zeta", random_seed=1234)
    ref_str = show_references(inputs, print_to_stdout=True)

    assert "2011MNRAS.411..955M" in ref_str
    assert "10.21105/joss.02582" in ref_str
    assert "10.1093/mnras/stz032" not in ref_str
    assert "10.1093/mnras/staa1131" not in ref_str
    assert "10.1093/mnras/stac185" not in ref_str
    assert "10.48550/arXiv.2504.17254" not in ref_str
