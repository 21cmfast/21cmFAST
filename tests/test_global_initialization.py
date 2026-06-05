"""Test the GlobalInitializationManager class."""

import pytest

from py21cmfast.drivers._global_initialization import GlobalInitializationManager


def test_global_initialization_is_singleton():
    """Test that the GlobalInitializationManager is a singleton."""
    with pytest.raises(
        RuntimeError,
        match="GlobalInitializationManager is a singleton and has already been instantiated",
    ):
        GlobalInitializationManager()
