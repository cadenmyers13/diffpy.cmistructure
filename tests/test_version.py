"""Unit tests for __version__.py."""

import diffpy.cmistructure  # noqa


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(diffpy.cmistructure, "__version__")
    assert diffpy.cmistructure.__version__ != "0.0.0"
