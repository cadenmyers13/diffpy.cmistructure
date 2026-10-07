import importlib.resources
import json
import logging
from functools import lru_cache
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    yield tmp_path


# diffpy.structure
@lru_cache()
def has_diffpy_structure():
    try:
        import diffpy.structure as m

        del m
        return True
    except ImportError:
        logger.warning(
            "Cannot import diffpy.structure, Structure tests skipped."
        )
        return False


# pyobjcryst
@lru_cache()
def has_pyobjcryst():
    try:
        import pyobjcryst as m

        del m
        return True
    except ImportError:
        logger.warning("Cannot import pyobjcryst, pyobjcryst tests skipped.")
        return False


@pytest.fixture(scope="session")
def diffpy_structure_available():
    return has_diffpy_structure()


@pytest.fixture(scope="session")
def pyobjcryst_available():
    return has_pyobjcryst()


@pytest.fixture(scope="session")
def datafile():
    """Fixture to load a test data file from the testdata package
    directory."""

    def _datafile(filename):
        return importlib.resources.files("tests.testdata").joinpath(filename)

    return _datafile
