import importlib.resources
import io
import json
import sys
from pathlib import Path

import pytest

from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet
from diffpy.structure import Structure


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


@pytest.fixture(scope="session")
def datafile():
    """Fixture to load a test data file from the testdata package
    directory."""

    def _datafile(filename):
        return importlib.resources.files("tests.testdata").joinpath(filename)

    return _datafile


@pytest.fixture(scope="session")
def capturestdout():
    def _capturestdout(f, *args, **kwargs):
        """Capture the standard output from a call of function f."""
        savestdout = sys.stdout
        fp = io.StringIO()
        try:
            sys.stdout = fp
            f(*args, **kwargs)
        finally:
            sys.stdout = savestdout
        return fp.getvalue()

    return _capturestdout


class GeneratorStandIn:
    """Stand in for a generator, such as a PDFGenerator, that holds a
    structure parameter set as ``.phase``."""

    def __init__(self, phase):
        self.phase = phase


@pytest.fixture
def lamno3_generator(datafile):
    structure = Structure()
    structure.read(datafile("LaMnO3.stru"))
    parameter_set = DiffpyStructureParSet("LaMnO3", structure)
    generator = GeneratorStandIn(parameter_set)
    return generator
