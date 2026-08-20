from importlib.metadata import version

import grit


def test_package_imports_with_installed_version() -> None:
    assert grit.__version__ == version("grit-research")
