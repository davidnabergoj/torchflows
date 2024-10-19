from pathlib import Path

try:
    import toml

    pyproject_toml_file = Path(__file__).parents[1] / "pyproject.toml"
    data = toml.load(pyproject_toml_file)
    __version__ = data["project"]["version"]
except ImportError:
    import importlib.metadata

    __version__ = importlib.metadata.version("torchflows")
