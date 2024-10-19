from setuptools import setup, find_packages
import pathlib
import sys
import importlib.metadata

root = pathlib.Path(__file__).parent.resolve()
sys.path.append(root.as_posix())

version = importlib.metadata.version('torchflows')

long_description = (root / "README.md").read_text(encoding="utf-8")

setup(
    name="Torchflows",
    version=f"{version}",
    description="Modern normalizing flows in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidnabergoj/torchflows",
    author="David Nabergoj",
    author_email="david.nabergoj@fri.uni-lj.si",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=[
        "python",
        "machine-learning",
        "pytorch",
        "generative-model",
        "sampling",
        "density-estimation",
        "normalizing-flow"
    ],
    packages=find_packages(exclude=["test"]),
    python_requires=">=3.7, <4",
    install_requires=[
        "torch>=2.0.1",
        "numpy",
        "torchdiffeq",
        "tqdm"
    ],
    project_urls={
        "Bug Reports": "https://github.com/davidnabergoj/torchflows/issues",
    },
)
