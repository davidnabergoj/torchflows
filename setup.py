from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="normalizing-flows",
    version="0.1",
    description="Modern normalizing flows in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidnabergoj/normalizing-flows",
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
    packages=["normalizing_flows"],
    python_requires=">=3.7, <4",
    install_requires=[
        "torch>=2.0.1",
        "numpy",
        "torchdiffeq",
        "tqdm"
    ],
    project_urls={
        "Bug Reports": "https://github.com/davidnabergoj/normalizing-flows/issues",
    },
)
