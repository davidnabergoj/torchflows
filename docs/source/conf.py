# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath('../../'))


def _get_version():
    # The package version is derived from git tags by setuptools-scm
    try:
        from importlib.metadata import version as package_version
        return package_version('torchflows')
    except Exception:
        pass
    try:
        from setuptools_scm import get_version
        return get_version(root='../..', relative_to=__file__)
    except Exception:
        return 'unknown'


release = _get_version()
version = '.'.join(release.split('.')[:2])

project = 'Torchflows'
copyright = '2024-2026, David Nabergoj'
author = 'David Nabergoj'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
    'sphinx.ext.viewcode',
]

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
epub_show_urls = 'footnote'
