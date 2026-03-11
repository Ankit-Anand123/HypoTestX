# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the hypotestx package importable from the repo root
sys.path.insert(0, os.path.abspath("../.."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "HypoTestX"
copyright = "2026, Ankit"
author = "Ankit"
release = "1.0.6"
version = "1.0.6"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
]

autosummary_generate = True

# ---------------------------------------------------------------------------
# autodoc defaults
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
    "member-order": "bysource",
}

# ---------------------------------------------------------------------------
# Napoleon (Google / NumPy docstring support)
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

# ---------------------------------------------------------------------------
# HTML theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "HypoTestX"
html_short_title = "HypoTestX"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
