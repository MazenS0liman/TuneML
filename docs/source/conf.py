# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the `tuneml` package importable for autodoc.
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "TuneML"
copyright = "2026, Mazen Soliman"
author = "Mazen Soliman"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_multiversion",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc -----------------------------------------------------------------
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_logo = "_static/images/tune-ml.png"
html_favicon = "_static/images/tune-ml.png"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# -- Copy button -------------------------------------------------------------
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
