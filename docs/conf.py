"""Configuration options for the docs."""

import os

# import sys
# from pathlib import Path
# from unittest.mock import MagicMock
# Import the package because if the import breaks, this gives a nice
# stack trace, whereas if it breaks inside the autodoc step, it's harder to debug.
import py21cmfast

extensions = [
    "sphinx.ext.autodoc.typehints",
#    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "autoapi.extension",
    "numpydoc",
    "nbsphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_console_highlighting",
]
if os.getenv("SPELLCHECK"):
    extensions += ("sphinxcontrib.spelling",)
    spelling_show_suggestions = True
    spelling_lang = "en_US"

autosectionlabel_prefix_document = True
# autodoc_use_legacy_class_based = True

# Auto-API settings
autoapi_options=[ 
    'members', 
    'undoc-members', 
    'show-inheritance', 
    'show-module-summary', 
    #'special-members', 
    'imported-members',
    'inherited-members',
]
autoapi_dirs = ["../src/py21cmfast"]
autoapi_add_toctree_entry = False  # We add it ourselves in index.rst
autoapi_python_class_content = "init"
autoapi_member_order = "groupwise"
autoapi_own_page_level = "class"
autoapi_keep_files = True
autodoc_typehints = 'both'
autoapi_template_dir = 'templates/autoapi'

autosummary_generate = False
numpydoc_show_class_members = False

source_suffix = ".rst"
master_doc = "index"
project = "21cmFAST"
year = "2020"
author = "The 21cmFAST collaboration"
copyright = f"{year}, {author}"
version = release = py21cmfast.__version__
templates_path = ["templates"]

pygments_style = "trac"
extlinks = {
    "issue": ("https://github.com/21cmFAST/21cmFAST/issues/%s", "#"),
    "pr": ("https://github.com/21cmFAST/21cmFAST/pull/%s", "PR #"),
}
# # on_rtd is whether we are on readthedocs.org
# on_rtd = os.environ.get("READTHEDOCS", None) == "True"

html_theme = "furo"
html_style = "css/custom.css"
html_logo = "images/Logo_square_transparent.png"

# These folders are copied to the documentation's HTML output
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

html_theme_options = {
    "sidebar_hide_name": True,
}
html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
html_short_title = f"{project}-{version}"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

mathjax_path = (
    "http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "templates",
    "**.ipynb_checkpoints",
]
