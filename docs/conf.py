# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]
if os.getenv("SPELLCHECK"):
    extensions += ("sphinxcontrib.spelling",)
    spelling_show_suggestions = True
    spelling_lang = "en_US"

autosectionlabel_prefix_document = True

autosummary_generate = True
numpydoc_show_class_members = False

source_suffix = ".rst"
master_doc = "index"
project = "21CMMC"
year = "2018"
author = "Brad Greig and Steven Murray"
copyright = "{0}, {1}".format(year, author)
version = release = "0.1.0"
templates_path = ["templates"]

pygments_style = "trac"
extlinks = {
    "issue": ("https://github.com/BradGreig/21CMMC/issues/%s", "#"),
    "pr": ("https://github.com/BradGreig/21CMMC/pull/%s", "PR #"),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = "sphinx_rtd_theme"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {"**": ["searchbox.html", "globaltoc.html", "sourcelink.html"]}
html_short_title = "%s-%s" % (project, version)

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
