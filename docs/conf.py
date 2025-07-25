"""Configuration options for the docs."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).absolute().parent.parent / "src"))


class Mock(MagicMock):
    """Make a Mock so that a package doesn't have to actually exist."""

    @classmethod
    def __getattr__(cls, name):
        """Get stuff."""
        return MagicMock()


MOCK_MODULES = [
    "py21cmfast.c_21cmfast",
    "click",
    "tqdm",
    "pyyaml",
    "h5py",
    "cached_property",
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# Get the version by running from shell -- this uses the git repo info, rather than
# requiring it to be installed.
out = subprocess.run(["python", "setup.py", "--version"], capture_output=True)

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
project = "21cmFAST"
year = "2020"
author = "The 21cmFAST collaboration"
copyright = f"{year}, {author}"
version = release = (
    out.stdout.decode().rstrip()
)  # Get it from `python setup.py --version`
templates_path = ["templates"]

pygments_style = "trac"
extlinks = {
    "issue": ("https://github.com/21cmFAST/21cmFAST/issues/%s", "#"),
    "pr": ("https://github.com/21cmFAST/21cmFAST/pull/%s", "PR #"),
}
# # on_rtd is whether we are on readthedocs.org
# on_rtd = os.environ.get("READTHEDOCS", None) == "True"

html_theme = "furo"

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
