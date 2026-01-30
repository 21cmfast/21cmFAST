"""Configuration options for the docs."""

import os
from sphinx.ext.autodoc import Documenter
from inspect import signature

# import sys
# from pathlib import Path
# from unittest.mock import MagicMock
# Import the package because if the import breaks, this gives a nice
# stack trace, whereas if it breaks inside the autodoc step, it's harder to debug.
import py21cmfast

class ClassDecoratedDocumenter(Documenter):
    """Document task definitions."""

    objtype = 'func'
    member_order = 11
    priority = 60000  # run before FunctionDocumenter

    
    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        raise ValueError(">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< HEY HEY !!!")
        
        return getattr(member, '__wrapped__')

        
    def format_args(self):
        wrapped = getattr(self.object, '__wrapped__', None)
        if wrapped is not None:
            sig = signature(wrapped)
            if "self" in sig.parameters or "cls" in sig.parameters:
                sig = sig.replace(parameters=list(sig.parameters.values())[1:])
            return str(sig)
        return ''
    
    def document_members(self, all_members=False):
        pass
        
    def check_module(self):
        # Normally checks if *self.object* is really defined in the module
        # given by *self.modname*. But since functions decorated with the @task
        # decorator are instances living in the celery.local, we have to check
        # the wrapped function instead.
        raise ValueError(">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< HEY HEY !!!")
        print(">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<< HEY HEY !!!")
        wrapped = getattr(self.object, '__wrapped__', None)
        if wrapped and getattr(wrapped, '__module__') == self.modname:
            return True
        return super().check_module()


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
    'special-members', 
    'imported-members',
    'inherited-members',
]
autoapi_dirs = ['../src/py21cmfast']
autoapi_add_toctree_entry = False  # We add it ourselves in index.rst
autoapi_python_class_content = 'init'
autoapi_member_order = 'groupwise'
autoapi_own_page_level = 'class'
autoapi_keep_files = True
autodoc_typehints = 'description'

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
html_style = 'css/custom.css'
html_logo = "images/Logo_square_transparent.png"

# These folders are copied to the documentation's HTML output
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
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

def setup(app):
    """Setup function for Sphinx."""
    print("SETTING UP THE CLASS BASED DECORATOR DOCUMENTER")
    app.add_autodocumenter(ClassDecoratedDocumenter)
    return {
        'parallel_read_safe': True
    }