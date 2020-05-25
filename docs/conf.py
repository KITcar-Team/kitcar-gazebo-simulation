#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# KITcar Simulation documentation build configuration file, created by
# sphinx-quickstart on Sun Feb 16 00:00:20 2020.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
from unittest.mock import Mock
from typing import List

sys.path.insert(0, os.path.abspath(".."))


# Prevent all of these libraries to be installed just for the configuration
MOCK_MODULES = [
    "yaml",
    "PIL",
    "pyxb",
    "pyxb.binding",
    "pyxb.binding.saxer",
    "pyxb.utils",
    "pyxb.utils.utility",
    "pyxb.utils.domutils",
    "pyxb.utils.six",
    "pyxb.binding",
    "pyxb.binding.datatypes",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()


def create_run_apidoc(root: str, exclude: List[str] = None):
    """Create the run apidoc function which is used by better-apidoc to build .rst files from modules and packages."""

    def run_apidoc(app):
        """Generate API documentation"""

        apidoc_args = [
            "better-apidoc",
            "-a",
            "-M",
            "-t",
            os.path.join(".", "templates"),
            "--force",
            "--no-toc",
            "--separate",
            "--ext-autodoc",
            "--ext-coverage",
            "-o",
            os.path.join(".", "content", "_source_files/"),
            root,
        ]

        if exclude:
            apidoc_args.extend(exclude)

        import better_apidoc

        better_apidoc.APP = app
        better_apidoc.main(apidoc_args)

    return run_apidoc


run_apidoc = create_run_apidoc(
    root="../simulation/",
    exclude=[
        # fmt: off
        "../*setup*",
        "../*gazebo-renderer*",
        "../*schema*",
        "../*machine_learning*",
        # fmt: on
    ],
)


def setup(app):
    app.connect("builder-inited", run_apidoc)


# -- Project information -----------------------------------------------------

project = "KITcar Simulation"
copyright = "2020, KITcar"
author = "KITcar"

# The master toctree document.
# master_doc = "index"
# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "1.0.0"
# The full version, including alpha/beta/rc tags.
release = "1.0.0"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.graphviz",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.programoutput",
    "sphinx-prompt",
    "breathe",
]

add_module_names = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

html_extra_path = ["content/tutorials/resources/rviz_master_launch.mp4"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# from crate.theme.rtd.conf.crate_server import *
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Breathe settings
os.chdir("..")
breathe_projects_source = {
    "kitcar-gazebo-simulation": (
        "..",
        # Add all header files in simulation/* to project!
        # (Exclude devel)
        [str(file) for file in Path("simulation").rglob("*.h") if "devel" not in str(file)],
    ),
}
os.chdir("docs")
