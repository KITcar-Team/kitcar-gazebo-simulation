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
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "KITcar Simulation"
copyright = "2021, KITcar"
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
    "autodocsumm",
]

add_module_names = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

html_extra_path = [
    "content/tutorials/resources/rviz_master_launch.mp4",
    "content/talks/ci_tests/resources/state_machine_output.mp4",
    "content/talks/ci_tests/resources/drive_test_obstacles.mp4",
    "content/talks/ci_tests/resources/drive_test_referee.mp4",
    "content/talks/ci_tests/resources/drive_test_failure.mp4",
    "content/talks/ci_tests/resources/speaker_output.mp4",
    "content/tutorials/resources/parking_drive_test.mp4",
    "content/tutorials/resources/birds_view_default_road.mp4",
    "content/tutorials/resources/gan_default_road.mp4",
    "content/tutorials/resources/camera_default_road.mp4",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# Set lineos style to table
# See also here:
# sphinx-doc.org/en/master/usage/configuration.html#confval-html_codeblock_linenos_style
html_codeblock_linenos_style = "table"

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
