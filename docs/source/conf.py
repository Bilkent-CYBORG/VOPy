# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
autodoc_mock_imports = [
    "cvxpy",
    "botorch",
    "clarabel",
    "gpytorch",
    "scipy",
    "matplotlib",
    "torch",
    "sklearn",
    "numpy",
]


# - Copy over examples folder to docs/source
# This makes it so that nbsphinx properly loads the notebook images
examples_source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
examples_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "examples"))

if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
os.mkdir(examples_dest)

# Include examples in documentation
# This adds a lot of time to the doc buiod; to bypass use the environment variable
# SKIP_EXAMPLES=true
for root, dirs, files in os.walk(examples_source):
    for dr in dirs:
        os.mkdir(os.path.join(root.replace(examples_source, examples_dest), dr))
    for fil in files:
        if os.path.splitext(fil)[1] in [".ipynb", ".md", ".rst"]:
            source_filename = os.path.join(root, fil)
            dest_filename = source_filename.replace(examples_source, examples_dest)

            # If we're skipping examples, put a dummy file in place
            if os.getenv("SKIP_EXAMPLES"):
                if dest_filename.endswith("index.rst"):
                    shutil.copyfile(source_filename, dest_filename)
                else:
                    with open(os.path.splitext(dest_filename)[0] + ".rst", "w") as f:
                        basename = os.path.splitext(os.path.basename(dest_filename))[0]
                        f.write(f"{basename}\n" + "=" * 80)

            # Otherwise, copy over the real example files
            else:
                shutil.copyfile(source_filename, dest_filename)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "VOPy"
copyright = "2024, Cahit Yildirim"
author = "Cahit Yildirim"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "notfound.extension",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

primary_domain = "py"  # Set the primary domain as Python globally to omit `py` prefix in docs.

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_js_files = [
    "hide_code_cells.js",
]
