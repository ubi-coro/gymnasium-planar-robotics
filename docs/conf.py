# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
import datetime

# sphinx-autobuild: source code directory
sys.path.insert(0, os.path.abspath('..'))

# version
with open(os.path.join(os.path.dirname(__file__), '../gymnasium_planar_robotics', '__init__.py')) as f:
    content_str = f.read()
    version_start_idx = content_str.find('__version__') + len('__version__ = ') + 1
    version_stop_idx = version_start_idx + content_str[version_start_idx:].find('\n')
    __version__ = content_str[version_start_idx : version_stop_idx - 1]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Gymnasium-Planar-Robotics'
year = datetime.date.today().year
if year == 2024:
    copyright = '2024, Lara Bergmann'
else:
    copyright = f'2024-{year}, Lara Bergmann'
author = 'Lara Bergmann'
release = f'v{__version__}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # include documentation from docstrings
    'sphinxcontrib.spelling',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

spelling_lang = 'en_US'
spelling_show_suggestions = True
spelling_warning = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['css/custom_theme.css']
html_title = 'Gymnasium-Planar-Robotics\nDocumentation'

html_theme_options = {
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/ubi-coro/gymnasium-planar-robotics',
            'icon': 'fa-brands fa-github',
        },
    ],
    'primary_sidebar_end': ['sidebar-ethical-ads'],
    'show_toc_level': 1,
    'collapse_navigation': False,
    'navigation_depth': 4,
    'secondary_sidebar_items': ['page-toc'],
    'navbar_center': ['navbar-nav'],
    'navbar_persistent': ['search-button'],
    'navbar_align': 'right',
    'pygment_light_style': 'default',
    'pygment_dark_style': 'monokai',
}
