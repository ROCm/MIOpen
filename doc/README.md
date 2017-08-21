
# MIOpen, AMD's Machine Intelligence Library

This folder contains sources for building the documentation. Online documentation can be found [here](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/).

## Building the Documentation

HTML and PDF documentation can be built using:

`cmake --build . --config Release --target doc` **OR** `make doc`

This will build a local searchable web site inside the ./MIOpen/doc/html folder and a PDF document inside the ./MIOpen/doc/pdf folder.

Documentation is built using generated using [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html) and should be installed separately.

HTML and PDFs are generated using [Sphinx](http://www.sphinx-doc.org/en/stable/index.html) and [Breathe](https://breathe.readthedocs.io/en/latest/), with the [ReadTheDocs theme](https://github.com/rtfd/sphinx_rtd_theme).

Requirements for both Sphinx, Breathe, and the ReadTheDocs theme can be filled for these in the MIOpen/doc folder:

`pip install -r ./requirements.txt`

Depending on your setup `sudo` may be required for the pip install.
