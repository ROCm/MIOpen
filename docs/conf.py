# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from rocm_docs import ROCmDocs

docs_core = ROCmDocs("MIOpen Documentation")
docs_core.run_doxygen()
docs_core.setup()

os.system("sed -e 's/MIOPEN_EXPORT //g' ../include/miopen/* > .doxygen/")

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
