# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
shutil.copy2('../../CHANGELOG.md','./')
shutil.copy2('../../RELEASE.md','./')

from rocm_docs import ROCmDocs

docs_core = ROCmDocs("ROCm Documentation")
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
