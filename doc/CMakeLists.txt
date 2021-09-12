################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
################################################################################

include(SphinxDoc)
include(DoxygenDoc)


set(DOXYGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen/output)
add_doxygen_doc(
    OUTPUT_DIRECTORY ${DOXYGEN_OUTPUT}
    INPUT 
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    # INCLUDE_PATH
    #     ${CMAKE_CURRENT_BINARY_DIR}/../include
    #     ${CMAKE_CURRENT_SOURCE_DIR}/../include
    MACRO_EXPANSION YES
    RECURSIVE YES
    GENERATE_XML YES
    GENERATE_LATEX YES
    USE_PDFLATEX YES
)

# Build HTML documentation with Sphinx #
# Sphinx pre-build steps
add_custom_target(prebuild_sphinx
    DEPENDS copyover_installreadme copyover_driverreadme
)
# Copy README.md as install.md
add_custom_target(copyover_installreadme
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../README.md ${CMAKE_CURRENT_SOURCE_DIR}/src/install.md
    # Replace section title
    COMMAND sed -e '0,/MIOpen/ s/MIOpen/Build and Install Instructions/' -i ${CMAKE_CURRENT_SOURCE_DIR}/src/install.md
    COMMENT "Copying over README.md to docs folder as install.md."
)
clean_doc_output(${CMAKE_CURRENT_SOURCE_DIR}/src/install.md)
# Copy driver/README.md as driver.md
add_custom_target(copyover_driverreadme
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../driver/README.md ${CMAKE_CURRENT_SOURCE_DIR}/src/driver.md
    COMMENT "Copying over MIOpenDriver README.md to docs folder as driver.md."
)
clean_doc_output(${CMAKE_CURRENT_SOURCE_DIR}/src/driver.md)

# Build
add_sphinx_doc(src
    BUILDER html
    OUTPUT_DIR html
    VARS
        breathe_projects.proj=${DOXYGEN_OUTPUT}/xml
        breathe_default_project=proj
    DEPENDS doxygen prebuild_sphinx
)

# Sphinx post-build steps
add_custom_target(postbuild_sphinx_html
    DEPENDS delete_miopenexporthtml
)
add_custom_target(delete_miopenexporthtml
    COMMAND sed -e s/MIOPEN_EXPORT// -i ${CMAKE_CURRENT_SOURCE_DIR}/html/*.html
    COMMENT "Removing MIOPEN_EXPORT from html document. ${CMAKE_CURRENT_SOURCE_DIR}/html/*.html"
    DEPENDS sphinx-HTML
)

# Target for HTML documentation
mark_as_doc(postbuild_sphinx_html)

# Build PDF documentation with Sphinx #
find_package(LATEX)
if(LATEX_FOUND)
    # Build
    add_sphinx_doc(src
        BUILDER latex
        OUTPUT_DIR pdf
        VARS
            breathe_projects.proj=${DOXYGEN_OUTPUT}/xml
            breathe_default_project=proj
        DEPENDS doxygen prebuild_sphinx
    )

    # Sphinx post-build steps
    add_custom_target(postbuild_sphinx_latex
        DEPENDS delete_export delete_slashmiopen
    )

    add_custom_target(delete_export
        COMMAND sed -e s/_EXPORT// -i ${CMAKE_CURRENT_SOURCE_DIR}/pdf/miopen.tex
        COMMENT "Removing MIOPEN_EXPORT from latex document. ${CMAKE_CURRENT_SOURCE_DIR}/pdf/miopen.tex"
        DEPENDS sphinx-LATEX
    )

    add_custom_target(delete_slashmiopen
        COMMAND sed -e s/sret{MIOPEN\\/sret{/ -i ${CMAKE_CURRENT_SOURCE_DIR}/pdf/miopen.tex
        COMMENT "Removing MIOPEN_EXPORT from latex document. ${CMAKE_CURRENT_SOURCE_DIR}/pdf/miopen.tex"
        DEPENDS sphinx-LATEX delete_export
    )

    # Target for PDF documentation
    add_custom_target(build_pdf
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pdf
        COMMAND make
        COMMENT "Building pdf documentation"
        DEPENDS postbuild_sphinx_latex
    )
    mark_as_doc(build_pdf)

else()
    message("Latex builder not found. Latex builder is required only for building the PDF documentation for MIOpen and is not necessary for building the library, or any other components. To build PDF documentation run make in ${CMAKE_CURRENT_SOURCE_DIR}/pdf, once a latex builder is installed.")
endif()
