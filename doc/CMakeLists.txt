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
add_sphinx_doc(src 
    BUILDER html 
    OUTPUT_DIR html 
    VARS 
        breathe_projects.proj=${DOXYGEN_OUTPUT}/xml
        breathe_default_project=proj
    DEPENDS doxygen
)


    add_custom_target(copyover_installreadme
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../README.md ${CMAKE_CURRENT_SOURCE_DIR}/src/install.md
        COMMENT "Copying over README.md to docs folder as install.md."
    )
    add_dependencies(copyover_installreadme doxygen)
    mark_as_doc(copyover_installreadme)

    add_custom_target(change_section_title_install
        COMMAND sed -e '0,/MIOpen/ s/MIOpen/Build and Install Instructions/' -i ${CMAKE_CURRENT_SOURCE_DIR}/src/install.md
        COMMENT "Replacing section title."
    )
    add_dependencies(change_section_title_install copyover_installreadme)
    mark_as_doc(change_section_title_install)

    add_custom_target(copyover_driverreadme
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../driver/README.md ${CMAKE_CURRENT_SOURCE_DIR}/src/driver.md
        COMMENT "Copying over MIOpenDriver README.md to docs folder as driver.md."
    )
    add_dependencies(copyover_driverreadme copyover_installreadme)
    mark_as_doc(copyover_driverreadme)



add_custom_target(delete_miopenexporthtml
    COMMAND sed -e s/MIOPEN_EXPORT// -i ${CMAKE_CURRENT_SOURCE_DIR}/html/*.html
    COMMENT "Removing MIOPEN_EXPORT from html document. ${CMAKE_CURRENT_SOURCE_DIR}/html/*.html"
)
add_dependencies(delete_miopenexporthtml sphinx-HTML copyover_driverreadme)
mark_as_doc(delete_miopenexporthtml)


find_package(LATEX)
if(LATEX_FOUND)
    add_sphinx_doc(src 
        BUILDER latex
        OUTPUT_DIR pdf
        VARS 
            breathe_projects.proj=${DOXYGEN_OUTPUT}/xml
            breathe_default_project=proj
        DEPENDS doxygen
    )

    add_custom_target(delete_export
        COMMAND sed -e s/_EXPORT// -i ${CMAKE_CURRENT_SOURCE_DIR}/pdf/miopen.tex
        COMMENT "Removing MIOPEN_EXPORT from latex document. ${CMAKE_CURRENT_SOURCE_DIR}/pdf/miopen.tex"
    )
    add_dependencies(delete_export sphinx-LATEX copyover_driverreadme)
    mark_as_doc(delete_export)


    add_custom_target(delete_slashmiopen
        COMMAND sed -e s/sret{MIOPEN\\/sret{/ -i ${CMAKE_CURRENT_SOURCE_DIR}/pdf/miopen.tex
        COMMENT "Removing MIOPEN_EXPORT from latex document. ${CMAKE_CURRENT_SOURCE_DIR}/pdf/miopen.tex"
    )
    add_dependencies(delete_slashmiopen delete_export)
    mark_as_doc(delete_slashmiopen)

    add_custom_target(build_pdf
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pdf
        COMMAND make
        COMMENT "Building pdf documentation"
    )
    add_dependencies(build_pdf delete_slashmiopen)
    mark_as_doc(build_pdf)

else()
    message("Latex builder not found. Latex builder is required only for building the PDF documentation for MIOpen and is not necessary for building the library, or any other components. To build PDF documentation run make in ${CMAKE_CURRENT_SOURCE_DIR}/pdf, once a latex builder is installed.")
endif()
