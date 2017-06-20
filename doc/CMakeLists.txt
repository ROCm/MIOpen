
include(SphinxDoc)
include(DoxygenDoc)

set(DOXYGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen/output/)
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
    AUTOLINK_SUPPORT YES
    SHOW_INCLUDE_FILES YES
    SHOW_FILES YES
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

add_sphinx_doc(src 
    BUILDER latex
    OUTPUT_DIR pdf
    VARS 
        breathe_projects.proj=${DOXYGEN_OUTPUT}/xml
        breathe_default_project=proj
    DEPENDS doxygen
)

