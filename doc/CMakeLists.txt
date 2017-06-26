
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
    add_dependencies(delete_export sphinx-LATEX)
    mark_as_doc(delete_export)


    add_custom_target(delete_slashmiopen
        COMMAND sed s/MIOPEN\\// -i ${CMAKE_CURRENT_SOURCE_DIR}/pdf/miopen.tex
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
    message("Latex builder not found. To build PDF documentation run make in ${CMAKE_CURRENT_SOURCE_DIR}/pdf, once a latex builder is installed.")
endif()
