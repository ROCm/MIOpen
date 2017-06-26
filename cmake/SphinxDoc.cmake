include(CMakeParseArguments)
include(MainDoc)
include(DoxygenDoc)

find_program(SPHINX_EXECUTABLE NAMES sphinx-build
    HINTS
    $ENV{SPHINX_DIR}
    PATH_SUFFIXES bin
    DOC "Sphinx documentation generator"
)

mark_as_advanced(SPHINX_EXECUTABLE)

set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/sphinx/_build")
 
# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/sphinx/_doctrees")
 
# HTML output directory
set(SPHINX_DEFAULT_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/sphinx/html")
function(add_sphinx_doc SRC_DIR)
    set(options)
    set(oneValueArgs BUILDER OUTPUT_DIR)
    set(multiValueArgs DEPENDS VARS TEMPLATE_VARS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    string(TOUPPER ${PARSE_BUILDER} BUILDER)

    set(ADDITIONAL_ARGS)
    foreach(VAR ${PARSE_VARS})
        list(APPEND ADDITIONAL_ARGS -D ${VAR})
    endforeach()
    foreach(VAR ${PARSE_TEMPLATE_VARS})
        list(APPEND ADDITIONAL_ARGS -A ${VAR})
    endforeach()

    if(PARSE_OUTPUT_DIR)
        get_filename_component(OUTPUT_DIR ${PARSE_OUTPUT_DIR} ABSOLUTE)
        set(SPHINX_${BUILDER}_DIR ${OUTPUT_DIR} CACHE PATH "Path to ${PARSE_BUILDER} output")
    else()
        set(SPHINX_${BUILDER}_DIR "${CMAKE_CURRENT_BINARY_DIR}/sphinx/${PARSE_BUILDER}" CACHE PATH "Path to ${PARSE_BUILDER} output")
    endif()

    add_custom_target(sphinx-${BUILDER}
        ${SPHINX_EXECUTABLE}
        -b ${PARSE_BUILDER}
        -d "${SPHINX_CACHE_DIR}"
        ${ADDITIONAL_ARGS}
        "${SRC_DIR}"
        "${SPHINX_${BUILDER}_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building ${PARSE_BUILDER} documentation with Sphinx"
    )
    clean_doc_output(${SPHINX_${BUILDER}_DIR})
    clean_doc_output(${SPHINX_CACHE_DIR})
    clean_doc_output(${BINARY_BUILD_DIR})
    mark_as_doc(sphinx-${BUILDER})
    if(PARSE_DEPENDS)
        add_dependencies(sphinx-${BUILDER} ${PARSE_DEPENDS})
    endif()

endfunction()


