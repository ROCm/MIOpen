include(CMakeParseArguments)

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
    set(oneValueArgs HTML_DIR)
    set(multiValueArgs VARS TEMPLATE_VARS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(ADDITIONAL_ARGS)
    foreach(VAR ${PARSE_VARS})
        list(APPEND ADDITIONAL_ARGS "-D ${VAR}")
    endforeach()
    foreach(VAR ${PARSE_TEMPLATE_VARS})
        list(APPEND ADDITIONAL_ARGS "-A ${VAR}")
    endforeach()

    if(PARSE_HTML_DIR)
        get_filename_component(HTML_DIR ${PARSE_HTML_DIR} ABSOLUTE)
        set(SPHINX_HTML_DIR ${HTML_DIR} CACHE PATH "Path to html output")
    else()
        set(SPHINX_HTML_DIR ${SPHINX_DEFAULT_HTML_DIR} CACHE PATH "Path to html output")
    endif()

    add_custom_target(doc
        ${SPHINX_EXECUTABLE}
        -b html
        -d "${SPHINX_CACHE_DIR}"
        ${ADDITIONAL_ARGS}
        "${SRC_DIR}"
        "${SPHINX_HTML_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building HTML documentation with Sphinx")
endfunction()

