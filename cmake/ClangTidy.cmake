include(CMakeParseArguments)

find_program(CLANG_TIDY_EXE 
    NAMES 
        clang-tidy
        clang-tidy-3.9
        clang-tidy-3.8
        clang-tidy-3.7
        clang-tidy-3.6
        clang-tidy-3.5
    PATHS
        /usr/local/opt/llvm/bin
)

if( NOT CLANG_TIDY_EXE )
    message( STATUS "Clang tidy not found" )
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

macro(enable_clang_tidy)
    set(options ANALYZE_TEMPORARY_DTORS)
    set(oneValueArgs)
    set(multiValueArgs CHECKS ERRORS EXTRA_ARGS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    string(REPLACE ";" "," CLANG_TIDY_CHECKS "${PARSE_CHECKS}")
    string(REPLACE ";" "," CLANG_TIDY_ERRORS "${PARSE_ERRORS}")
    string(REPLACE ";" " " CLANG_TIDY_EXTRA_ARGS "${PARSE_EXTRA_ARGS}")
    
    message(STATUS "Clang tidy checks: ${CLANG_TIDY_CHECKS}")

    if (${PARSE_ANALYZE_TEMPORARY_DTORS})
        set(CLANG_TIDY_ANALYZE_TEMPORARY_DTORS "-analyze-temporary-dtors")
    endif()

    set(CLANG_TIDY_COMMAND 
        ${CLANG_TIDY_EXE} 
        -p ${CMAKE_BINARY_DIR} 
        -checks='${CLANG_TIDY_CHECKS}'
        -warnings-as-errors='${CLANG_TIDY_ERRORS}'
        -extra-arg='${CLANG_TIDY_EXTRA_ARGS}'
        ${CLANG_TIDY_ANALYZE_TEMPORARY_DTORS}
    )
    add_custom_target(tidy)
endmacro()

function(clang_tidy_check TARGET)
    get_target_property(SOURCES ${TARGET} SOURCES)
    add_custom_target(tidy-${TARGET}
        COMMAND ${CLANG_TIDY_COMMAND} ${SOURCES}
        # TODO: Use generator expressions instead
        # COMMAND ${CLANG_TIDY_COMMAND} $<TARGET_PROPERTY:${TARGET},SOURCES>
        # COMMAND ${CLANG_TIDY_COMMAND} $<JOIN:$<TARGET_PROPERTY:${TARGET},SOURCES>, >
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "clang-tidy: Running clang-tidy on target ${TARGET}..."
    )
    add_dependencies(tidy tidy-${TARGET})
endfunction()

