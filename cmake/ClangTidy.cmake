include(CMakeParseArguments)

find_program(CLANG_TIDY_EXE 
    NAMES 
        clang-tidy
        clang-tidy-3.9
        clang-tidy-3.8
        clang-tidy-3.7
        clang-tidy-3.6
        clang-tidy-3.5
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

macro(enable_clang_tidy)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs CHECKS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    message(STATUS "Clang tidy checks: ${PARSE_CHECKS}")
    set(CLANG_TIDY_COMMAND ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR} -checks='${PARSE_CHECKS}')
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

