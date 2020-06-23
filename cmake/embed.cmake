    
find_program(EMBED_LD ld)
find_program(EMBED_OBJCOPY objcopy)

function(generate_embed_source EMBED_NAME)
    set(options)
    set(oneValueArgs SRC HEADER)
    set(multiValueArgs OBJECTS SYMBOLS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(EXTERNS)
    set(INIT_KERNELS)

    # TODO: check the lens are the same
    list(LENGTH PARSE_SYMBOLS SYMBOLS_LEN)
    list(LENGTH PARSE_OBJECTS OBJECTS_LEN)
    if(NOT ${SYMBOLS_LEN} EQUAL ${OBJECTS_LEN})
        message(FATAL_ERROR "Symbols and objects dont match: ${SYMBOLS_LEN} != ${OBJECTS_LEN}")
    endif()
    math(EXPR LEN "${SYMBOLS_LEN} - 1")

    foreach(idx RANGE ${LEN})
        list(GET PARSE_SYMBOLS ${idx} SYMBOL)
        list(GET PARSE_OBJECTS ${idx} OBJECT)
    # foreach(SYMBOL ${PARSE_SYMBOLS})
        set(START_SYMBOL "_binary_${SYMBOL}_start")
        set(END_SYMBOL "_binary_${SYMBOL}_end")
        string(APPEND EXTERNS "
            extern const char ${START_SYMBOL}[];
            extern const char ${END_SYMBOL}[];
        ")

        get_filename_component(BASE_NAME "${OBJECT}" NAME)

        string(APPEND INIT_KERNELS "
            { \"${BASE_NAME}\", { ${START_SYMBOL}, ${END_SYMBOL}} },
        ")
    endforeach()

    file(WRITE "${PARSE_HEADER}" "
#include <unordered_map>
const std::unordered_map<std::string, std::pair<const char*,const char*>>& ${EMBED_NAME}();
")

    file(WRITE "${PARSE_SRC}" "
#include <${EMBED_NAME}.hpp>
${EXTERNS}
const std::unordered_map<std::string, std::pair<const char*,const char*>>& ${EMBED_NAME}()
{
    static const std::unordered_map<std::string, std::pair<const char*,const char*>> result = {${INIT_KERNELS}};
    return result;
}
")
endfunction()

function(embed_file OUTPUT_FILE OUTPUT_SYMBOL FILE)
    set(${OUTPUT_FILE} "${FILE}.o" PARENT_SCOPE)
    set(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # Glob is used to compute the relative path
    file(GLOB FILES RELATIVE ${WORKING_DIRECTORY} ${FILE})
    foreach(REL_FILE ${FILES})
        string(MAKE_C_IDENTIFIER "${REL_FILE}" SYMBOL)
        set(${OUTPUT_SYMBOL} ${SYMBOL} PARENT_SCOPE)
        add_custom_command(
            OUTPUT "${FILE}.o"
            COMMAND ${EMBED_LD} -r -o "${REL_FILE}.o" -z noexecstack --format=binary "${REL_FILE}" 
            COMMAND ${EMBED_OBJCOPY} --rename-section .data=.rodata,alloc,load,readonly,data,contents "${REL_FILE}.o"
            WORKING_DIRECTORY ${WORKING_DIRECTORY}
            DEPENDS ${FILE}
            VERBATIM
        )
    endforeach()
endfunction()

function(add_embed_library EMBED_NAME)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/embed)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/embed/${EMBED_NAME})
    set(EMBED_DIR ${CMAKE_CURRENT_BINARY_DIR}/embed/${EMBED_NAME})
    set(SRC_FILE "${EMBED_DIR}/${EMBED_NAME}.cpp")
    set(HEADER_FILE "${EMBED_DIR}/include/${EMBED_NAME}.hpp")
    set(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set(OUTPUT_FILES)
    set(SYMBOLS)
    message(STATUS "Embedding files")
    foreach(FILE ${ARGN})
        # message(STATUS "Embedding file ${FILE}")
        embed_file(OUTPUT_FILE OUTPUT_SYMBOL ${FILE})
        list(APPEND OUTPUT_FILES ${OUTPUT_FILE})
        list(APPEND SYMBOLS ${OUTPUT_SYMBOL})
    endforeach()
    message(STATUS "Generating embedding library ${EMBED_NAME}")
    generate_embed_source(${EMBED_NAME} SRC ${SRC_FILE} HEADER ${HEADER_FILE} OBJECTS ${OUTPUT_FILES} SYMBOLS ${SYMBOLS})
    add_library(${EMBED_NAME} STATIC ${OUTPUT_FILES} "${SRC_FILE}")
    target_include_directories(${EMBED_NAME} PUBLIC "${EMBED_DIR}/include")
    set_target_properties(${EMBED_NAME} PROPERTIES POSITION_INDEPENDENT_CODE On)
endfunction()
