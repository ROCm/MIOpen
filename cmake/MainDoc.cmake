
if(NOT TARGET doc)
    add_custom_target(doc)
endif()

function(mark_as_doc)
    add_dependencies(doc ${ARGN})
endfunction()

function(clean_doc_output DIR)
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${DIR})
endfunction()
