
if(NOT TARGET doc)
    add_custom_target(doc)
endif()

function(mark_as_doc)
    add_dependencies(doc ${ARGN})
endfunction()
