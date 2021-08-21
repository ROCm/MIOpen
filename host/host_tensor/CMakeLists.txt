include_directories(BEFORE
    include
)

set(HOST_TENSOR_SOURCE
    src/host_tensor.cpp;
    src/device.cpp;
)

## the library target
add_library(host_tensor SHARED ${HOST_TENSOR_SOURCE}) 

target_include_directories(host_tensor SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)

target_link_libraries(host_tensor PRIVATE hip::device)
target_link_libraries(host_tensor INTERFACE hip::host)

target_compile_features(host_tensor PUBLIC)
set_target_properties(host_tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS host_tensor LIBRARY DESTINATION lib) 
