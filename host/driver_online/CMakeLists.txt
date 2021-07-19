include_directories(BEFORE
    include
    ${PROJECT_BINARY_DIR}/host/online_compilation/include
    ${PROJECT_SOURCE_DIR}/host/online_compilation/include
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/problem_transform
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/driver
    ${PROJECT_SOURCE_DIR}/external/rocm/include
    ${PROJECT_SOURCE_DIR}/external/half/include
)

set(CONV_FWD_DRIVER_ONLINE_SOURCE conv_fwd_driver_online.cpp)

add_executable(conv_fwd_driver_online ${CONV_FWD_DRIVER_ONLINE_SOURCE})

target_link_libraries(conv_fwd_driver_online PRIVATE host_tensor)
target_link_libraries(conv_fwd_driver_online PRIVATE online_compilation)
