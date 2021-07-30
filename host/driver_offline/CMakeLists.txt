include_directories(BEFORE
    include
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

set(CONV_FWD_DRIVER_OFFLINE_SOURCE conv_fwd_driver_offline.cpp)
set(CONV_BWD_DRIVER_OFFLINE_SOURCE conv_bwd_driver_offline.cpp)

add_executable(conv_fwd_driver_offline ${CONV_FWD_DRIVER_OFFLINE_SOURCE})
add_executable(conv_bwd_driver_offline ${CONV_BWD_DRIVER_OFFLINE_SOURCE})

target_link_libraries(conv_fwd_driver_offline PRIVATE host_tensor)
target_link_libraries(conv_bwd_driver_offline PRIVATE host_tensor)
