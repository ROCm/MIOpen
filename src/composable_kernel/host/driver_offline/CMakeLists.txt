set(__driver_offline_include_dirs
    include
    ${PROJECT_SOURCE_DIR}/host/host_tensor/include
    ${PROJECT_SOURCE_DIR}/host/solver/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/utility
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_description
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/tensor_operation
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/problem_transform
    ${PROJECT_SOURCE_DIR}/composable_kernel/include/driver
    ${PROJECT_SOURCE_DIR}/external/rocm/include
)

add_executable(conv_fwd_driver_offline src/conv_fwd_driver_offline.cpp)
target_link_libraries(conv_fwd_driver_offline PRIVATE host_tensor)
target_include_directories(conv_fwd_driver_offline PRIVATE ${__driver_offline_include_dirs})
add_executable(conv_bwd_driver_offline src/conv_bwd_driver_offline.cpp)
target_link_libraries(conv_bwd_driver_offline PRIVATE host_tensor)
target_include_directories(conv_bwd_driver_offline PRIVATE ${__driver_offline_include_dirs})