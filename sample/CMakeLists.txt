message(STATUS "Building MIOpen standalone CBA integration sample")
cmake_minimum_required( VERSION 3.15 )
add_compile_definitions(__HIP_PLATFORM_AMD__)

project(MIOpenCBASample CXX)
set(CMAKE_CXX_STANDARD 17)

list(
    APPEND
    CMAKE_PREFIX_PATH
    ${ROCM_PATH}/llvm
    ${ROCM_PATH}
    ${ROCM_PATH}/hip
    /opt/rocm/llvm
    /opt/rocm
    /opt/rocm/hip 
)

find_package(hip REQUIRED)
find_package(miopen CONFIG REQUIRED)

add_executable(sample_conv_cba_fused_op sample_conv_cba_fused_op.cpp)

target_link_libraries(sample_conv_fused_op PRIVATE miopen hip)
target_include_directories(sample_conv_fused_op PRIVATE ${HIP_INCLUDE_DIRS})

# install(TARGETS sample_conv_fused_op EXPORT sample_conv_fused_op LIBRARY DESTINATION bin/samples)