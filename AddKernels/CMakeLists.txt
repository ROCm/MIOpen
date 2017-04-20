project(AddKernels)

set(ADD_KERNELS_SOURCE include_inliner.cpp addkernels.cpp)

add_executable(addkernels EXCLUDE_FROM_ALL ${ADD_KERNELS_SOURCE})
