PROJECT(AddKernels)

SET(ADD_KERNELS_SOURCE addkernels.cpp)

ADD_EXECUTABLE(addkernels ${ADD_KERNELS_SOURCE})

ADD_CUSTOM_TARGET(addkernels_run
	COMMAND addkernels -target ${PROJECT_BINARY_DIR}/../src/include/mlopen_kernels.h -guard GUARD_MLOPEN_KERNELS_HPP_
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/Kernels/
	DEPENDS addkernels)