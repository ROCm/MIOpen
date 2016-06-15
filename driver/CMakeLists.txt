cmake_minimum_required( VERSION 2.8.12 )

project(driver)
message (STATUS "Building MLOpen Driver")

set( MLOPEN_LIB_SUFFIX "")
if( ${BUILD64} )
	set( MLOPEN_LIB_SUFFIX "64")
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
endif( )

include_directories (
	${MLOpen_DIRS_INCLUDE}
	)

add_executable(MLOpenDriver MLOpenDriver.cpp)
target_link_libraries(MLOpenDriver MLOpen)

# MLOpen depends on OpenCL
if( MLOpen_BACKEND MATCHES "OpenCL")
	MESSAGE( "MLOpen linking OpenCL: ${OPENCL_INCLUDE_DIRS}" )
	include_directories( ${OPENCL_INCLUDE_DIRS} )
	target_link_libraries( MLOpenDriver ${OPENCL_LIBRARIES} )
endif()
