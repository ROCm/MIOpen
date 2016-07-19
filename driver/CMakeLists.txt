cmake_minimum_required( VERSION 2.8.12 )

project(driver)
message (STATUS "Building MLOpen Driver")

set( MLOPEN_LIB_SUFFIX "")
if( ${BUILD64} )
	set( MLOPEN_LIB_SUFFIX "64")
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
endif( )

add_executable(MLOpenDriver MLOpenDriver.cpp InputFlags.cpp)
target_link_libraries(MLOpenDriver MLOpen)

