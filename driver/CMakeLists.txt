cmake_minimum_required( VERSION 2.8.12 )

project(driver)
message (STATUS "Building MIOpen Driver")

set( MIOPEN_LIB_SUFFIX "")
if( ${BUILD64} )
	set( MIOPEN_LIB_SUFFIX "64")
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
endif( )

#add_executable(MIOpenDriver MIOpenDriver.cpp InputFlags.cpp)

add_executable(MIOpenDriver EXCLUDE_FROM_ALL main.cpp InputFlags.cpp)
target_link_libraries(MIOpenDriver MIOpen)

