find_path(OpenCL_INCLUDE_DIRS
	NAMES OpenCL/cl.h CL/cl.h
	HINTS
	${OpenCL_ROOT}/include
	$ENV{AMDAPPSDKROOT}/include
	$ENV{CUDA_PATH}/include
	PATHS
	/usr/include
	/usr/local/include
	/usr/local/cuda/include
	/opt/cuda/include
	DOC "OpenCL header file path"
	)
mark_as_advanced( OpenCL_INCLUDE_DIRS )

# Search for 64bit libs if FIND_LIBRARY_USE_LIB64_PATHS is set to true in the global environment, 32bit libs else
get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )

if( LIB64 )
	find_library( OpenCL_LIBRARIES
		NAMES OpenCL
		HINTS
		${OpenCL_ROOT}/lib
		$ENV{AMDAPPSDKROOT}/lib
		$ENV{CUDA_PATH}/lib
		DOC "OpenCL dynamic library path"
		PATH_SUFFIXES x86_64 x64 x86_64/sdk
		PATHS
		/usr/lib
		/usr/local/cuda/lib
		/opt/cuda/lib
		)
else( )
	find_library( OpenCL_LIBRARIES
		NAMES OpenCL
		HINTS
		${OPENCL_ROOT}/lib
		$ENV{AMDAPPSDKROOT}/lib
		$ENV{CUDA_PATH}/lib
		DOC "OpenCL dynamic library path"
		PATH_SUFFIXES x86 Win32

		PATHS
		/usr/lib
		/usr/local/cuda/lib
		/opt/cuda/lib
		)
endif( )
mark_as_advanced( OpenCL_LIBRARIES )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( OpenCL DEFAULT_MSG OpenCL_LIBRARIES OpenCL_INCLUDE_DIRS )

set(OPENCL_FOUND ${OpenCL_FOUND} CACHE INTERNAL "")
set(OPENCL_LIBRARIES ${OpenCL_LIBRARIES} CACHE INTERNAL "")
set(OPENCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS} CACHE INTERNAL "")

if( NOT OpenCL_FOUND )
	message( STATUS "FindOpenCL looked for libraries named: OpenCL" )
endif()
