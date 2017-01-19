find_path(OPENCL_INCLUDE_DIRS
	NAMES OpenCL/cl.h CL/cl.h
	HINTS
	${OPENCL_ROOT}/include
	$ENV{AMDAPPSDKROOT}/include
	$ENV{CUDA_PATH}/include
	PATHS
	/usr/include
	/usr/local/include
	/usr/local/cuda/include
	/opt/cuda/include
	/opt/rocm/opencl/include
	DOC "OpenCL header file path"
	)
mark_as_advanced( OPENCL_INCLUDE_DIRS )

# Search for 64bit libs if FIND_LIBRARY_USE_LIB64_PATHS is set to true in the global environment, 32bit libs else
get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )

if( LIB64 )
	find_library( OPENCL_LIBRARIES
		NAMES OpenCL
		HINTS
		${OPENCL_ROOT}/lib
		$ENV{AMDAPPSDKROOT}/lib
		$ENV{CUDA_PATH}/lib
		DOC "OpenCL dynamic library path"
		PATH_SUFFIXES x86_64 x64 x86_64/sdk
		PATHS
		/usr/lib
		/usr/local/cuda/lib
		/opt/cuda/lib
		/opt/rocm/opencl/lib
		)
else( )
	find_library( OPENCL_LIBRARIES
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
mark_as_advanced( OPENCL_LIBRARIES )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( OPENCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS )

set(OpenCL_FOUND ${OPENCL_FOUND} CACHE INTERNAL "")
set(OpenCL_LIBRARIES ${OPENCL_LIBRARIES} CACHE INTERNAL "")
set(OpenCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS} CACHE INTERNAL "")

if( NOT OPENCL_FOUND )
	message( STATUS "FindOpenCL looked for libraries named: OpenCL" )
endif()
