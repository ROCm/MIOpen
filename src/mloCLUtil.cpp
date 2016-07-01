/**********************************************************************
Copyright (c)2016 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

?   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
?   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
#include "mlo_internal.hpp"
#include "mloUtils.hpp"


void tokenize(const std::string& str,
	std::vector<std::string>& tokens,
	const std::string& delimiters)
{
	// Skip delimiters at beginning.
	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// Find first "non-delimiter".
	std::string::size_type pos = str.find_first_of(delimiters, lastPos);

	while (std::string::npos != pos || std::string::npos != lastPos)
	{
		// Found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// Skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
		// Find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}
}


/**
* getCurrentDir
* Get current directory
* @return string
*/
static std::string mloGetCurrentDir() {
	const size_t pathSize = 4096;
	char currentDir[pathSize];
	// Check if we received the path
	if (getcwd(currentDir, pathSize) != NULL) {
		return std::string(currentDir);
	}
	return std::string("");
}

std::string mloGetPath()
{
#ifdef _WIN32
	char buffer[MAX_PATH];
#ifdef UNICODE
	if (!GetModuleFileName(NULL, (LPWCH)buffer, sizeof(buffer)))
	{
		throw std::string("GetModuleFileName() failed!");
	}
#else
	if (!GetModuleFileName(NULL, buffer, sizeof(buffer)))
	{
		throw std::string("GetModuleFileName() failed!");
	}
#endif
	std::string str(buffer);
	/* '\' == 92 */
	int last = (int)str.find_last_of((char)92);
#else
	char buffer[PATH_MAX + 1];
	ssize_t len;
	if ((len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1)) == -1)
	{
		throw std::string("readlink() failed!");
	}
	buffer[len] = '\0';
	std::string str(buffer);
	/* '/' == 47 */
	int last = (int)str.find_last_of((char)47);
#endif
	return str.substr(0, last + 1);
}

int mloGetContextDeviceFromCLQueue(cl_context & context, cl_device_id & device, cl_command_queue * profile_q, const cl_command_queue & q)
{
	int status = 0;
	size_t ret_sz;
	status = clGetCommandQueueInfo(q,
		CL_QUEUE_CONTEXT,
		sizeof(cl_context),
		&context,
		&ret_sz);

	status = clGetCommandQueueInfo(q,
		CL_QUEUE_DEVICE,
		sizeof(cl_device_id),
		&device,
		&ret_sz);
	if (profile_q)
	{
		// DO NOT KNOW HOW TO DO THAT
#if 0 //def CL_VERSION_2_0 
		uint prop[2] = { CL_QUEUE_PROFILING_ENABLE, 0 };

		*profile_q = clCreateCommandQueueWithProperties(context,
			device,
			(const cl_queue_properties *)prop,
			&status);
#else
		*profile_q = clCreateCommandQueue(context,
			device,
			CL_QUEUE_PROFILING_ENABLE,
			&status);
#endif
	}
	return(status);

}
/**
* mloLoadOpenCLProgramFromSource
* create the opencl program

*/

int mloLoadOpenCLProgramFromSource(cl_program & program, const cl_context& context,
									std::string kernel_path, std::string kernel_nm, 
									bool quiet)
{
	cl_int status = CL_SUCCESS;
	mloFile kernelFile;
	std::string kernelPath = (kernel_path == "") ? mloGetPath() : kernel_path;
	kernelPath.append(std::string("/") + kernel_nm.c_str());
	if (!kernelFile.open(kernelPath.c_str()))//bool
	{
		std::cout << "Failed to load kernel file: " << kernelPath << std::endl;
		return -1;
	}

	if (!quiet)
	{
		std::cout << "Ocl source file is : " << kernelPath.c_str() << std::endl;
	}

	const char * source = kernelFile.source().c_str();
	size_t sourceSize[] = { strlen(source) };
	program = clCreateProgramWithSource(context,
		1,
		&source,
		sourceSize,
		&status);
	if (status != CL_SUCCESS)
	{
		std::cout << "clCreateProgramWithSource failed." << std::endl;
		return -1;
	}
	return 0;
}



/**
* mloBuildOpenCLProgram
* builds the opencl program
*/
int mloBuildOpenCLProgram(const cl_context& context,
	cl_device_id device,
	cl_program program,
	const std::string flagsStr,
	bool quiet)
{
	cl_int status = CL_SUCCESS;

	if (!quiet && flagsStr.size() != 0)
	{
		std::cout << "Build Options are : " << flagsStr.c_str() << std::endl;
	}
	/* create a cl program executable for all the devices specified */
	status = clBuildProgram(program, 1, &device, flagsStr.c_str(), NULL, NULL);

#if 0
	std::vector<std::string> tokens;

	tokenize(flagsStr, tokens);

	{
		std::ofstream ofs;
		std::string fileName = buildData.kernelPath + buildData.kernelName + ".opt.txt";
		ofs.open(fileName, std::ofstream::out | std::ofstream::trunc);


		for (std::vector<std::string>::iterator it = tokens.begin(); it != tokens.end(); ++it)
		{
			if ((*it) == "-D" || (*it) == "-I")
			{
				ofs << (*it) << " ";
			}
			else
			{
				ofs << (*it) << "\n";
			}
		}
		ofs.close();

	}
#endif
	if (status != CL_SUCCESS)
	{
		if (status == CL_BUILD_PROGRAM_FAILURE)
		{
			cl_int logStatus;
			char *buildLog = NULL;
			size_t buildLogSize = 0;
			logStatus = clGetProgramBuildInfo(
				program,
				device,
				CL_PROGRAM_BUILD_LOG,
				buildLogSize,
				buildLog,
				&buildLogSize);
//			CHECK_OPENCL_ERROR(logStatus, "clGetProgramBuildInfo failed.");
			buildLog = (char*)malloc(buildLogSize);
//			CHECK_ALLOCATION(buildLog, "Failed to allocate host memory. (buildLog)");
			memset(buildLog, 0, buildLogSize);
			logStatus = clGetProgramBuildInfo(
				program,
				device,
				CL_PROGRAM_BUILD_LOG,
				buildLogSize,
				buildLog,
				NULL);

			std::cout << " \n\t\t\tBUILD LOG\n";
			std::cout << " ************************************************\n";
			std::cout << buildLog << std::endl;
			std::cout << " ************************************************\n";
			free(buildLog);
		}
		std::cout << "clBuildProgram failed with status " << status << std::endl;
		return status;
	}
	return CL_SUCCESS;
}



int mloExecuteNoWait(
	const mlo_ocl_args & args,
	cl_command_queue queue,
	cl_kernel ocl_kernel,
	const std::vector<size_t> & gv_wk,
	const std::vector<size_t> & lv_wk,
	cl_event * event
	)
{
	int ret = CL_SUCCESS;

	mlo_ocl_args::const_iterator ai;
	for (ai = args.begin(); ai != (args).end(); ++ai)
	{
		int i = (*ai).first;
		const mlo_ocl_arg arg = (*ai).second;
		ret |= clSetKernelArg(ocl_kernel, i, arg.first, arg.second);
	}
	if (ret != CL_SUCCESS)
	{
		std::cout << "parmeters failed." << std::endl;
		return(-1);
	}

	size_t g_wk[3] = { 1, 1, 1 };
	size_t l_wk[3] = { 1, 1, 1 };

	for (size_t i = 0; i < gv_wk.size(); ++i)
	{
		g_wk[i] = gv_wk[i];
	}
	for (size_t i = 0; i < lv_wk.size(); ++i)
	{
		l_wk[i] = lv_wk[i];
	}


	cl_command_queue ocl_queue = queue;

	ret = clEnqueueNDRangeKernel(ocl_queue, ocl_kernel, 3, NULL, g_wk, l_wk, 0, NULL, event);

	if (ret != CL_SUCCESS)
	{
		std::cout << "ERROR: Kernel failed with error " << ret << std::endl;
	}
	else  if (event)
	{
		clWaitForEvents(1, event);
	}

	return(ret);
}


/**
* mloGetDeviceInfo
*/
int mloGetDeviceInfo(cl_device_id deviceId,
	int & maxComputeUnits,
	int & maxWorkItemDims,
	std::vector<size_t> & maxWorkItemSize,
	size_t & maxWorkGroupSize,
	int & maxClockFrequency,
	size_t & maxMemAllocSize,
	size_t &localMemSize,
	size_t &timerResolution,
	std::string & deviceName)
{
	cl_int status = CL_SUCCESS;
	//Get max compute units
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_COMPUTE_UNITS,
		sizeof(cl_uint),
		&maxComputeUnits,
		NULL);
	//Get max work item dimensions
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
		sizeof(cl_uint),
		&maxWorkItemDims,
		NULL);

	size_t *t_maxWorkItemSizes = new size_t[maxWorkItemDims];
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_WORK_ITEM_SIZES,
		maxWorkItemDims * sizeof(size_t),
		t_maxWorkItemSizes,
		NULL);

	for (int i = 0; i < maxWorkItemDims; ++i)
	{
		maxWorkItemSize.push_back(t_maxWorkItemSizes[i]);
	}

	delete[] t_maxWorkItemSizes;
	// Maximum work group size
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof(size_t),
		&maxWorkGroupSize,
		NULL);
#if 0
	// Preferred vector sizes of all data types
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
		sizeof(cl_uint),
		&preferredCharVecWidth,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) failed");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
		sizeof(cl_uint),
		&preferredShortVecWidth,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT) failed");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
		sizeof(cl_uint),
		&preferredIntVecWidth,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT) failed");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
		sizeof(cl_uint),
		&preferredLongVecWidth,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG) failed");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
		sizeof(cl_uint),
		&preferredFloatVecWidth,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) failed");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
		sizeof(cl_uint),
		&preferredDoubleVecWidth,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) failed");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
		sizeof(cl_uint),
		&preferredHalfVecWidth,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF) failed");
#endif
	// Clock frequency
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_CLOCK_FREQUENCY,
		sizeof(cl_uint),
		&maxClockFrequency,
		NULL);

#if 0
	// Address bits
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_ADDRESS_BITS,
		sizeof(cl_uint),
		&addressBits,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_ADDRESS_BITS) failed");
#endif
	// Maximum memory alloc size
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_MEM_ALLOC_SIZE,
		sizeof(cl_ulong),
		&maxMemAllocSize,
		NULL);
#if 0
	// Image support
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_IMAGE_SUPPORT,
		sizeof(cl_bool),
		&imageSupport,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_IMAGE_SUPPORT) failed");
	// Maximum read image arguments
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_READ_IMAGE_ARGS,
		sizeof(cl_uint),
		&maxReadImageArgs,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_MAX_READ_IMAGE_ARGS) failed");
	// Maximum write image arguments
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
		sizeof(cl_uint),
		&maxWriteImageArgs,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_MAX_WRITE_IMAGE_ARGS) failed");
	// 2D image and 3D dimensions
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_IMAGE2D_MAX_WIDTH,
		sizeof(size_t),
		&image2dMaxWidth,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH) failed");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_IMAGE2D_MAX_HEIGHT,
		sizeof(size_t),
		&image2dMaxHeight,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT) failed");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_IMAGE3D_MAX_WIDTH,
		sizeof(size_t),
		&image3dMaxWidth,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_IMAGE3D_MAX_WIDTH) failed");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_IMAGE3D_MAX_HEIGHT,
		sizeof(size_t),
		&image3dMaxHeight,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_IMAGE3D_MAX_HEIGHT) failed");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_IMAGE3D_MAX_DEPTH,
		sizeof(size_t),
		&image3dMaxDepth,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_IMAGE3D_MAX_DEPTH) failed");
	// Maximum samplers
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_SAMPLERS,
		sizeof(cl_uint),
		&maxSamplers,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_MAX_SAMPLERS) failed");
	// Maximum parameter size
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_PARAMETER_SIZE,
		sizeof(size_t),
		&maxParameterSize,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_MAX_PARAMETER_SIZE) failed");
	// Memory base address align
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MEM_BASE_ADDR_ALIGN,
		sizeof(cl_uint),
		&memBaseAddressAlign,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN) failed");
	// Minimum data type align size
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
		sizeof(cl_uint),
		&minDataTypeAlignSize,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE) failed");
	// Single precision floating point configuration
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_SINGLE_FP_CONFIG,
		sizeof(cl_device_fp_config),
		&singleFpConfig,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_SINGLE_FP_CONFIG) failed");
	// Double precision floating point configuration
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_DOUBLE_FP_CONFIG,
		sizeof(cl_device_fp_config),
		&doubleFpConfig,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_DOUBLE_FP_CONFIG) failed");
	// Global memory cache type
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
		sizeof(cl_device_mem_cache_type),
		&globleMemCacheType,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE) failed");
	// Global memory cache line size
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
		sizeof(cl_uint),
		&globalMemCachelineSize,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) failed");
	// Global memory cache size
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
		sizeof(cl_ulong),
		&globalMemCacheSize,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) failed");
	// Global memory size
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_GLOBAL_MEM_SIZE,
		sizeof(cl_ulong),
		&globalMemSize,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE) failed");
	// Maximum constant buffer size
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
		sizeof(cl_ulong),
		&maxConstBufSize,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) failed");
	// Maximum constant arguments
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_MAX_CONSTANT_ARGS,
		sizeof(cl_uint),
		&maxConstArgs,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_MAX_CONSTANT_ARGS) failed");
	// Local memory type
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_LOCAL_MEM_TYPE,
		sizeof(cl_device_local_mem_type),
		&localMemType,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_LOCAL_MEM_TYPE) failed");
#endif
	int sz0 = sizeof(cl_ulong);
	int sz1 = sizeof(localMemSize);
	// Local memory size
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_LOCAL_MEM_SIZE,
		sizeof(cl_ulong),
		&localMemSize,
		NULL);

#if 0
	// Error correction support
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_ERROR_CORRECTION_SUPPORT,
		sizeof(cl_bool),
		&errCorrectionSupport,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_ERROR_CORRECTION_SUPPORT) failed");
#endif
	// Profiling timer resolution
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PROFILING_TIMER_RESOLUTION,
		sizeof(size_t),
		&timerResolution,
		NULL);
#if 0
	// Endian little
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_ENDIAN_LITTLE,
		sizeof(cl_bool),
		&endianLittle,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_ENDIAN_LITTLE) failed");
	// Device available
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_AVAILABLE,
		sizeof(cl_bool),
		&available,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_AVAILABLE) failed");
	// Device compiler available
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_COMPILER_AVAILABLE,
		sizeof(cl_bool),
		&compilerAvailable,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_COMPILER_AVAILABLE) failed");
	// Device execution capabilities
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_EXECUTION_CAPABILITIES,
		sizeof(cl_device_exec_capabilities),
		&execCapabilities,
		NULL);
	CHECK_OPENCL_ERROR(status,
		"clGetDeviceInfo(CL_DEVICE_EXECUTION_CAPABILITIES) failed");
	// Device queue properities
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_QUEUE_PROPERTIES,
		sizeof(cl_command_queue_properties),
		&queueProperties,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_QUEUE_PROPERTIES) failed");
	// Platform
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PLATFORM,
		sizeof(cl_platform_id),
		&platform,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_PLATFORM) failed");
#endif
	// Device name
	size_t tempSize = 0;
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_NAME,
		0,
		NULL,
		&tempSize);
	char * t_name = new char[tempSize];
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_NAME,
		sizeof(char) * tempSize,
		t_name,
		NULL);

	deviceName = std::string(t_name);
	delete[] t_name;
#if 0
	// Vender name
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_VENDOR,
		0,
		NULL,
		&tempSize);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_VENDOR) failed");
	if (vendorName != NULL) delete[] vendorName;
	vendorName = new char[tempSize];
	CHECK_ALLOCATION(vendorName, "Failed to allocate memory(venderName)");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_VENDOR,
		sizeof(char) * tempSize,
		vendorName,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_VENDOR) failed");
	// Driver name
	status = clGetDeviceInfo(
		deviceId,
		CL_DRIVER_VERSION,
		0,
		NULL,
		&tempSize);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DRIVER_VERSION) failed");
	if (driverVersion != NULL) delete[] driverVersion;
	driverVersion = new char[tempSize];
	CHECK_ALLOCATION(driverVersion, "Failed to allocate memory(driverVersion)");
	status = clGetDeviceInfo(
		deviceId,
		CL_DRIVER_VERSION,
		sizeof(char) * tempSize,
		driverVersion,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DRIVER_VERSION) failed");
	// Device profile
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PROFILE,
		0,
		NULL,
		&tempSize);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_PROFILE) failed");
	if (profileType != NULL) delete[] profileType;
	profileType = new char[tempSize];
	CHECK_ALLOCATION(profileType, "Failed to allocate memory(profileType)");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_PROFILE,
		sizeof(char) * tempSize,
		profileType,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_PROFILE) failed");
	// Device version
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_VERSION,
		0,
		NULL,
		&tempSize);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_VERSION) failed");
	if (deviceVersion != NULL) delete[] deviceVersion;
	deviceVersion = new char[tempSize];
	CHECK_ALLOCATION(deviceVersion, "Failed to allocate memory(deviceVersion)");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_VERSION,
		sizeof(char) * tempSize,
		deviceVersion,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_VERSION) failed");
	// Device extensions
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_EXTENSIONS,
		0,
		NULL,
		&tempSize);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_EXTENSIONS) failed");
	if (extensions != NULL) delete[] extensions;
	extensions = new char[tempSize];
	CHECK_ALLOCATION(extensions, "Failed to allocate memory(extensions)");
	status = clGetDeviceInfo(
		deviceId,
		CL_DEVICE_EXTENSIONS,
		sizeof(char) * tempSize,
		extensions,
		NULL);
	CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_EXTENSIONS) failed");
	// Device parameters of OpenCL 1.1 Specification
#ifdef CL_VERSION_1_1
	std::string deviceVerStr(deviceVersion);
	size_t vStart = deviceVerStr.find(" ", 0);
	size_t vEnd = deviceVerStr.find(" ", vStart + 1);
	std::string vStrVal = deviceVerStr.substr(vStart + 1, vEnd - vStart - 1);
	if (vStrVal.compare("1.0") > 0)
	{
		// Native vector sizes of all data types
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
			sizeof(cl_uint),
			&nativeCharVecWidth,
			NULL);
		CHECK_OPENCL_ERROR(status,
			"clGetDeviceInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR) failed");
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
			sizeof(cl_uint),
			&nativeShortVecWidth,
			NULL);
		CHECK_OPENCL_ERROR(status,
			"clGetDeviceInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT) failed");
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
			sizeof(cl_uint),
			&nativeIntVecWidth,
			NULL);
		CHECK_OPENCL_ERROR(status,
			"clGetDeviceInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT) failed");
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
			sizeof(cl_uint),
			&nativeLongVecWidth,
			NULL);
		CHECK_OPENCL_ERROR(status,
			"clGetDeviceInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG) failed");
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
			sizeof(cl_uint),
			&nativeFloatVecWidth,
			NULL);
		CHECK_OPENCL_ERROR(status,
			"clGetDeviceInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT) failed");
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
			sizeof(cl_uint),
			&nativeDoubleVecWidth,
			NULL);
		CHECK_OPENCL_ERROR(status,
			"clGetDeviceInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE) failed");
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
			sizeof(cl_uint),
			&nativeHalfVecWidth,
			NULL);
		CHECK_OPENCL_ERROR(status,
			"clGetDeviceInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF) failed");
		// Host unified memory
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_HOST_UNIFIED_MEMORY,
			sizeof(cl_bool),
			&hostUnifiedMem,
			NULL);
		CHECK_OPENCL_ERROR(status,
			"clGetDeviceInfo(CL_DEVICE_HOST_UNIFIED_MEMORY) failed");
		// Device OpenCL C version
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_OPENCL_C_VERSION,
			0,
			NULL,
			&tempSize);
		CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_OPENCL_C_VERSION) failed");
		if (openclCVersion != NULL) delete[] openclCVersion;
		openclCVersion = new char[tempSize];
		CHECK_ALLOCATION(openclCVersion, "Failed to allocate memory(openclCVersion)");
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_OPENCL_C_VERSION,
			sizeof(char) * tempSize,
			openclCVersion,
			NULL);
		CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_OPENCL_C_VERSION) failed");
	}
#endif
#ifdef CL_VERSION_2_0
	if (checkOpenCL2_XCompatibility())
	{
		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_SVM_CAPABILITIES,
			sizeof(cl_device_svm_capabilities),
			&svmcaps,
			NULL);
		CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_SVM_CAPABILITIES) failed");

		status = clGetDeviceInfo(
			deviceId,
			CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
			sizeof(cl_uint),
			&maxQueueSize,
			NULL);
		CHECK_OPENCL_ERROR(status, "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE) failed");
	}
#endif

#endif
	return CL_SUCCESS;
		}



		/**
		* ReadEventTime
		* auxiliary functions to read event time
		* @param cl_event& event
		* @param double* event duration in secs
		* @return status of event duration reading
		*/
		int mloReadEventTime(cl_event& event, double & time)
		{
			// Calculate performance
			cl_ulong startTime;
			cl_ulong endTime;

			cl_int status = CL_SUCCESS;

			// Get kernel profiling info
			status = clGetEventProfilingInfo(event,
				CL_PROFILING_COMMAND_START,
				sizeof(cl_ulong),
				&startTime,
				0);
			//CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.(startTime)");

			status = clGetEventProfilingInfo(event,
				CL_PROFILING_COMMAND_END,
				sizeof(cl_ulong),
				&endTime,
				0);
			//CHECK_OPENCL_ERROR(status, "clGetEventProfilingInfo failed.(endTime)");

			time = (double)(1e-6 * (endTime - startTime));
			return status;
		}


#if 0
static cl_int spinForEventsComplete(cl_uint num_events, cl_event *event_list)
{
	cl_int ret = 0;

	cl_int param_value;
	size_t param_value_size_ret;

	for (cl_uint e = 0; e < num_events; e++)
	{
		while (1)
		{
			ret |= clGetEventInfo(event_list[e],
				CL_EVENT_COMMAND_EXECUTION_STATUS,
				sizeof(cl_int),
				&param_value,
				&param_value_size_ret);

			if (param_value == CL_COMPLETE)
				break;
		}
	}

	for (cl_uint e = 0; e < num_events; e++)
		clReleaseEvent(event_list[e]);
	return ret;
}

/**
 * waitForEventAndRelease
 * waits for a event to complete and release the event afterwards
 * @param event cl_event object
 * @return 0 if success else nonzero
 */
static int waitForEventAndRelease(cl_event *event)
{
    cl_int status = CL_SUCCESS;

	status = clWaitForEvents(1, event);
	CHECK_OPENCL_ERROR(status, "clWaitForEvents Failed with Error Code:");

    status = clReleaseEvent(*event);
    CHECK_OPENCL_ERROR(status, "clReleaseEvent Failed with Error Code:");

    return ALIBS_SUCCESS;
}

/**
 * getLocalThreads
 * get Local Threads number
 */
static size_t getLocalThreads(size_t globalThreads, size_t maxWorkItemSize)
{
    if(maxWorkItemSize < globalThreads)
    {
        if(globalThreads%maxWorkItemSize == 0)
        {
            return maxWorkItemSize;
        }
        else
        {
            for(size_t i=maxWorkItemSize-1; i > 0; --i)
            {
                if(globalThreads%i == 0)
                {
                    return i;
                }
            }
        }
    }
    else
    {
        return globalThreads;
    }
    return ALIBS_SUCCESS;
}








#endif
