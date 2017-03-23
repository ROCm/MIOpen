#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <mlopen/gcn_asm_utils.h>
#include <mlopen/clhelper.hpp>
#include <mlopen/kernel.hpp>
#include <mlopen/errors.hpp>
#include <mlopen/stringutils.hpp>
#ifndef WIN32 //Linux or APPLE
#include <unistd.h>
#include <sys/types.h> 
#include <sys/wait.h>
#endif //WIN32

#ifndef WIN32 //Linux or APPLE
class TempFile
{
public:
    TempFile(const std::string& path_template)
        : _path(path_template)
    {
        _fd = mkstemp(&_path[0]);
        if (_fd == -1) { MLOPEN_THROW("Error: TempFile: mkstemp()"); }
    }

    ~TempFile()
    {
        const int remove_rc = std::remove(_path.c_str());
        const int close_rc = close(_fd);
        if (remove_rc != 0 || close_rc != 0) {
#ifndef NDEBUG // Be quiet in release versions.
            std::fprintf(stderr, "Error: TempFile: On removal of '%s', remove_rc = %d, close_rc = %d.\n", _path.c_str(), remove_rc, close_rc);
#endif
        }
    }

    inline operator const std::string&() { return _path; }

private:
    std::string _path;
    int _fd;
};
#endif

namespace mlopen {

static cl_program CreateProgram(cl_context ctx, const char* char_source, size_t size)
{
	cl_int status;
	auto result = clCreateProgramWithSource(ctx,
		1,
		&char_source,
		&size,
		&status);

	if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status, "Error Creating OpenCL Program (cl_program) in LoadProgram()"); }

	return result;
}

/*
 * Temporary function which emulates online assembly feature of OpenCL-on-ROCm being developed.
 * Not intended to be used in production code, so error handling is very straghtforward,
 * just catch whatever possible and throw an exception.
 */
static void ExperimentalAmdgcnAssemble(cl_device_id device, std::string& source, const std::string& params)
{
#ifndef WIN32 //Linux or APPLE
    TempFile outfile("amdgcn-asm-out-XXXXXX");

	std::vector<std::string> args ({
		"-x",
		"assembler",
		"-target",
		"amdgcn--amdhsa",
	});

	{ 	// Add -mcpu=name as reported by OpenCL-on-ROCm runtime.
		char name[64] = {0};
		if (CL_SUCCESS != clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr)) {
			MLOPEN_THROW("Error: X-AMDGCN-ASM: clGetDeviceInfo()");
		}
        args.push_back("-mcpu=" + std::string(name));
	}

	{
		std::istringstream iss(params);
		std::string param;
		while (iss >> param) {
            args.push_back(param);
		};
	}
    args.push_back("-");
    args.push_back("-o");
    args.push_back(outfile);
	
    std::istringstream clang_stdin(source);
    ExecuteGcnAssembler(args, &clang_stdin);

	std::ifstream file(outfile, std::ios::binary | std::ios::ate);
	bool outfile_read_failed = false;
	do {
		const auto size = file.tellg();
		if (size == -1) { outfile_read_failed = true; break; }
		source.resize(size, '\0');
		file.seekg(std::ios::beg);
		if (file.fail()) { outfile_read_failed = true; break; }
		if (file.rdbuf()->sgetn(&source[0], size) != size) { outfile_read_failed = true; break; }
	} while (0);
	file.close();
	if (outfile_read_failed) {
		MLOPEN_THROW("Error: X-AMDGCN-ASM: outfile_read_failed");
	}
#else
	(void)source; // -warning
	(void)params; // -warning
	MLOPEN_THROW("Error: X-AMDGCN-ASM: online assembly under Windows is not supported");
#endif //WIN32
}

static cl_program CreateProgramWithBinary(cl_context ctx, cl_device_id device, const char* char_source, size_t size)
{
	cl_int status, binaryStatus;
	auto result = clCreateProgramWithBinary(ctx,
		1,
		&device,
		reinterpret_cast<const size_t*>(&size),
		reinterpret_cast<const unsigned char**>(&char_source),
		&status,
		&binaryStatus);

	if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status, "Error creating code object program (cl_program) in LoadProgramFromBinary()"); }

	return result;
}

static void BuildProgram(cl_program program, cl_device_id device, const std::string& params = "")
{
	auto status = clBuildProgram(program,
		1, &device, params.c_str(),
		nullptr,
		nullptr);

	if (status != CL_SUCCESS)
	{
		std::string msg = "Error Building OpenCL Program in BuildProgram()\n";
		std::vector<char> errorbuf(1024 * 1024);
		size_t psize;
		clGetProgramBuildInfo(program,
			device,
			CL_PROGRAM_BUILD_LOG,
			1024 * 1024,
			errorbuf.data(),
			&psize);

		msg += errorbuf.data();
		if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status, msg); }
	}
}

ClProgramPtr LoadProgram(cl_context ctx, cl_device_id device, const std::string &program_name, const std::string& params, bool is_kernel_str)
{
	bool is_binary = false;
	std::string source;
	if (is_kernel_str) {
		source = program_name;
	} else {
		source = mlopen::GetKernelSrc(program_name);
		auto is_asm = mlopen::EndsWith(program_name, ".s");
		if (is_asm) { // Overwrites source (asm text) by binary results of assembly:
			ExperimentalAmdgcnAssemble(device, source, params);
			is_binary = true;
		} else {
			is_binary = mlopen::EndsWith(program_name, ".so");
		}
	}

	cl_program result = nullptr;
	if (is_binary) {
		result = CreateProgramWithBinary(ctx, device, source.data(), source.size());
		BuildProgram(result, device);
	} else {
		result = CreateProgram(ctx, source.data(), source.size());
		BuildProgram(result, device, params + " -cl-std=CL1.2");
	}
	return ClProgramPtr{ result };
}

ClKernelPtr CreateKernel(cl_program program, const std::string& kernel_name)
{
	cl_int status;
	ClKernelPtr result{clCreateKernel(program, 
			kernel_name.c_str(), 
			&status)};

	if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status); }

	return result;
}

cl_device_id GetDevice(cl_command_queue q)
{
	cl_device_id device;
	cl_int status = clGetCommandQueueInfo(q,
			CL_QUEUE_DEVICE, 
			sizeof(cl_device_id),
			&device, 
			nullptr);
	if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status, "Error Getting Device Info from Queue in GetDevice()"); }

	return device;
}

cl_context GetContext(cl_command_queue q)
{
	cl_context context;
	cl_int status = clGetCommandQueueInfo(q,
			CL_QUEUE_CONTEXT, 
			sizeof(cl_context),
			&context, 
			nullptr);
	if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status, "Error Getting Device Info from Queue in GetDevice()"); }
	return context;
}

ClAqPtr CreateQueueWithProfiling(cl_context ctx, cl_device_id dev) 
{
	cl_int status;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
	ClAqPtr q{clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &status)};
#ifdef __clang__
#pragma clang diagnostic pop
#endif

	if(status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status); }

	return q;
}

} // namespace mlopen
