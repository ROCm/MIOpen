#include <cstring>
#include <vector>
#include <string>
#include <fstream>
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
        const auto file_removed = std::remove(_path.c_str()) == 0;
        const auto fd_closed = close(_fd) == 0;
        if (! (file_removed && fd_closed)) {
            assert(file_removed && fd_closed); // Nice copypaste do shut make tidy.
        }
    }

    inline operator char*() { return &_path[0]; }
    inline operator const char*() const { return _path.c_str(); }

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
	std::string exec_path(std::getenv("MLOPEN_EXPERIMENTAL_GCN_ASM_PATH")); // asciz
	{	// shut clang-analyzer-alpha.security.taint.TaintPropagation
		static const char bad[] = "!#$*;<>?@\\^`{|}";
		for (char * c = &exec_path[0]; c < (&exec_path[0] + exec_path.length()) ; ++c) {
			if (std::iscntrl(*c)) {
				*c = '_';
				continue;
			}
			for (const char * b = &bad[0]; b < (&bad[0] + sizeof(bad) - 1); ++b) {
				if (*b == *c) {
					*c = '_';
					break;
				}
			}
		}
	}
	std::vector<std::string> opt_storage ({
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
		opt_storage.push_back("-mcpu=" + std::string(name));
	}

	{
		std::istringstream iss(params);
		std::string param;
		while (iss >> param) {
			opt_storage.push_back(param);
		};
	}
	opt_storage.push_back("-");
	opt_storage.push_back("-o");

	std::vector<char*> args;
	args.push_back(&exec_path[0]);
	for (auto& opt : opt_storage) {
		args.push_back(&opt[0]);
	}
	
	TempFile outfile("amdgcn-asm-out-XXXXXX");
	args.push_back(outfile);
	args.push_back(nullptr);
	
	static const int read_side = 0;
	static const int write_side = 1;
	static const int pipe_sides = 2;
	int clang_stdin[pipe_sides];
	if (pipe(clang_stdin)) { MLOPEN_THROW("Error: X-AMDGCN-ASM: pipe()"); }

	int wstatus;
	pid_t pid = fork();
	if (pid == 0) {
		if (dup2(clang_stdin[read_side], STDIN_FILENO) == -1 ) { std::exit(EXIT_FAILURE); }
		if (close(clang_stdin[read_side])) { std::exit(EXIT_FAILURE); }
		if (close(clang_stdin[write_side])) { std::exit(EXIT_FAILURE); }
		execv(exec_path.c_str(), args.data());
		std::exit(EXIT_FAILURE);
	} else {
		if (close(clang_stdin[read_side])) { MLOPEN_THROW("Error: X-AMDGCN-ASM: close(clang_stdin[read_side])"); }
		if (pid == -1) {
			(void)close(clang_stdin[write_side]);
			MLOPEN_THROW("Error X-AMDGCN-ASM: fork()");
		}
		if (write(clang_stdin[write_side], source.data(), source.size()) == -1) {
			MLOPEN_THROW("Error: X-AMDGCN-ASM: write()");
		}
		if (close(clang_stdin[write_side])) { MLOPEN_THROW("Error: X-AMDGCN-ASM: close(clang_stdin[write_side])"); }
		if (waitpid(pid, &wstatus, 0) != pid) { MLOPEN_THROW("Error: X-AMDGCN-ASM: waitpid()"); }
	}
	
	if (WIFEXITED(wstatus)) {
		const int exit_status = WEXITSTATUS(wstatus);
		if (exit_status != 0) {
			MLOPEN_THROW("Error: X-AMDGCN-ASM: Assembly error (" + std::to_string(exit_status) + ")");
		}
	} else {
		MLOPEN_THROW("Error: X-AMDGCN-ASM: clang terminated abnormally");
	}

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
