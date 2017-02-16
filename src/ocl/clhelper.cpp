#include <vector>
#include <string>
#include <fstream>
#include <mlopen/clhelper.hpp>
#include <mlopen/kernel.hpp>
#include <mlopen/errors.hpp>
#include <mlopen/stringutils.hpp>
/* FIXME check if linux */
#include <unistd.h>
#include <sys/types.h> 
#include <sys/wait.h>

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
static void ExperimentalAmdgcnAssemble(std::string& source, const std::string& params)
{
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
		"-mcpu=fiji",  // TODO Set to the "device name" reported by OpenCL-on-ROCm runtime.
	});

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
	
	char outfile[] ="amdgcn-asm-out-XXXXXX";
	{	// We need filename for -o, so tmpfile() is not ok.
		const int fd_temp = mkstemp(&outfile[0]);
		if (fd_temp == -1) { MLOPEN_THROW("Error: X-AMDGCN-ASM: mkstemp()"); }
		if (close(fd_temp)) { MLOPEN_THROW("Error: X-AMDGCN-ASM: close(fd_temp)"); }
	}
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
		if (dup2(clang_stdin[read_side], STDIN_FILENO) == -1 ) { _exit(EXIT_FAILURE); }
		if (close(clang_stdin[read_side])) { _exit(EXIT_FAILURE); }
		if (close(clang_stdin[write_side])) { _exit(EXIT_FAILURE); }
		execv(exec_path.c_str(), args.data());
		_exit(EXIT_FAILURE);
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
	std::remove(outfile);
	if (outfile_read_failed) {
		MLOPEN_THROW("Error: X-AMDGCN-ASM: outfile_read_failed");
	}
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
			ExperimentalAmdgcnAssemble(source, params);
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
