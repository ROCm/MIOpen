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
	close(mkstemp(&outfile[0]));
	args.push_back(outfile);
	args.push_back(nullptr);
	
	static const int read_fd = 0;
	static const int write_fd = 1;
	static const int pipe_sides = 2;

	int childStdin[pipe_sides];

	pipe(childStdin);

	int status;
	pid_t pid = fork();

	if (pid == 0)
	{
		dup2(childStdin[read_fd], STDIN_FILENO);

		close(childStdin[read_fd]);
		close(childStdin[write_fd]);

		execv(exec_path.c_str(), args.data());
		_exit(EXIT_FAILURE);
	}
	else
	{
		close(childStdin[read_fd]);

		if (pid < 0)
		{
			close(childStdin[write_fd]);
			MLOPEN_THROW("Error assembling kernel source: fork failed.");
		}

		write(childStdin[write_fd], source.data(), source.size());
		close(childStdin[write_fd]);

		if (waitpid(pid, &status, 0) != pid) { MLOPEN_THROW("Error assembling kernel source: waitpid failed."); }
	}

	if (status != 0) { MLOPEN_THROW("Error assembling kernel source, clang error code " + std::to_string(status)); }

	std::ifstream file(outfile, std::ios::binary | std::ios::ate);
	const auto size = file.tellg();

	source.resize(size, ' ');
	file.seekg(std::ios::beg);
	file.rdbuf()->sgetn(&source[0], size);
	file.close();
	std::remove(outfile);
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

ClProgramPtr LoadProgram(cl_context ctx, cl_device_id device, const std::string &program_name, const std::string& params)
{
	bool is_binary;
	auto source = mlopen::GetKernelSrc(program_name);
	auto is_asm = mlopen::EndsWith(program_name, ".s");

	if (is_asm)
	{
		// Overwrites source (asm text) by binary results of assembly:
		ExperimentalAmdgcnAssemble(source, params);
		is_binary = true;
	}
	else
	{
		is_binary = mlopen::EndsWith(program_name, ".so");
	}

	cl_program result;
	auto char_source = source.data();
	auto size = source.size();

	if (is_binary)
		result = CreateProgramWithBinary(ctx, device, char_source, size);
	else
		result = CreateProgram(ctx, char_source, size);

	if (is_binary)
		BuildProgram(result, device);
	else
		BuildProgram(result, device, params + " -cl-std=CL1.2");

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
