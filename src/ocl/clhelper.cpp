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

static void Assemble(std::string& source, const std::string& params)
{
	const auto asm_path_env_p = std::getenv("MLOPEN_EXPERIMENTAL_GCN_ASM_PATH");

	std::vector<char*> args({
		asm_path_env_p,
		const_cast<char*>("-x"),
		const_cast<char*>("assembler"),
		const_cast<char*>("-target"),
		const_cast<char*>("amdgcn--amdhsa"),
		const_cast<char*>("-mcpu=fiji"), // FIXME set to actual architecture probably
	});

	std::istringstream iss(params);
	std::vector<std::string> paramsVector;
	std::string outPath("MLOpen_CLang_Out_XXXXXX");
	
	auto outFD = mkstemp(const_cast<char*>(outPath.c_str()));

	do
	{
		std::string param;
		iss >> param;
		paramsVector.push_back(param);
		args.push_back(const_cast<char*>(paramsVector.rbegin()->c_str()));
	} while (iss);

	args.push_back(const_cast<char*>("-"));
	args.push_back(const_cast<char*>("-o"));
	args.push_back(const_cast<char*>(outPath.c_str()));
	args.push_back(nullptr);

	static const int read_fd = 0;
	static const int write_fd = 1;
	static const int pipe_sides = 2;

	int pipes[pipe_sides];

	pipe(pipes);
	write(pipes[write_fd], source.data(), source.size());

	int status;
	pid_t pid = fork();

	if (pid == 0)
	{
		dup2(pipes[read_fd], STDIN_FILENO);
		close(pipes[read_fd]);
		close(pipes[write_fd]);

		execv(asm_path_env_p, args.data());
		_exit(EXIT_FAILURE);
	}
	else
	{
		close(pipes[read_fd]);
		close(pipes[write_fd]);

		if (pid < 0)
		{
			status = -1;
		}
		else
		{
			if (waitpid(pid, &status, 0) != pid)
				status = -1;
		}
	}

	if (status != 0)
	{
		std::ostringstream stringStream;
		stringStream << "Error assembling kernel source: ";

		if (status > 0)
			stringStream << "clang error code " << status;
		else
			stringStream << "unable to fork or call waitpid";

		MLOPEN_THROW(stringStream.str());
	}

	std::ifstream file(outPath, std::ios::binary | std::ios::ate);
	auto size = file.tellg();

	file.seekg(std::ios::beg);
	source.resize(size, ' ');
	file.rdbuf()->sgetn(const_cast<char*>(source.c_str()), size);
	file.close();
	close(outFD);

	std::remove(outPath.c_str());
}

static cl_program CreateBinaryProgram(cl_context ctx, cl_device_id device, const char* char_source, size_t size)
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
		Assemble(source, params);
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
		result = CreateBinaryProgram(ctx, device, char_source, size);
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
