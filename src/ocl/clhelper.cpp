/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/clhelper.hpp>
#include <miopen/errors.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/kernel.hpp>
#include <miopen/kernel_warnings.hpp>
#include <miopen/logger.hpp>
#include <miopen/mlir_build.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/ocldeviceinfo.hpp>
#include <miopen/rocm_features.hpp>
#include <miopen/tmp_dir.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/write_file.hpp>
#include <miopen/env.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP)

namespace miopen {

#if WORKAROUND_MLOPEN_ISSUE_1711
void WorkaroundIssue1711(std::string& name)
{
    auto loc_p = name.find('+');
    if(loc_p != std::string::npos)
        name = name.substr(0, loc_p);
}
#endif

static cl_program CreateProgram(cl_context ctx, const char* char_source, size_t size)
{
    cl_int status;
    auto result = clCreateProgramWithSource(ctx, 1, &char_source, &size, &status);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status,
                               "Error Creating OpenCL Program (cl_program) in LoadProgram()");
    }

    return result;
}

static std::string ClAssemble(cl_device_id device,
                              const std::string& source,
                              const std::string& params,
                              const TargetProperties& target)
{
    std::string name = miopen::GetDeviceInfo<CL_DEVICE_NAME>(device);
#if WORKAROUND_MLOPEN_ISSUE_1711
    WorkaroundIssue1711(name);
#endif
    return AmdgcnAssemble(source, std::string("-mcpu=") + name + " " + params, target);
}

static cl_program
CreateProgramWithBinary(cl_context ctx, cl_device_id device, const char* char_source, size_t size)
{
    cl_int status, binaryStatus;
    auto result = clCreateProgramWithBinary(ctx,
                                            1,
                                            &device,
                                            reinterpret_cast<const size_t*>(&size),
                                            reinterpret_cast<const unsigned char**>(&char_source),
                                            &status,
                                            &binaryStatus);

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(
            status, "Error creating code object program (cl_program) in LoadProgramFromBinary()");
    }

    return result;
}

static std::string BuildProgramInfo(cl_program program, cl_device_id device)
{
    std::vector<char> errorbuf(1024 * 1024);
    size_t psize;
    clGetProgramBuildInfo(
        program, device, CL_PROGRAM_BUILD_LOG, 1024 * 1024, errorbuf.data(), &psize);
    return errorbuf.data();
}

static void BuildProgram(cl_program program, cl_device_id device, const std::string& params = "")
{
    auto status = clBuildProgram(program, 1, &device, params.c_str(), nullptr, nullptr);

#if MIOPEN_INSTALLABLE
    // Do not show messages (warnings etc) to the end users after successful builds.
    // Show everything to developers.
    if(status != CL_SUCCESS)
#endif
    {
        auto msg = BuildProgramInfo(program, device);
        if(!msg.empty())
            MIOPEN_LOG_WE("Build log: " << msg);
    }

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "clBuildProgram() failed:");
    }
}

ClProgramPtr LoadBinaryProgram(cl_context ctx, cl_device_id device, const std::string& source)
{
    ClProgramPtr result{CreateProgramWithBinary(ctx, device, source.data(), source.size())};
    BuildProgram(result.get(), device);
    return result;
}

ClProgramPtr LoadBinaryProgram(cl_context ctx, cl_device_id device, const std::vector<char>& source)
{
    ClProgramPtr result{CreateProgramWithBinary(ctx, device, source.data(), source.size())};
    BuildProgram(result.get(), device);
    return result;
}

ClProgramPtr LoadProgram(cl_context ctx,
                         cl_device_id device,
                         const TargetProperties& target,
                         const std::string& program,
                         std::string params,
                         bool is_kernel_str,
                         const std::string& kernel_src)
{
    std::string source;
    std::string program_name;

    if(is_kernel_str)
    {
        source       = program;
        program_name = "(unknown)";
    }
    else
    {
        program_name = program;
        // For mlir build, leave both source and kernel_src to be empty
        if((kernel_src.empty()) && !(miopen::EndsWith(program_name, ".mlir")))
            source = miopen::GetKernelSrc(program_name);
        else
            source = kernel_src;
    }

    bool load_binary = false;
    if(miopen::EndsWith(program_name, ".s"))
    {
        source      = ClAssemble(device, source, params, target); // Puts output binary into source.
        load_binary = true;
    }

    if(load_binary || miopen::EndsWith(program_name, ".so"))
        return LoadBinaryProgram(ctx, device, source);

    if(miopen::EndsWith(program_name, ".cpp"))
    {
        boost::optional<miopen::TmpDir> dir(program_name);
#if MIOPEN_BUILD_DEV
        params += " -Werror";
#ifdef __linux__
        params += HipKernelWarningsString();
#endif
#endif
        auto hsaco_file = HipBuild(dir, program_name, source, params, target);
        // load the hsaco file as a data stream and then load the binary
        std::string buf;
        bin_file_to_str(hsaco_file, buf);
        return LoadBinaryProgram(ctx, device, buf);
    }
#if MIOPEN_USE_MLIR
    else if(miopen::EndsWith(program_name, ".mlir"))
    {
        std::vector<char> buffer;
        MiirGenBin(params, buffer);
        return LoadBinaryProgram(ctx, device, buffer);
    }
#endif
    else // OpenCL programs.
    {
        ClProgramPtr result{CreateProgram(ctx, source.data(), source.size())};
        if(miopen::IsEnabled(MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP{}))
            params += " -Wf,-mwavefrontsize64 -Wf,-mcumode";
#if MIOPEN_BUILD_DEV
        params += " -Werror";
#ifdef __linux__
        params += is_kernel_str ? MiopengemmWarningsString() : OclKernelWarningsString();
#endif
#endif
        params += " -cl-std=CL1.2";
        MIOPEN_LOG_I2("Building OpenCL program: '" << program_name << "', options: '" << params);
        BuildProgram(result.get(), device, params);
        return result;
    }
}

void GetProgramBinary(const ClProgramPtr& program, std::string& binary)
{
    size_t binary_size;
    clGetProgramInfo(program.get(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, nullptr);
    binary.resize(binary_size);
    char* src[1] = {&binary[0]};
    if(clGetProgramInfo(program.get(), CL_PROGRAM_BINARIES, sizeof(src), &src, nullptr) !=
       CL_SUCCESS)
        MIOPEN_THROW(miopenStatusInternalError, "Could not extract binary from program");
}

void SaveProgramBinary(const ClProgramPtr& program, const std::string& name)
{
    std::string binary;
    GetProgramBinary(program, binary);
    std::ofstream fout(name.c_str(), std::ios::out | std::ios::binary);
    fout.write(binary.data(), binary.size());
}

ClKernelPtr CreateKernel(cl_program program, const std::string& kernel_name)
{
    cl_int status;
    ClKernelPtr result{clCreateKernel(program, kernel_name.c_str(), &status)};

    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status);
    }

    return result;
}

cl_device_id GetDevice(cl_command_queue q)
{
    cl_device_id device;
    cl_int status =
        clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Error Getting Device Info from Queue in GetDevice()");
    }

    return device;
}

cl_context GetContext(cl_command_queue q)
{
    cl_context context;
    cl_int status =
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "Error Getting Device Info from Queue in GetDevice()");
    }
    return context;
}

} // namespace miopen
