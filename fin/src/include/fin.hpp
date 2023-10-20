/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all
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
#ifndef GUARD_FIN_HPP
#define GUARD_FIN_HPP

// using float16 = half_float::half;
#include "config.h"
#include "tensor.hpp"
#include "base64.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <miopen/kernel_cache.hpp>
#include <miopen/handle.hpp>
#include <miopen/nogpu/handle_impl.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/md5.hpp>
#include <miopen/bz2.hpp>
#include <miopen/binary_cache.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/load_file.hpp>
#include <numeric>
#include <vector>

using json = nlohmann::json;

#if FIN_BACKEND_OPENCL
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#elif FIN_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif

namespace fin {

const int INVOKE_LIMIT = 4;

class BaseFin
{
    public:
    BaseFin() {}
    virtual ~BaseFin() {}
    void Usage();
    std::string ParseBaseArg(const int argc, const char* argv[]);
    miopen::Handle& GetHandle()
    {
        static auto handle = miopen::Handle{};
        return handle;
    }
    miopenDataType_t GetDataType() { return data_type; }

#if FIN_BACKEND_OPENCL
    cl_command_queue& GetStream() { return q; }
#elif FIN_BACKEND_HIP
    hipStream_t& GetStream() { return q; }
#endif

    virtual int ProcessStep(const std::string& step_name) = 0;
    void
    InitNoGpuHandle(miopen::Handle& handle, const std::string& arch, const unsigned long num_cu);
    void VerifyDevProps(const std::string& in_arch, const unsigned long in_num_cu);

    json output;

    int GetSolverList()
    {
        std::vector<std::unordered_map<std::string, std::string>> solvers;
        for(const auto& id :
            miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
        {
            std::unordered_map<std::string, std::string> solver;
            solver["id"]      = std::to_string(id.Value());
            solver["name"]    = id.ToString();
            solver["tunable"] = "0";
            solver["dynamic"] = "0";
            solver["type"]    = "convolution";
            if(id.GetSolver().IsTunable())
                solver["tunable"] = "1";
            if(id.GetSolver().IsDynamic())
                solver["dynamic"] = "1";
            solvers.push_back(solver);
        }

        for(const auto& id :
            miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Batchnorm))
        {
            std::unordered_map<std::string, std::string> solver;
            solver["id"]      = std::to_string(id.Value());
            solver["name"]    = id.ToString();
            solver["tunable"] = "0";
            solver["dynamic"] = "0";
            solver["type"]    = "batch_norm";
            solvers.push_back(solver);
        }

        output["all_solvers"] = solvers;
        return 0;
    }

    json BuildJsonKernelList(const miopen::Handle& handle,
                             const std::vector<miopen::solver::KernelInfo>& kernels)
    {
        // Get the binary
        json kernel_list = json::array();
        for(const auto& kern : kernels)
        {
            json kernel;

            std::string comp_opts = kern.comp_options;
            if(!miopen::EndsWith(kern.kernel_file, ".mlir"))
            {
                comp_opts += " -mcpu=" + handle.GetDeviceName();
            }
            auto hsaco = miopen::LoadBinary(handle.GetTargetProperties(),
                                            handle.GetMaxComputeUnits(),
                                            kern.kernel_file,
                                            comp_opts,
                                            false);

            if(hsaco.empty())
            {
                auto p = handle.LoadProgram(kern.kernel_file, kern.comp_options, false, "");
                hsaco  = p.IsCodeObjectInMemory()
                             ? p.GetCodeObjectBlob()
                             : miopen::LoadFile(p.GetCodeObjectPathname().string());
                if(hsaco.empty())
                {
                    std::cerr << "Got empty code object" << std::endl;
                    throw std::runtime_error("Got empty code object");
                }
            }
            // Compress the blob
            auto md5_sum             = miopen::md5(hsaco);
            auto size                = hsaco.size();
            bool success             = false;
            auto compressed_hsaco    = miopen::compress(hsaco, &success);
            const auto encoded_hsaco = base64_encode(compressed_hsaco);
            kernel["kernel_file"]    = kern.kernel_file;
            kernel["comp_options"]   = kern.comp_options;

            if(success)
            {
                kernel["uncompressed_size"] = size;
                kernel["md5_sum"]           = md5_sum;
                kernel["blob"]              = encoded_hsaco;
            }
            else
            {
                kernel["md5_sum"]           = "Failed to compress kernel";
                kernel["uncompressed_size"] = 0;
                kernel["blob"]              = "";
            }
            kernel_list.push_back(kernel);
            // std::cerr << "Successfully added new kernel to json output" << std::endl;
        }
        return kernel_list;
    }

    void SolutionHasProgram(const miopen::Handle& handle,
                            const miopen::solver::ConvSolution& solution)
    {
        for(auto& kern : solution.construction_params)
        {
            std::string kernel_file = kern.kernel_file;
            std::string comp_opts   = kern.comp_options;

            if(!miopen::EndsWith(kernel_file, ".o"))
            {
                std::cerr << "with added extensions ";
                if(!miopen::EndsWith(kernel_file, ".mlir"))
                    comp_opts += " -mcpu=" + handle.GetDeviceName();
                kernel_file += ".o";
            }

            std::cerr << "checking binary : " << kernel_file << " : " << comp_opts << std::endl;

            if(!handle.HasProgram(kernel_file, comp_opts))
            {
                std::cerr << "Binary object check failed, either tuning params have changed or "
                             "fin is unable to write binary to program cache"
                          << std::endl;
            }
        }
    }

    void UpdateSolutionOpts(const miopen::Handle& handle, miopen::solver::ConvSolution& solution)
    {
        for(auto& kern : solution.construction_params)
        {
            if(miopen::EndsWith(kern.kernel_file, ".o"))
                continue;
            if(!miopen::EndsWith(kern.kernel_file, ".mlir"))
                kern.comp_options += " -mcpu=" + handle.GetDeviceName();

            kern.kernel_file += ".o";
        }
    }

    float BenchmarkInvoker(const miopen::Invoker& invoker,
                           const miopen::Handle& h,
                           const miopen::conv::DataInvokeParams& invoke_ctx)
    {
        float kernel_time;
        std::vector<float> ktimes;
        // warmup run
        invoker(h, invoke_ctx);
        for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
        {
            invoker(h, invoke_ctx);
            kernel_time = h.GetKernelTime();
            ktimes.push_back(kernel_time);
            std::cerr << "kernel_time : " << kernel_time << std::endl;
        }
        sort(ktimes.begin(), ktimes.end());
        kernel_time = ktimes[(ktimes.size() - 1) / 2];
        std::cerr << "kernel_time median : " << kernel_time << std::endl;
        return kernel_time;
    }

    float BenchmarkInvoker(const miopen::Invoker& invoker,
                           const miopen::Handle& h,
                           const miopen::conv::WrWInvokeParams& invoke_ctx)
    {
        float kernel_time;
        std::vector<float> ktimes;
        // warmup run
        invoker(h, invoke_ctx);
        for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
        {
            invoker(h, invoke_ctx);
            kernel_time = h.GetKernelTime();
            ktimes.push_back(kernel_time);
            std::cerr << "kernel_time : " << kernel_time << std::endl;
        }
        sort(ktimes.begin(), ktimes.end());
        kernel_time = ktimes[(ktimes.size() - 1) / 2];
        std::cerr << "kernel_time median : " << kernel_time << std::endl;
        return kernel_time;
    }

    protected:
    template <typename Tgpu>
    void InitDataType();
    miopenDataType_t data_type = miopenFloat; // the datatype passed in through the command line

#if FIN_BACKEND_OPENCL
    cl_command_queue q;
#elif FIN_BACKEND_HIP
    hipStream_t q;
#endif
};

// "std::is_same<Tgpu, float>{}" used to avoid "static_assert" compilation
// error,
// which occurs when the condition does not depend in any way on the template
// parameters.
template <typename Tgpu>
void BaseFin::InitDataType()
{
    static_assert(std::is_same<Tgpu, float>{}, "unsupported Tgpu");
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vs)
{
    os << "{ size: " << vs.size() << ", entries: ";
    for(auto& v : vs)
        os << v << " ";
    os << "}";
    return os;
}
} // namespace fin
#endif // GUARD_FIN_HPP
