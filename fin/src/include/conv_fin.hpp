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
#ifndef GUARD_CONV_FIN_HPP
#define GUARD_CONV_FIN_HPP

#include "base64.hpp"
#include "error.hpp"
#include "fin.hpp"
#include "random.hpp"
#include "tensor.hpp"

#include <miopen/algorithm.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/binary_cache.hpp>
#include <miopen/bz2.hpp>
#include <miopen/conv/context.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/convolution.hpp>
#include <miopen/find_db.hpp>
#include <miopen/invoker.hpp>
#include <miopen/load_file.hpp>
#include <miopen/md5.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/solver_id.hpp>

#if MIOPEN_MODE_NOGPU
#include <miopen/kernel_cache.hpp>
#include <miopen/nogpu/handle_impl.hpp>
#endif

#include <boost/range/adaptor/sliced.hpp>
#include <boost/filesystem.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <vector>

namespace fin {

const int INVOKE_LIMIT = 2;
using json             = nlohmann::json;
// TODO: Create a config class to encapsulate config
// related code, such as checking direction etc
template <typename Tgpu, typename Tcpu>
class ConvFin : public Fin
{
    public:
    ConvFin() : Fin() {}
    ConvFin(json _job) : Fin(), job(_job)
    {
        if(job.contains("config"))
            PrepConvolution();
    }

    void VerifyDevProps()
    {
        std::cerr << "Verifying device properties" << std::endl;
        std::string arch    = job["arch"];
        arch                = arch.substr(0, arch.find(':'));
        const size_t num_cu = job["num_cu"];
        std::ignore         = num_cu;
        if(arch == "gfx900")
        {
            assert(num_cu == 56 || num_cu == 64);
        }
        else if(arch == "gfx906")
        {
            assert(num_cu == 60 || num_cu == 64);
        }
        else if(arch == "gfx908")
        {
            assert(num_cu == 120);
        }
        else if(arch == "gfx1030")
        {
            assert(num_cu == 72 || num_cu == 36);
        }
        else if(arch == "gfx90a")
        {
            assert(num_cu == 110);
        }
        else
            throw std::runtime_error("Invalid Arch Name");
    }

    void PrepConvolution()
    {
        VerifyDevProps();
        command         = job["config"];
        command["bias"] = 0;
        // timing is always enabled
        is_fwd = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 1);
        is_bwd = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 2);
        is_wrw = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 4);
        SetConvDescriptor();
        // workspace_dev = nullptr; // TODO: replaced with a tensor class
        // the variable name is implementation dependent, checking size instead
    }

    // Getters and setters
    std::vector<int> GetInputTensorLengths();
    std::vector<int> GetWeightTensorLengths();
    std::vector<int> GetBiasTensorLengths();
    int SetConvDescriptor();
    std::vector<size_t> GetOutputTensorLengths() const;
    miopenDataType_t GetOutputType() const
    {
        return (data_type == miopenInt8 || data_type == miopenInt8x4) ? miopenFloat : data_type;
    }
    miopen::conv::Direction GetDirection() const;

    int ProcessStep(const std::string& step_name) override;

    // Steps
    int AllocateBuffers();
    int CalcWorkspace();
    int FillBuffers();
    int CopyToDevice();
    int CopyFromDevice();
    int RunGPU();
    int TestApplicability();
    int TestPerfDbValid();
    int GetandSetData();
    int GetSolverList();
    int MIOpenFind();
    int MIOpenFindCompile();
    int MIOpenFindEval();

    // Utility functions
    bool IsInputTensorTransform() const;
    void InitNoGpuHandle(miopen::Handle& handle);
    json command;
    json job;

    tensor<Tgpu, Tcpu> inputTensor;
    tensor<Tgpu, Tcpu> inputTensor_vect4;
    tensor<Tgpu, Tcpu> outputTensor;
    tensor<Tgpu, Tcpu> weightTensor;
    tensor<Tgpu, Tcpu> weightTensor_vect4;
    tensor<Tgpu, Tcpu> biasTensor;
    tensor<Tgpu, Tcpu> workspace;
    miopen::ConvolutionDescriptor convDesc;

    bool wrw_allowed = 0, bwd_allowed = 0, forward_allowed = 1;
    bool is_fwd            = true;
    bool is_bwd            = false;
    bool is_wrw            = false; // TODO: check redundancy with above
    int immediate_solution = 0;
    std::vector<std::string> steps_processed;
};

template <typename Tgpu, typename Tref>
miopen::conv::Direction ConvFin<Tgpu, Tref>::GetDirection() const
{
    return is_fwd ? miopen::conv::Direction::Forward
                  : (is_bwd ? miopen::conv::Direction::BackwardData
                            : miopen::conv::Direction::BackwardWeights);
}

template <typename Tgpu, typename Tref>
void ConvFin<Tgpu, Tref>::InitNoGpuHandle(miopen::Handle& handle)
{
#if MIOPEN_MODE_NOGPU
    handle.impl->device_name        = job["arch"];
    handle.impl->num_cu             = job["num_cu"];
    handle.impl->max_mem_alloc_size = 32UL * 1024 * 1024 * 1024; // 32 GB
    handle.impl->global_mem_size    = 32UL * 1024 * 1024 * 1024;
    handle.impl->target_properties.Init(&handle);
#else
    std::ignore = handle;
#endif
}
template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::MIOpenFindCompile()
{
    std::cerr << "MIOpenFindCompile" << std::endl;
    std::cerr << "Processing command: " << command << std::endl;
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error(
        "Unable to perform MIOpenFindCompile MIOpen was not compiled using HIPNOGPU backend");
#endif
    const auto conv_dir = GetDirection();
    const miopen::ProblemDescription problem(
        inputTensor.desc, weightTensor.desc, outputTensor.desc, convDesc, conv_dir);
    GetHandle().EnableProfiling(true);
    auto ctx    = miopen::ConvolutionContext{problem};
    auto handle = miopen::Handle{};
#if MIOPEN_MODE_NOGPU
    InitNoGpuHandle(handle);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "for MIOpenFindCompile");
#endif
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    ctx.SetupFloats();

    const auto network_config   = ctx.BuildConfKey();
    const bool is_winograd_only = convDesc.IsWinograd3x3SupportedAndFast(ctx);
    output["is_winograd_only"]  = is_winograd_only;
    output["network_config"]    = network_config;
    std::ostringstream ss;
    problem.Serialize(ss);
    output["db_key"] = ss.str();

    auto db = GetDb(ctx);
    json find_result;
    const auto& tgt_props  = handle.GetTargetProperties();
    const std::string arch = tgt_props.Name();
    const size_t num_cu    = handle.GetMaxComputeUnits();
    std::cerr << "Job Arch: " << job["arch"] << ": Handle Arch: " << arch << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"] << ": Handle Num Cu: " << num_cu << std::endl;
    // since applicability has been run, the solver list should come from Tuna
    for(const auto& solver_id :
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
    {
        json res_item;
        // remove the user db files
        boost::filesystem::remove_all(miopen::GetCachePath(false));
        auto process_solver = [&]() -> bool {
            std::cerr << "Processing Solver: " << solver_id.ToString() << std::endl;
            res_item["solver_id"] = solver_id.ToString();
            if(solver_id.ToString() == "ConvBiasActivAsm1x1U" ||
               solver_id.ToString().find("Fused") != std::string::npos)
            {
                std::cerr << "Skipping fused solvers" << std::endl;
                return false;
            }
            const auto& s         = solver_id.GetSolver();
            const auto algo       = solver_id.GetAlgo(conv_dir);
            res_item["algorithm"] = algo;
            if(s.IsEmpty())
            {
                res_item["reason"] = "Empty Solver";
                std::cerr << "Skipping invalid solver: " << solver_id.ToString() << std::endl;
                return false;
            }
            if(!s.IsApplicable(ctx))
            {
                res_item["reason"] = "Not Applicable";
                std::cerr << "Skipping inapplicable solver: " << solver_id.ToString() << std::endl;
                return false;
            }
            miopen::solver::ConvSolution solution;
            try
            {
                solution = s.FindSolution(ctx, db, {}); // auto tune is not expected here
            }
            catch(const std::exception& e)
            {
                res_item["reason"] = std::string("Solver throws exception") + e.what();
                std::cerr << "Exception during solution construction, solver_name: "
                          << solver_id.ToString() << e.what() << std::endl;
                return true;
            }
            res_item["reason"]    = "Success";
            res_item["workspace"] = solution.workspace_sz;
            // Get the binary
            json kernel_list = json::array();
            for(const auto& k : solution.construction_params)
            {
                json kernel;
                auto comp_opts = k.comp_options;
                // if(comp_opts[0] != ' ')
                //     comp_opts    = ' ' + comp_opts;
                auto p           = handle.LoadProgram(k.kernel_file, comp_opts, false, "");
                const auto hsaco = p.IsCodeObjectInMemory()
                                       ? p.GetCodeObjectBlob()
                                       : miopen::LoadFile(p.GetCodeObjectPathname().string());
                if(hsaco.empty())
                {
                    std::cerr << "Got empty code object" << std::endl;
                    throw std::runtime_error("Got empty code object");
                }
                // Compress the blob
                auto md5_sum             = miopen::md5(hsaco);
                auto size                = hsaco.size();
                bool success             = false;
                auto compressed_hsaco    = miopen::compress(hsaco, &success);
                const auto encoded_hsaco = base64_encode(compressed_hsaco);
                kernel["kernel_file"]    = k.kernel_file;
                kernel["comp_options"]   = k.comp_options;
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
                std::cerr << "Successfully added new kernel" << std::endl;
            }
            res_item["kernel_objects"] = kernel_list;
            return true;
        };

        auto res = process_solver();
        if(res)
        {
            res_item["find_compiled"] = res;
            find_result.push_back(res_item);
        }
    }
    output["miopen_find_compile_result"] = find_result;
    return 1;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::MIOpenFindEval()
{
    std::cerr << "MIOpenFindEval" << std::endl;
    std::cerr << "Processing command: " << command << std::endl;
// Before this step is executed, the following steps should have been evaluated
// alloc_buf only if only timing is required
// alloc_buf, fill_buf and copy_buf_to_device if numerical accuracy would be
// checked ??
#if MIOPEN_MODE_NOGPU
    throw std::runtime_error("Unable to run MIOpenFindEval, Invalid MIOpen backend: HIPNOGPU");
#endif
    const auto conv_dir = GetDirection();
    // The first arg to the DataInvokeParams changes based on direction
    const miopen::ProblemDescription problem(
        inputTensor.desc, weightTensor.desc, outputTensor.desc, convDesc, conv_dir);
    GetHandle().EnableProfiling(true);
    auto ctx = miopen::ConvolutionContext{problem};
    auto& h  = GetHandle();
    ctx.SetStream(&(h));
    ctx.DetectRocm();
    ctx.SetupFloats();

    const auto network_config   = ctx.BuildConfKey();
    const bool is_winograd_only = convDesc.IsWinograd3x3SupportedAndFast(ctx);
    output["is_winograd_only"]  = is_winograd_only;
    output["network_config"]    = network_config;
    std::ostringstream ss;
    problem.Serialize(ss);
    output["db_key"] = ss.str();

    auto db = GetDb(ctx);
    json find_result;
    const auto& tgt_props  = h.GetTargetProperties();
    const std::string arch = tgt_props.Name();
    const size_t num_cu    = h.GetMaxComputeUnits();
    std::cerr << "Job Arch: " << job["arch"] << ": Handle Arch: " << arch << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"] << ": Handle Num Cu: " << num_cu << std::endl;
    for(const auto& kinder :
        job["miopen_find_compile_result"]) // The "miopen_find_compile_result" list generated
                                           // by miopen_find_compile operation
    {
        // Somehow the direction changes mid loop !
        json res_item;
        boost::system::error_code ec;
        boost::filesystem::remove_all(miopen::GetCachePath(false), ec);
        // boost::filesystem::remove_all(miopen::GetCachePath(true), ec);
        if(ec)
        {
            std::cerr << "Error while removing MIOpen cache: " << ec.message();
        }
        auto process_solver = [&]() -> bool {
            const std::string solver_name = kinder["solver_id"];
            std::cerr << "Processing solver: " << solver_name << std::endl;
            const auto solver_id    = miopen::solver::Id{solver_name};
            const auto& s           = solver_id.GetSolver();
            res_item["solver_name"] = solver_name;
            const auto algo         = solver_id.GetAlgo(conv_dir);
            res_item["algorithm"]   = algo;
            if(s.IsEmpty())
            {
                std::cerr << "Skipping invalid solver: " << solver_id.ToString() << std::endl;
                return false;
            }
            if(!s.IsApplicable(ctx))
            {
                std::cerr << "Solver inapplicable: " << solver_name << std::endl;
                throw std::runtime_error(
                    "InApplicable solver was sent to fin, check Tuna for errors");
                return false;
            }
            std::cerr << solver_name << " is applicable" << std::endl;
            const auto solution   = s.FindSolution(ctx, db, {}); // auto tune is not expected here
            res_item["workspace"] = solution.workspace_sz;
            // Get the binary
            std::cerr << "loading binaries from fin input" << std::endl;
            for(const auto& kernel_obj : kinder["kernel_objects"])
            {
                const auto size          = kernel_obj["uncompressed_size"];
                const auto md5_sum       = kernel_obj["md5_sum"];
                const auto encoded_hsaco = kernel_obj["blob"];
                const auto decoded_hsaco = base64_decode(encoded_hsaco);
                const auto hsaco         = miopen::decompress(decoded_hsaco, size);
                std::string comp_opts    = kernel_obj["comp_options"];
                std::string kernel_file  = kernel_obj["kernel_file"];
                if(miopen::md5(hsaco) == md5_sum)
                {
                    auto p = miopen::Program{kernel_file, hsaco};
                    h.AddProgram(p, kernel_file, comp_opts);
                }
                else
                {
                    std::cerr << "Corrupt Binary Object" << std::endl;
                    throw std::runtime_error("Corrupt binary object");
                    return false;
                }
            }
            for(const auto& kern : solution.construction_params)
            {
                if(!h.HasProgram(kern.kernel_file, kern.comp_options))
                {
                    std::cerr << "Binary object check failed, either tuning params have changed or "
                                 "fin is unable to write binary to program cache"
                              << std::endl;
                }
            }
            std::cerr << "Checking for workspace" << std::endl;
            if(solution.workspace_sz > workspace.desc.GetNumBytes())
            {
                std::cerr << "Allocating " << solution.workspace_sz << " bytes for workspace"
                          << std::endl;
                workspace = tensor<Tgpu, Tref>{
                    q,
                    std::vector<size_t>{static_cast<size_t>(solution.workspace_sz / sizeof(Tgpu))},
                    false,
                    false};
                workspace.AllocateBuffers();
            }
            if(!solution.invoker_factory)
            {
                std::cerr << "Invoker not implemeted" << std::endl;
                res_item["reason"] = "Invoker not implemented";
                return false;
            }
            try
            {
                std::cerr << "Preparing invokers" << std::endl;
                const auto invoker =
                    h.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
                // This is required because DataInvokeParams switches tensor order due to
                // direction and it does not have a
                // copy constructor or a default constructor
                std::cerr << "Finished preparing invokers" << std::endl;
                if(conv_dir == miopen::conv::Direction::Forward)
                {
                    const auto invoke_ctx =
                        miopen::conv::DataInvokeParams{{inputTensor.desc,
                                                        inputTensor.gpuData.buf.get(),
                                                        weightTensor.desc,
                                                        weightTensor.gpuData.buf.get(),
                                                        outputTensor.desc,
                                                        outputTensor.gpuData.buf.get()},
                                                       workspace.gpuData.buf.get(),
                                                       workspace.desc.GetNumBytes(),
                                                       convDesc.attribute.gfx90aFp16alt.GetFwd()};
                    for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
                        invoker(h, invoke_ctx);
                }
                else if(conv_dir == miopen::conv::Direction::BackwardData)
                {
                    const auto invoke_ctx =
                        miopen::conv::DataInvokeParams{{outputTensor.desc,
                                                        outputTensor.gpuData.buf.get(),
                                                        weightTensor.desc,
                                                        weightTensor.gpuData.buf.get(),
                                                        inputTensor.desc,
                                                        inputTensor.gpuData.buf.get()},
                                                       workspace.gpuData.buf.get(),
                                                       workspace.desc.GetNumBytes(),
                                                       convDesc.attribute.gfx90aFp16alt.GetBwd()};
                    for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
                        invoker(h, invoke_ctx);
                }
                else if(conv_dir == miopen::conv::Direction::BackwardWeights)
                {
                    const auto invoke_ctx =
                        miopen::conv::WrWInvokeParams{{outputTensor.desc,
                                                       outputTensor.gpuData.buf.get(),
                                                       inputTensor.desc,
                                                       inputTensor.gpuData.buf.get(),
                                                       weightTensor.desc,
                                                       weightTensor.gpuData.buf.get()},
                                                      workspace.gpuData.buf.get(),
                                                      workspace.desc.GetNumBytes(),
                                                      convDesc.attribute.gfx90aFp16alt.GetWrW()};
                    for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
                        invoker(h, invoke_ctx);
                }
                else
                {
                    throw std::runtime_error("Invalid Direction");
                }
            }
            catch(const std::exception& e)
            {
                res_item["reason"] = std::string("Invoker exeception: ") + e.what();
                return false;
            }
            const auto time    = h.GetKernelTime();
            res_item["time"]   = time;
            res_item["reason"] = "Success";

            return true;
        };

        auto res              = process_solver();
        res_item["evaluated"] = res;
        find_result.push_back(res_item);
    }
    output["miopen_find_eval_result"] = find_result;
    return 1;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::MIOpenFind()
{
    // Before this step is executed, the following steps should have been evaluted
    // alloc_buf only if only timing is required
    // alloc_buf, fill_buf and copy_buf_to_device if numerical accuracy would be checked ??
    const auto conv_dir = GetDirection();
    // assert(conv_dir == miopen::conv::Direction::Forward);
    // The first arg to the DataInvokeParams changes based on direction
    const miopen::ProblemDescription problem(
        inputTensor.desc, weightTensor.desc, outputTensor.desc, convDesc, conv_dir);
    GetHandle().EnableProfiling(true);
    auto ctx = miopen::ConvolutionContext{problem};
    auto& h  = GetHandle();
    ctx.SetStream(&(h));
    ctx.DetectRocm();
    ctx.SetupFloats();

    const auto network_config   = ctx.BuildConfKey();
    const bool is_winograd_only = convDesc.IsWinograd3x3SupportedAndFast(ctx);
    output["is_winograd_only"]  = is_winograd_only;
    output["network_config"]    = network_config;
    std::ostringstream ss;
    problem.Serialize(ss);
    output["db_key"] = ss.str();
    miopen::ConvolutionUserBuffers bufs(workspace.gpuData.buf.get(), workspace.desc.GetNumBytes());
    if(conv_dir == miopen::conv::Direction::Forward)
        bufs.SetFwd(inputTensor.gpuData.buf.get(),
                    weightTensor.gpuData.buf.get(),
                    outputTensor.gpuData.buf.get());
    else if(conv_dir == miopen::conv::Direction::BackwardData)
        bufs.SetBwd(inputTensor.gpuData.buf.get(),
                    weightTensor.gpuData.buf.get(),
                    outputTensor.gpuData.buf.get());
    else if(conv_dir == miopen::conv::Direction::BackwardWeights)
        bufs.SetWrW(inputTensor.gpuData.buf.get(),
                    weightTensor.gpuData.buf.get(),
                    outputTensor.gpuData.buf.get());

    ctx.SetBufs(bufs);

    auto db = GetDb(ctx);
    json find_result;
    const auto& tgt_props  = h.GetTargetProperties();
    const std::string arch = tgt_props.Name();
    assert(arch == job["arch"]);
    const size_t num_cu = h.GetMaxComputeUnits();
    assert(num_cu == job["num_cu"]);
    for(const auto& solver_id :
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
    {
        json res_item;
        auto process_solver = [&]() -> bool {
            res_item["solver_id"] = solver_id.ToString();
            const auto& s         = solver_id.GetSolver();
            const auto algo       = solver_id.GetAlgo(conv_dir);
            res_item["algorithm"] = algo;
            if(s.IsEmpty())
            {
                res_item["reason"] = "Empty Solver";
                std::cerr << "Skipping invalid solver: " << solver_id.ToString() << std::endl;
                return false;
            }
            if(!s.IsApplicable(ctx))
            {
                res_item["reason"] = "Not Applicable";
                return false;
            }
            const auto solution   = s.FindSolution(ctx, db, {}); // auto tune is not expected here
            res_item["workspace"] = solution.workspace_sz;
            // Get the binary
            miopen::solver::PrecompileKernels(h, solution.construction_params);
            json kernel_list = json::array();
            for(const auto& k : solution.construction_params)
            {
                json kernel;
                const auto hsaco = miopen::LoadBinary(
                    tgt_props, num_cu, k.kernel_file, k.comp_options + " -mcpu=" + arch, false);
                if(hsaco.empty())
                    throw std::runtime_error("Got empty code object");
                // Compress the blob
                auto md5_sum          = miopen::md5(hsaco);
                kernel["md5_sum"]     = md5_sum;
                auto size             = hsaco.size();
                bool success          = false;
                auto compressed_hsaco = miopen::compress(hsaco, &success);
                if(success)
                {
                    const auto encoded_hsaco    = base64_encode(compressed_hsaco);
                    kernel["uncompressed_size"] = size;
                    kernel["blob"]              = encoded_hsaco;
                }
                else
                {
                    const auto encoded_hsaco    = base64_encode(hsaco);
                    kernel["uncompressed_size"] = 0;
                    kernel["blob"]              = encoded_hsaco;
                }
                kernel_list.push_back(kernel);
            }
            res_item["kernel_objects"] = kernel_list;
            if(solution.workspace_sz > workspace.desc.GetNumBytes())
            {
                res_item["reason"] = "Insufficient Workspace";
                return false;
            }
            if(!solution.invoker_factory)
            {
                res_item["reason"] = "Invoker not implemented";
                return false;
            }
            try
            {
                const auto invoker =
                    h.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
                // This required because DataInvokeParams switches tensor order
                // due to direction and it does not have a
                // copy constructor or a default constructor
                if(conv_dir == miopen::conv::Direction::Forward)
                {
                    const auto invoke_ctx =
                        miopen::conv::DataInvokeParams{{inputTensor.desc,
                                                        inputTensor.gpuData.buf.get(),
                                                        weightTensor.desc,
                                                        weightTensor.gpuData.buf.get(),
                                                        outputTensor.desc,
                                                        outputTensor.gpuData.buf.get()},
                                                       workspace.gpuData.buf.get(),
                                                       workspace.desc.GetNumBytes(),
                                                       convDesc.attribute.gfx90aFp16alt.GetFwd()};
                    for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
                        invoker(h, invoke_ctx);
                }
                else if(conv_dir == miopen::conv::Direction::BackwardData)
                {
                    const auto invoke_ctx =
                        miopen::conv::DataInvokeParams{{outputTensor.desc,
                                                        outputTensor.gpuData.buf.get(),
                                                        weightTensor.desc,
                                                        weightTensor.gpuData.buf.get(),
                                                        inputTensor.desc,
                                                        inputTensor.gpuData.buf.get()},
                                                       workspace.gpuData.buf.get(),
                                                       workspace.desc.GetNumBytes(),
                                                       convDesc.attribute.gfx90aFp16alt.GetBwd()};
                    for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
                        invoker(h, invoke_ctx);
                }
                else if(conv_dir == miopen::conv::Direction::BackwardWeights)
                {
                    const auto invoke_ctx =
                        miopen::conv::WrWInvokeParams{{outputTensor.desc,
                                                       outputTensor.gpuData.buf.get(),
                                                       inputTensor.desc,
                                                       inputTensor.gpuData.buf.get(),
                                                       weightTensor.desc,
                                                       weightTensor.gpuData.buf.get()},
                                                      workspace.gpuData.buf.get(),
                                                      workspace.desc.GetNumBytes(),
                                                      convDesc.attribute.gfx90aFp16alt.GetWrW()};
                    for(auto idx = 0; idx < INVOKE_LIMIT; idx++)
                        invoker(h, invoke_ctx);
                }
                else
                {
                    throw std::runtime_error("Invalid Direction");
                }
            }
            catch(const std::exception& e)
            {
                res_item["reason"] = std::string("Invoker exeception: ") + e.what();
                return false;
            }
            const auto time    = h.GetKernelTime();
            res_item["time"]   = time;
            res_item["reason"] = "Success";

            return true;
        };

        auto res              = process_solver();
        res_item["evaluated"] = res;
        find_result.push_back(res_item);
    }

    output["miopen_find_result"] = find_result;
    return 1;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::TestApplicability()
{
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif
    const auto conv_dir = GetDirection();
    const miopen::ProblemDescription problem(
        inputTensor.desc, weightTensor.desc, outputTensor.desc, convDesc, conv_dir);
    auto ctx    = miopen::ConvolutionContext{problem};
    auto handle = miopen::Handle{};
#if MIOPEN_MODE_NOGPU
    InitNoGpuHandle(handle);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif

    ctx.SetStream(&handle);
    ctx.DetectRocm();
    ctx.SetupFloats();
    const auto network_config = ctx.BuildConfKey();
    std::vector<std::string> app_solvers;
    for(const auto& id :
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
    {
        std::cerr << "Testing: " << id.ToString() << std::endl;
        auto solver = id.GetSolver();
        if(id.IsValid() && !solver.IsEmpty())
        {
            try
            {
                if(solver.IsApplicable(ctx))
                    app_solvers.push_back(id.ToString());
            }
            catch(...)
            {
                std::cerr << id.ToString() << "(" << id.Value() << ")"
                          << " raised an exception"
                          << "for " << std::string(network_config) << " config: " << job
                          << std::endl;
            }
        }
        else
        {
            std::cerr << "Solver: " << id.ToString() << " is invalid or empty" << std::endl;
        }
    }
    output["applicable_solvers"] = app_solvers;
    return 0;
}

class ParamString
{
    std::string values;

    public:
    ParamString() {}
    ParamString(std::string in_val) : values(in_val) {}

    void Serialize(std::ostream& stream) const { stream << values; }
    bool Deserialize(const std::string& s)
    {
        values = s;
        return true;
    }
};

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::TestPerfDbValid()
{
    bool ret     = true;
    namespace fs = boost::filesystem;
    std::cout << miopen::GetSystemDbPath() << std::endl;

    std::vector<fs::path> contents;
    std::copy(fs::directory_iterator(miopen::GetSystemDbPath()),
              fs::directory_iterator(),
              std::back_inserter(contents));
    for(auto const& db_file : contents)
    {
        std::string pathstr = db_file.native();
        std::string filestr = db_file.filename().native();

        if(job["arch"].size() > 0 and job["num_cu"].size() > 0)
        {
            std::string arch = job["arch"];
            int num_cu       = job["num_cu"];
            std::stringstream db_name;
            db_name << arch;
            if(num_cu > 64)
                db_name << std::hex << num_cu << ".db";
            else
                db_name << "_" << num_cu << ".db";

            if(filestr.compare(db_name.str()) != 0)
                continue;
        }

        if(pathstr.compare(pathstr.size() - 3, 3, ".db") != 0)
            continue;

        std::cout << pathstr << "/" << filestr << std::endl;

        auto sql = miopen::SQLite{pathstr, true};

        // pull out records for all configs from perf_db
        std::unordered_map<std::string, std::unordered_map<std::string, miopen::DbRecord>> records;
        std::map<std::string, std::unordered_map<std::string, std::string>> perfdb_entries;
        std::vector<std::map<std::string, std::string>> err_list;
        auto select_query = "SELECT config, solver, params, id FROM perf_db;";
        auto stmt         = miopen::SQLite::Statement{sql, select_query};
        while(true)
        {
            auto rc = stmt.Step(sql);
            if(rc == SQLITE_ROW)
            {
                const auto config_id = stmt.ColumnText(0);
                const auto solver_id = stmt.ColumnText(1);
                const auto params    = stmt.ColumnText(2);
                const auto perf_id   = stmt.ColumnText(3);
                records[config_id][solver_id].SetValues(solver_id, ParamString(params));
                perfdb_entries[perf_id]["config"] = config_id;
                perfdb_entries[perf_id]["solver"] = solver_id;
            }
            else if(rc == SQLITE_DONE)
                break;
            else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
                MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
        }

        // iterate through each config
        for(auto it = perfdb_entries.begin(); it != perfdb_entries.end(); it++)
        {
            auto solver_nm = it->second["solver"];
            auto config_id = it->second["config"];
            auto record    = records.find(config_id)->second.find(solver_nm)->second;

            auto slv_id = miopen::solver::Id(solver_nm);
            if(!slv_id.IsValid())
            {
                std::map<std::string, std::string> err;
                err["perfdb_id"] = it->first;
                err["config"]    = config_id;
                err["solver"]    = solver_nm;
                err_list.push_back(err);
                ret = false;
                continue;
            }

            auto solver = slv_id.GetSolver();

            // check if the params in the record deserialize
            if(!solver.TestSysDbRecord(record))
            {
                std::map<std::string, std::string> err;
                err["perfdb_id"] = it->first;
                err["config"]    = config_id;
                err["solver"]    = solver_nm;
                err_list.push_back(err);
                ret = false;
            }
        }
        std::string listing = filestr + "_errors";
        output[listing]     = err_list;
    }

    if(ret)
        output["clear"] = "true";

    return ret;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::GetSolverList()
{
    // pair.first = id, pair. second = string id
    std::vector<std::unordered_map<std::string, std::string>> solvers;
    for(const auto& id :
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
    {
        std::unordered_map<std::string, std::string> solver;
        solver["id"]      = std::to_string(id.Value());
        solver["name"]    = id.ToString();
        solver["tunable"] = "0";
        solver["dynamic"] = "0";
        if(id.GetSolver().IsTunable())
            solver["tunable"] = "1";
        if(id.GetSolver().IsDynamic())
            solver["dynamic"] = "1";
        solvers.push_back(solver);
    }

    output["all_solvers"] = solvers;
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::RunGPU()
{
    assert(false);
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::CopyToDevice()
{
#if MIOPEN_MODE_NOGPU
    throw std::runtime_error("Unable to copy buffers to device with NOGPU backend");
    return -1;
#else
    auto status = inputTensor.ToDevice();
    status |= inputTensor_vect4.ToDevice();
    status |= weightTensor.ToDevice();
    status |= outputTensor.ToDevice();
    status |= biasTensor.ToDevice();
    status |= workspace.ToDevice();
    return status;
#endif
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::CopyFromDevice()
{
#if MIOPEN_MODE_NOGPU
    throw std::runtime_error("Unable to copy buffers to device with NOGPU backend");
    return -1;
#else
    auto status = inputTensor.FromDevice();
    status |= inputTensor_vect4.FromDevice();
    status |= weightTensor.FromDevice();
    status |= outputTensor.FromDevice();
    status |= biasTensor.FromDevice();
    status |= workspace.FromDevice();
    return status;
#endif
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::ProcessStep(const std::string& step_name)
{
    steps_processed.push_back(step_name);
    if(step_name == "alloc_buf")
        return AllocateBuffers();
    if(step_name == "fill_buf")
        return FillBuffers();
    if(step_name == "copy_buf_to_device")
        return CopyToDevice();
    if(step_name == "copy_buf_from_device")
        return CopyFromDevice();
    if(step_name == "applicability")
    {
        return TestApplicability();
    }
    if(step_name == "perf_db_test")
        return TestPerfDbValid();
    if(step_name == "get_solvers")
        return GetSolverList();
    if(step_name == "miopen_find")
    {
        return MIOpenFind();
    }
    if(step_name == "miopen_find_compile")
    {
        return MIOpenFindCompile();
    }
    if(step_name == "miopen_find_eval")
    {
        return MIOpenFindEval();
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::GetandSetData()
{
    auto in_len  = GetInputTensorLengths();
    auto wei_len = GetWeightTensorLengths();

    // auto y_type = GetOutputType();

    inputTensor = {GetHandle().GetStream(), in_len, (is_fwd || is_wrw), is_bwd};

    weightTensor = {GetHandle().GetStream(), wei_len, (is_fwd || is_bwd), is_wrw};
    // conv, input and weight tensor descriptors need to be set before we can know the
    // output lengths
    auto out_len = GetOutputTensorLengths();
    outputTensor = {GetHandle().GetStream(), out_len, (is_bwd || is_wrw), is_fwd};

    if(IsInputTensorTransform())
    {
        std::vector<int> in_len_v4(in_len.begin(), in_len.end());
        in_len_v4[1] = ((in_len[1] + 3) / 4) * 4;
        std::vector<int> wei_len_v4(wei_len.begin(), wei_len.end());
        wei_len_v4[1] = ((wei_len[1] + 3) / 4) * 4;

        inputTensor_vect4  = {GetHandle().GetStream(), in_len_v4, (is_fwd || is_wrw), is_bwd};
        weightTensor_vect4 = {GetHandle().GetStream(), wei_len_v4, (is_fwd || is_bwd), is_wrw};
    }

    // Conv Desc is already setup from the job descriptor

    if(command["bias"].get<int>() != 0)
    {
        auto bias_len = GetBiasTensorLengths();
        biasTensor    = {GetHandle().GetStream(), bias_len, true, true};
    }
    // TODO: further investigate the warmup iteration, I dont think its necessary and can be
    // GetHandle()d in the main execution loop

    return (0);
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvFin<Tgpu, Tref>::GetInputTensorLengths()
{
    std::vector<int> in_lens;

    int spatial_dim = command["spatial_dim"];
    in_lens.resize(2 + spatial_dim);

    in_lens[0] = command["batchsize"];
    in_lens[1] = command["in_channels"];

    auto in_spatial_lens = boost::adaptors::slice(in_lens, 2, 2 + spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0] = command["in_h"];
        in_spatial_lens[1] = command["in_w"];
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0] = command["in_d"];
        in_spatial_lens[1] = command["in_h"];
        in_spatial_lens[2] = command["in_w"];
    }
    else
    {
        FIN_THROW("unsupported convolution dimension");
    }

    return in_lens;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvFin<Tgpu, Tref>::GetWeightTensorLengths()
{
    std::vector<int> wei_lens;

    int spatial_dim = command["spatial_dim"];
    wei_lens.resize(2 + spatial_dim);

    auto wei_spatial_lens = boost::adaptors::slice(wei_lens, 2, 2 + spatial_dim);

    int group_count = std::max(int(command["group_count"]), 1);

    int wei_k_len = command["out_channels"];
    int wei_c_len = command["in_channels"];

    if(spatial_dim == 2)
    {
        wei_spatial_lens[0] = command["fil_h"];
        wei_spatial_lens[1] = command["fil_w"];
    }
    else if(spatial_dim == 3)
    {
        wei_spatial_lens[0] = command["fil_d"];
        wei_spatial_lens[1] = command["fil_h"];
        wei_spatial_lens[2] = command["fil_w"];
    }
    else
    {
        FIN_THROW("unsupported convolution dimension");
    }

    if(group_count > 1)
    {
        if(wei_c_len % group_count != 0 || wei_k_len % group_count != 0 ||
           group_count > wei_c_len || group_count > wei_k_len)
        {
            FIN_THROW("Invalid group number\n");
        }
    }

    miopenConvolutionMode_t mode;
    if((command["conv_mode"]) == "conv")
    {
        mode = miopenConvolution;
    }
    else if((command["conv_mode"]) == "trans")
    {
        mode = miopenTranspose;
    }
    else
    {
        FIN_THROW("Incorrect Convolution Mode\n");
    }

    if(mode == miopenTranspose)
    {
        wei_lens[0] = wei_c_len;
        wei_lens[1] = wei_k_len / group_count;
    }
    else
    {
        wei_lens[0] = wei_k_len;
        wei_lens[1] = wei_c_len / group_count;
    }

    return wei_lens;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvFin<Tgpu, Tref>::GetBiasTensorLengths()
{
    int spatial_dim = command["spatial_dim"];

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = command["out_channels"];

    return bias_lens;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::SetConvDescriptor()
{
    size_t spatial_dim = command["spatial_dim"];

    std::vector<int> in_spatial_lens(spatial_dim);
    std::vector<int> wei_spatial_lens(spatial_dim);
    std::vector<int> pads(spatial_dim);
    std::vector<int> conv_strides(spatial_dim);
    std::vector<int> conv_dilations(spatial_dim);
    std::vector<int> trans_output_pads(spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0]   = command["in_h"];
        in_spatial_lens[1]   = command["in_w"];
        wei_spatial_lens[0]  = command["fil_h"];
        wei_spatial_lens[1]  = command["fil_w"];
        pads[0]              = command["pad_h"];
        pads[1]              = command["pad_w"];
        conv_strides[0]      = command["conv_stride_h"];
        conv_strides[1]      = command["conv_stride_w"];
        conv_dilations[0]    = command["dilation_h"];
        conv_dilations[1]    = command["dilation_w"];
        trans_output_pads[0] = 0; // command["trans_output_pad_h"];
        trans_output_pads[1] = 0; // command["trans_output_pad_w"];
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0]   = command["in_d"];
        in_spatial_lens[1]   = command["in_h"];
        in_spatial_lens[2]   = command["in_w"];
        wei_spatial_lens[0]  = command["fil_d"];
        wei_spatial_lens[1]  = command["fil_h"];
        wei_spatial_lens[2]  = command["fil_w"];
        pads[0]              = command["pad_d"];
        pads[1]              = command["pad_h"];
        pads[2]              = command["pad_w"];
        conv_strides[0]      = command["conv_stride_d"];
        conv_strides[1]      = command["conv_stride_h"];
        conv_strides[2]      = command["conv_stride_w"];
        conv_dilations[0]    = command["dilation_d"];
        conv_dilations[1]    = command["dilation_h"];
        conv_dilations[2]    = command["dilation_w"];
        trans_output_pads[0] = 0; // command["trans_output_pad_d"];
        trans_output_pads[1] = 0; // command["trans_output_pad_h"];
        trans_output_pads[2] = 0; // command["trans_output_pad_w"];
    }
    else
    {
        FIN_THROW("unsupported convolution dimension");
    }

    int out_c       = command["out_channels"];
    int in_c        = command["in_channels"];
    int group_count = std::max(int(command["group_count"]), 1);

    if(group_count > 1)
    {
        if(in_c % group_count != 0 || out_c % group_count != 0 || group_count > in_c ||
           group_count > out_c)
        {
            printf("Invalid group number\n");
            exit(0);
        }
    }

    miopenConvolutionMode_t c_mode;
    if((command["conv_mode"]) == "conv")
    {
        c_mode = miopenConvolution;
    }
    else if((command["conv_mode"]) == "trans")
    {
        c_mode = miopenTranspose;
    }
    else
    {
        printf("Incorrect Convolution Mode\n");
        exit(0);
    }

    miopenPaddingMode_t p_mode = miopenPaddingDefault;
    if((command["pad_mode"]) == "same")
        p_mode = miopenPaddingSame;
    else if((command["pad_mode"]) == "valid")
        p_mode = miopenPaddingValid;

    // adjust padding based on user-defined padding mode
    if(c_mode == miopenConvolution &&
       (miopen::all_of(conv_dilations, [](auto v) { return v == 1; }) ||
        miopen::all_of(wei_spatial_lens, [](auto v) { return v == 1; })))
    {
        if(p_mode == miopenPaddingSame)
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] =
                    (in_spatial_lens[i] % conv_strides[i] == 0)
                        ? (std::max((wei_spatial_lens[i] - conv_strides[i]), 0))
                        : (std::max((wei_spatial_lens[i] - (in_spatial_lens[i] % conv_strides[i])),
                                    0));
                pads[i] /= 2;
            }
        }
        else if(p_mode == miopenPaddingValid)
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] = 0;
            }
        }
    }

    convDesc = miopen::ConvolutionDescriptor{spatial_dim,
                                             c_mode,
                                             p_mode,
                                             pads,
                                             conv_strides,
                                             conv_dilations,
                                             trans_output_pads,
                                             group_count};

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<size_t> ConvFin<Tgpu, Tref>::GetOutputTensorLengths() const
{
    return convDesc.GetForwardOutputTensor(inputTensor.desc, weightTensor.desc).GetLengths();
}

template <typename Tgpu, typename Tref>
bool ConvFin<Tgpu, Tref>::IsInputTensorTransform() const
{
    return (data_type == miopenInt8 && int(command["in_channels"]) % 4 != 0) ||
           data_type == miopenInt8x4;
}

namespace detail {

template <typename T>
T RanGenWeights()
{
    return RAN_GEN<T>(static_cast<T>(-0.5), static_cast<T>(0.5));
}

// Shift FP16 distribution towards positive numbers,
// otherwise Winograd FP16 validation fails.
template <>
float16 RanGenWeights()
{
    return RAN_GEN<float16>(static_cast<float16>(-1.0 / 3.0), static_cast<float16>(0.5));
}

} // namespace detail

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::AllocateBuffers()
{
#if MIOPEN_MODE_NOGPU
    throw std::runtime_error("Unable to allocate buffers with NOGPU backend");
#else
    GetandSetData();
    inputTensor.AllocateBuffers();
    inputTensor_vect4.AllocateBuffers();
    weightTensor.AllocateBuffers();
    outputTensor.AllocateBuffers();
    biasTensor.AllocateBuffers();
    // The workspace is actually allocated when the solver is about to be run
    // since it varies from solver to solver
    workspace.AllocateBuffers();
#endif
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::CalcWorkspace()
{
    // if(solver is known)
    // Find workspace for solver using the GetSolution mechanism
    // else
    // if(!immediate_solution)
    size_t ws_sizeof_find_fwd = 0;
    size_t ws_sizeof_find_wrw = 0;
    size_t ws_sizeof_find_bwd = 0;
    auto is_transform         = IsInputTensorTransform();

    if(is_wrw)
        ws_sizeof_find_wrw = convDesc.BackwardWeightsGetWorkSpaceSize(
            GetHandle(), outputTensor.desc, inputTensor.desc, weightTensor.desc);
    if(is_bwd)
    {
        ws_sizeof_find_bwd =
            (convDesc.mode == miopenTranspose)
                ? convDesc.ForwardGetWorkSpaceSize(
                      GetHandle(), weightTensor.desc, outputTensor.desc, inputTensor.desc)
                : convDesc.BackwardDataGetWorkSpaceSize(
                      GetHandle(), weightTensor.desc, outputTensor.desc, inputTensor.desc);
    }
    if(is_fwd)
    {
        ws_sizeof_find_fwd = (convDesc.mode == miopenTranspose)
                                 ? convDesc.BackwardDataGetWorkSpaceSize(
                                       GetHandle(),
                                       (is_transform ? weightTensor_vect4.desc : weightTensor.desc),
                                       (is_transform ? inputTensor_vect4.desc : inputTensor.desc),
                                       outputTensor.desc)
                                 : convDesc.ForwardGetWorkSpaceSize(
                                       GetHandle(),
                                       (is_transform ? weightTensor_vect4.desc : weightTensor.desc),
                                       (is_transform ? inputTensor_vect4.desc : inputTensor.desc),
                                       outputTensor.desc);
    }

    const auto wsSizeof =
        std::max(std::max(ws_sizeof_find_bwd, ws_sizeof_find_wrw), ws_sizeof_find_fwd);
    if(wsSizeof != 0)
        workspace = tensor<Tgpu, Tref>{q,
                                       std::vector<unsigned int>{static_cast<unsigned int>(
                                           std::ceil(wsSizeof / sizeof(Tgpu)))},
                                       true,
                                       false};
    return wsSizeof;
}

template <typename Tgpu>
Tgpu init_in(bool is_int8, size_t idx)
{
    (void)idx;
    if(is_int8)
    {
        float Data_scale = 127.0;
        return static_cast<Tgpu>(Data_scale *
                                 RAN_GEN<float>(static_cast<float>(0.0), static_cast<float>(1.0)));
    }
    else
    {
        Tgpu Data_scale = static_cast<Tgpu>(0.01);
        return Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
}

template <typename Tgpu>
Tgpu init_out(bool is_int8, size_t idx)
{
    (void)idx;
    if(is_int8)
    {
        return static_cast<Tgpu>(0); // int8 is inference only
    }
    else
    {
        Tgpu Data_scale = static_cast<Tgpu>(0.01);
        return Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
}

template <typename Tgpu>
Tgpu init_wei(bool is_int8, size_t idx)
{
    (void)idx;
    if(is_int8)
    {
        float Data_scale = 127.0;
        return static_cast<Tgpu>(Data_scale * 2 * detail::RanGenWeights<float>());
    }
    else
    {
        Tgpu Data_scale = static_cast<Tgpu>(0.01);
        return Data_scale * detail::RanGenWeights<Tgpu>();
    }
}

template <typename Tgpu>
Tgpu init_bias(bool is_int8, size_t idx)
{
    (void)idx;
    (void)is_int8;
    return static_cast<Tgpu>(idx % 8) +
           RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::FillBuffers()
{
#if MIOPEN_MODE_NOGPU
    throw std::runtime_error("Unable to fill buffers with NOGPU backend");
#else
    // TODO: Do we need to initialized tensors ?
    auto is_int8 = (data_type == miopenInt8 || data_type == miopenInt8x4);
    srand(0);

    inputTensor.FillBuffer(std::bind(init_in<Tgpu>, is_int8, std::placeholders::_1));
    outputTensor.FillBuffer(std::bind(init_out<Tgpu>, is_int8, std::placeholders::_1));
    weightTensor.FillBuffer(std::bind(init_wei<Tgpu>, is_int8, std::placeholders::_1));
    if(command["bias"].get<int>() != 0)
    {
        biasTensor.FillBuffer(std::bind(init_bias<Tgpu>, is_int8, std::placeholders::_1));
    }
#endif
    return 0;
}
} // namespace fin
#endif // GUARD_MIOPEN_CONV_FIN_HPP
