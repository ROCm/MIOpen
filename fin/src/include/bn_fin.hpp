/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#ifndef GUARD_BN_FIN_HPP
#define GUARD_BN_FIN_HPP

#include "error.hpp"
#include "fin.hpp"
#include "tensor.hpp"

#include <miopen/execution_context.hpp>
#include <miopen/miopen.h>
#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batchnorm/solvers.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/solver.hpp>
#include <miopen/solver_id.hpp>

#include <nlohmann/json.hpp>

#define EPSILON 1e-3

namespace fin {

using json = nlohmann::json;
template <typename Tgpu, typename Tcpu>
class BNFin : public BaseFin
{
    public:
    BNFin() : BaseFin() {}
    BNFin(json _job) : BaseFin(), job(_job)
    {
        if(job.contains("config"))
            PrepBatchNorm();
    }

    void PrepBatchNorm()
    {
        BaseFin::VerifyDevProps(job["arch"], job["num_cu"]);
        command         = job["config"];
        command["bias"] = 0;
        SetBNDescriptor();
        is_fwd_train = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 1);
        is_fwd_infer = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 2);
        is_bwd       = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 4);
    }

    // Getters and setters
    std::vector<int> GetInputTensorLengths();
    std::vector<int> GetBiasTensorLengths();
    int SetBNDescriptor();
    miopen::conv::Direction GetDirection() const;

    int ProcessStep(const std::string& step_name) override;

    // Steps
    int TestApplicability();
    int GetandSetData();
    std::vector<miopen::solver::ConvSolution> GetBNSolutions(miopen::ExecutionContext& ctx);
    miopen::batchnorm::ProblemDescription GetProblemDescription();
    auto GetAlgorithm();
    int MIOpenFindCompile();

    // Utility functions
    auto GetFwdTrainSolvers();
    auto GetFwdInferSolvers();
    auto GetBwdSolvers();

    json command;
    json job;

    miopenBatchNormMode_t bn_mode;
    std::vector<std::string> steps_processed;
    bool saveMeanVar        = false;
    bool keepRunningMeanVar = false;
    double epsilon          = 1.0;
    double expAvgFactor     = 1.0;
    bool isDepthSpecified   = false;
    bool is_fwd_train       = true;
    bool is_fwd_infer       = false;
    bool is_bwd             = false;

    tensor<Tgpu, Tcpu> inputTensor;
    tensor<Tgpu, Tcpu> outputTensor;
    tensor<Tgpu, Tcpu> biasScaleTensor;

    // for backward
    tensor<Tgpu, Tcpu> dyInputTensor;
    tensor<Tgpu, Tcpu> dxOutputTensor;
};

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::TestApplicability()
{
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif

    auto& handle = GetHandle();
    auto ctx     = miopen::ExecutionContext(&handle);
#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif
    ctx.SetStream(&handle);
    ctx.DetectRocm();

    std::vector<std::string> app_solvers;

    for(const auto& sln : GetBNSolutions(ctx))
    {
        std::cerr << sln.solver_id << std::endl;
        if(!sln.invoker_factory)
        {
            MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + sln.solver_id);
        }
        app_solvers.push_back(sln.solver_id);
    }
    for(auto& elem : app_solvers)
    {
        std::cerr << elem << std::endl;
    }

    output["applicable_solvers"] = app_solvers;
    return 0;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::GetandSetData()
{

    SetBNDescriptor();

    auto in_len = GetInputTensorLengths();

    std::vector<int> sb_len;
    if(bn_mode == miopenBNPerActivation)
    {
        // 1xCxHxW | in_len.size = 4
        sb_len = {1, in_len[1], in_len[2], in_len[3]};

        // 1xCxDxHxW | in_len.size = 5
        if(in_len.size() == 5)
        {
            sb_len.push_back(in_len[4]);
        }
    }
    else if(bn_mode == miopenBNSpatial)
    { // 1xCx1x1
        sb_len = {1, in_len[1], 1, 1};

        // 1xCx1x1x1
        if(in_len.size() == 5)
        {
            sb_len.push_back(1);
        }
    }
    if(command["bias"].get<int>() != 0)
    {
        biasScaleTensor = {GetHandle().GetStream(), GetBiasTensorLengths(), true, true};
    }
    else
    {
        biasScaleTensor = {GetHandle().GetStream(), sb_len, true, true};
    }

    inputTensor  = {GetHandle().GetStream(), in_len, true, false};
    outputTensor = {GetHandle().GetStream(), in_len, false, true};

    // backwards
    dyInputTensor  = {GetHandle().GetStream(), in_len, false, true};
    dxOutputTensor = {GetHandle().GetStream(), in_len, true, false};
    return (0);
}

template <typename Tgpu, typename Tref>
std::vector<int> BNFin<Tgpu, Tref>::GetInputTensorLengths()
{
    int in_n = command["batchsize"];
    int in_c = command["in_channels"];
    int in_h = command["in_h"];
    int in_w = command["in_w"];
    int in_d = command["in_d"];

    if(command["in_d"] > 1)
    {
        isDepthSpecified = true;
        // NxCxDxHxW -> NxCx(D*H)xW
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else
    {
        isDepthSpecified = false;
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
}

template <typename Tgpu, typename Tref>
std::vector<int> BNFin<Tgpu, Tref>::GetBiasTensorLengths()
{
    int spatial_dim = 2;
    if(command["in_d"] > 1)
    {
        spatial_dim = 3;
    }

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = command["out_channels"];

    return bias_lens;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::ProcessStep(const std::string& step_name)
{
    steps_processed.push_back(step_name);
    if(step_name == "applicability")
    {
        return TestApplicability();
    }
    if(step_name == "miopen_find_compile")
    {
        return MIOpenFindCompile();
    }
    return 0;
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::SetBNDescriptor()
{
    // batch norm mode type
    bn_mode = command["mode"] == 0 ? miopenBNPerActivation : miopenBNSpatial;

    // save off mean and variance?
    saveMeanVar = command["save"] == 0 ? false : true;

    // keep running mean and variance
    keepRunningMeanVar = command["run"] == 0 ? false : true;

    epsilon = 1;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
auto BNFin<Tgpu, Tref>::GetFwdTrainSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::batchnorm::BnFwdTrainingSpatialSingle,
                                           miopen::solver::batchnorm::BnFwdTrainingSpatialMultiple,
                                           miopen::solver::batchnorm::BnFwdTrainingPerActivation>{};
}

template <typename Tgpu, typename Tref>
auto BNFin<Tgpu, Tref>::GetFwdInferSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::batchnorm::BnFwdInference>{};
}

template <typename Tgpu, typename Tref>
auto BNFin<Tgpu, Tref>::GetBwdSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::batchnorm::BnBwdTrainingSpatialSingle,
                                           miopen::solver::batchnorm::BnBwdTrainingSpatialMultiple,
                                           miopen::solver::batchnorm::BnBwdTrainingPerActivation>{};
}

template <typename Tgpu, typename Tref>
miopen::batchnorm::ProblemDescription BNFin<Tgpu, Tref>::GetProblemDescription()
{
    if(is_fwd_train)
    {
        return miopen::batchnorm::ProblemDescription{bn_mode,
                                                     inputTensor.desc,
                                                     outputTensor.desc,
                                                     biasScaleTensor.desc,
                                                     expAvgFactor,
                                                     epsilon,
                                                     saveMeanVar,
                                                     keepRunningMeanVar};
    }
    else if(is_fwd_infer)
    {
        return miopen::batchnorm::ProblemDescription(
            bn_mode, inputTensor.desc, outputTensor.desc, biasScaleTensor.desc, epsilon);
    }
    else if(is_bwd)
    {
        return miopen::batchnorm::ProblemDescription(bn_mode,
                                                     inputTensor.desc,
                                                     dyInputTensor.desc,
                                                     dxOutputTensor.desc,
                                                     biasScaleTensor.desc,
                                                     epsilon,
                                                     saveMeanVar);
    }
    else
    {
        throw std::runtime_error("Unable to get solvers for batch norm");
    }
}

template <typename Tgpu, typename Tref>
std::vector<miopen::solver::ConvSolution>
BNFin<Tgpu, Tref>::GetBNSolutions(miopen::ExecutionContext& ctx)
{
    const auto problem = GetProblemDescription();
    if(is_fwd_train)
    {
        return GetFwdTrainSolvers().SearchForSolutions(ctx, problem, 1);
    }
    else if(is_fwd_infer)
    {
        return GetFwdInferSolvers().SearchForSolutions(ctx, problem, 1);
    }
    else if(is_bwd)
    {
        return GetBwdSolvers().SearchForSolutions(ctx, problem, 1);
    }
    else
    {
        throw std::runtime_error("Unable to to get solutions for batch norm");
    }
}

template <typename Tgpu, typename Tref>
auto BNFin<Tgpu, Tref>::GetAlgorithm()
{
    if(is_fwd_train)
    {
        return bn_mode == miopenBNSpatial
                   ? miopen::AlgorithmName{"miopenBatchNormForwardTrainingSpatial"}
                   : miopen::AlgorithmName{"miopenBatchNormForwardTrainingPerActivation"};
    }
    else if(is_fwd_infer)
    {
        return miopen::AlgorithmName{"miopenBatchNormalizationForwardInference"};
    }
    else if(is_bwd)
    {
        return bn_mode == miopenBNSpatial
                   ? miopen::AlgorithmName{"miopenBatchNormBackwardPropSpatial"}
                   : miopen::AlgorithmName{"miopenBatchNormBackwardPropPerActivation"};
    }
    else
    {
        throw std::runtime_error("Unable to get solvers for batch norm");
    }
}

template <typename Tgpu, typename Tref>
int BNFin<Tgpu, Tref>::MIOpenFindCompile()
{
    std::cerr << "MIOpenFinCompile" << std::endl;
    std::cerr << "Processing command: " << command << std::endl;
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error(
        "Unable to perform MIOpenFindCompile MIOpen was not compiled using HIPNOGPU backend");
#endif
    auto& handle = GetHandle();
    auto ctx     = miopen::ExecutionContext(&handle);
    GetHandle().EnableProfiling(true);
#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "for Batch Norm find_compile");
#endif
    ctx.SetStream(&handle);
    ctx.DetectRocm();

    const auto problem         = GetProblemDescription();
    const auto network_config  = problem.MakeNetworkConfig();
    output["network_config"]   = network_config;
    output["db_key"]           = network_config.ToString();
    output["is_winograd_only"] = false;

    json find_result;
    std::cerr << "Job Arch: " << job["arch"]
              << ": Handle Arch: " << handle.GetTargetProperties().Name() << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"]
              << ": Handle Num Cu: " << handle.GetMaxComputeUnits() << std::endl;

    for(const auto& sln : GetBNSolutions(ctx))
    {
        // remove the user db files
        boost::filesystem::remove_all(miopen::GetCachePath(false));
        json res_item;
        res_item["solver_name"] = sln.solver_id;
        res_item["algorithm"]   = GetAlgorithm();

        res_item["workspace"] = sln.workspace_sz;
        std::vector<miopen::solver::KernelInfo> kernels;
        for(auto&& kernel : sln.construction_params)
        {
            kernels.push_back(kernel);
        }
        std::ignore      = miopen::solver::PrecompileKernels(handle, kernels);
        json kernel_list = json::array();
        for(const auto& k : kernels)
        {
            json kernel;
            auto comp_opts   = k.comp_options;
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
        res_item["reason"]         = "Success";
        res_item["find_compiled"]  = true;
        find_result.push_back(res_item);
    }
    output["miopen_find_compile_result"] = find_result;
    return 1;
}

} // namespace fin
#endif // GUARD_MIOPEN_BN_FIN_HPP
