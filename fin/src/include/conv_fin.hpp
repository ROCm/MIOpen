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
#include <miopen/version.h>

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

#define MIOPEN_ALLSOLVER 1

namespace fin {

const int INVOKE_LIMIT = 2;
using json             = nlohmann::json;
// TODO: Create a config class to encapsulate config
// related code, such as checking direction etc
template <typename Tgpu, typename Tcpu>
class ConvFin : public BaseFin
{
    public:
    ConvFin() : BaseFin() {}
    ConvFin(json _job) : BaseFin(), job(_job)
    {
        if(job.contains("config"))
            PrepConvolution();
    }

    void PrepConvolution()
    {
        BaseFin::VerifyDevProps(job["arch"], job["num_cu"]);
        command         = job["config"];
        command["bias"] = 0;
        // timing is always enabled
        is_fwd = (command["direction"].get<std::string>().compare("F") == 0);
        is_bwd = (command["direction"].get<std::string>().compare("B") == 0);
        is_wrw = (command["direction"].get<std::string>().compare("W") == 0);
        SetConvDescriptor();
        // workspace_dev = nullptr; // TODO: replaced with a tensor class
        // the variable name is implementation dependent, checking size instead
    }

    // Getters and setters
    std::vector<int> GetInputTensorLengths();
    std::vector<int> GetWeightTensorLengths();
    std::vector<int> GetBiasTensorLengths();
    int SetConvDescriptor();

    miopen::ConvolutionContext GetCmdConvContext(json _command);
    miopen::ConvolutionContext BuildContext(miopen::SQLite& sql, std::string config_id);

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

    int TestPerfDbEntries(
        const std::string config_id,
        const miopen::ConvolutionContext& ctx,
        const std::map<std::string, std::unordered_map<std::string, std::string>>& perf_ids,
        const std::unordered_map<std::string, miopen::DbRecord>& records,
        std::vector<std::map<std::string, std::string>>& err_list,
        std::vector<std::string>& pdb_id);

    int TestPerfDbValid();
    int GetandSetData();
    int MIOpenFind();
    int MIOpenFindCompile();
    int MIOpenFindEval();
    // function used to Search the Precompiled Kernels
    int SearchPreCompiledKernels();
    int MIOpenPerfCompile();
    int MIOpenPerfEval();

    // Utility functions
    bool IsInputTensorTransform() const;
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
int ConvFin<Tgpu, Tref>::MIOpenPerfCompile()
{
#if MIOPEN_ALLSOLVER
    std::cerr << "MIOpenPerfCompile" << std::endl;
    std::cerr << "Processing command: " << command << std::endl;
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error(
        "Unable to perform MIOpenPerfCompile MIOpen was not compiled using HIPNOGPU backend");
#endif
    const auto conv_dir = GetDirection();
    const miopen::ProblemDescription problem(
        inputTensor.desc, weightTensor.desc, outputTensor.desc, convDesc, conv_dir);
    GetHandle().EnableProfiling(true);
    // cppcheck-suppress unreadVariable
    auto ctx = miopen::ConvolutionContext{problem};
    // cppcheck-suppress unreadVariable
    auto handle = miopen::Handle{};
#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "for MIOpenPerfCompile");
#endif
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    ctx.SetupFloats();

    const auto network_config   = ctx.problem.BuildConfKey();
    const bool is_winograd_only = convDesc.IsWinograd3x3SupportedAndFast(ctx);
    output["is_winograd_only"]  = is_winograd_only;
    output["network_config"]    = network_config;
    std::ostringstream ss;
    problem.Serialize(ss);
    output["db_key"] = ss.str();

    auto db = GetDb(ctx);
    json perf_result;
    const auto& tgt_props  = handle.GetTargetProperties();
    const std::string arch = tgt_props.Name();
    const size_t num_cu    = handle.GetMaxComputeUnits();
    std::cerr << "Job Arch: " << job["arch"] << ": Handle Arch: " << arch << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"] << ": Handle Num Cu: " << num_cu << std::endl;

    std::vector<miopen::solver::Id> solver_list;
    if(job.contains("solvers"))
        for(std::string solver_str : job["solvers"]) // cppcheck-suppress useStlAlgorithm
            solver_list.push_back(miopen::solver::Id(solver_str));
    else
        solver_list = miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution);

    for(const auto& solver_id : solver_list)
    {
        json res_item;
        // remove the user db files
        boost::filesystem::remove_all(miopen::GetCachePath(false));
        auto process_solver = [&]() -> bool {
            std::cerr << "Processing Solver: " << solver_id.ToString() << std::endl;
            const auto& s           = solver_id.GetSolver();
            const auto algo         = solver_id.GetAlgo(conv_dir);
            res_item["solver_name"] = solver_id.ToString();
            res_item["algorithm"]   = algo;

            if(solver_id.ToString() == "ConvBiasActivAsm1x1U" ||
               solver_id.ToString().find("Fused") != std::string::npos)
            {
                res_item["reason"] = "Skip Fused";
                std::cerr << "Skipping fused solvers" << std::endl;
                return false;
            }
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
            res_item["tunable"] = true;
            if(!s.IsTunable())
                res_item["tunable"] = false;

            std::vector<miopen::solver::ConvSolution> all_solutions;
            if(s.IsTunable())
            {
                try
                {
                    all_solutions = s.GetAllSolutions(ctx);
                }
                catch(const std::exception& e)
                {
                    res_item["reason"] = "Failed getting solutions";
                    std::cerr << "Error getting solutions: " << e.what() << std::endl;
                    return false;
                }
            }
            else
                all_solutions.push_back(s.FindSolution(ctx, db, {}));

            // PrecompileKernels call saves to binary_cache,
            // this needs to be escaped if KERN_CACHE is not on.
            std::vector<miopen::solver::KernelInfo> kernels;
            for(const auto& current_solution : all_solutions)
                for(auto&& kernel :
                    current_solution.construction_params) // cppcheck-suppress useStlAlgorithm
                    kernels.push_back(kernel);
            std::ignore = miopen::solver::PrecompileKernels(handle, kernels);

            res_item["reason"]         = "Success";
            res_item["kernel_objects"] = BuildJsonKernelList(handle, kernels);
            return true;
        };

        auto res = process_solver();
        if(res)
        {
            res_item["perf_compiled"] = res;
            perf_result.push_back(res_item);
        }
    }
    output["miopen_perf_compile_result"] = perf_result;
#else
    throw std::runtime_error("Unsupported feature");
#endif
    return 1;
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
    // cppcheck-suppress unreadVariable
    auto ctx = miopen::ConvolutionContext{problem};
    // cppcheck-suppress unreadVariable
    auto handle = miopen::Handle{};
#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "for MIOpenFindCompile");
#endif
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    ctx.SetupFloats();

    const auto network_config   = ctx.problem.BuildConfKey();
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
    bool dynamic_only = false;
    if(job.contains("dynamic_only"))
        dynamic_only = job["dynamic_only"];

    std::vector<miopen::solver::Id> solver_list;
    if(job.contains("solvers"))
        for(std::string solver_str : job["solvers"]) // cppcheck-suppress useStlAlgorithm
            solver_list.push_back(miopen::solver::Id(solver_str));
    else
        solver_list = miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution);

    // since applicability has been run, the solver list should come from Tuna
    for(const auto& solver_id : solver_list)
    {
        json res_item;
        // remove the user db files
        boost::filesystem::remove_all(miopen::GetCachePath(false));
        auto process_solver = [&]() -> bool {
            std::cerr << "Processing Solver: " << solver_id.ToString() << std::endl;
            const auto& s           = solver_id.GetSolver();
            const auto algo         = solver_id.GetAlgo(conv_dir);
            res_item["solver_name"] = solver_id.ToString();
            res_item["algorithm"]   = algo;

            if(solver_id.ToString() == "ConvBiasActivAsm1x1U" ||
               solver_id.ToString().find("Fused") != std::string::npos)
            {
                res_item["reason"] = "Skip Fused";
                std::cerr << "Skipping fused solvers" << std::endl;
                return false;
            }
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
            if(dynamic_only && !s.IsDynamic())
            {
                res_item["reason"] = "Not Dynamic";
                std::cerr << "Skipping static solver: " << solver_id.ToString() << std::endl;
                return false;
            }

            res_item["params"]  = s.GetPerfCfgParams(ctx, db);
            res_item["tunable"] = false;
            if(s.IsTunable())
                res_item["tunable"] = true;

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
                return false;
            }
            res_item["reason"]         = "Success";
            res_item["workspace"]      = solution.workspace_sz;
            res_item["kernel_objects"] = BuildJsonKernelList(handle, solution.construction_params);
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
int ConvFin<Tgpu, Tref>::MIOpenPerfEval()
{
#if MIOPEN_ALLSOLVER
    std::cerr << "MIOpenPerfEval" << std::endl;
    std::cerr << "Processing command: " << command << std::endl;
// Before this step is executed, the following steps should have been evaluated
// alloc_buf only if only timing is required
// alloc_buf, fill_buf and copy_buf_to_device if numerical accuracy would be
// checked ??
#if MIOPEN_MODE_NOGPU
    throw std::runtime_error("Unable to run MIOpenPerfEval, Invalid MIOpen backend: HIPNOGPU");
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

    const auto network_config   = ctx.problem.BuildConfKey();
    const bool is_winograd_only = convDesc.IsWinograd3x3SupportedAndFast(ctx);
    output["is_winograd_only"]  = is_winograd_only;
    output["network_config"]    = network_config;
    std::ostringstream ss;
    problem.Serialize(ss);
    output["db_key"] = ss.str();

    auto db = GetDb(ctx);
    json perf_result;
    const auto& tgt_props  = h.GetTargetProperties();
    const std::string arch = tgt_props.Name();
    const size_t num_cu    = h.GetMaxComputeUnits();
    std::cerr << "Job Arch: " << job["arch"] << ": Handle Arch: " << arch << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"] << ": Handle Num Cu: " << num_cu << std::endl;

    for(const auto& kinder :
        job["miopen_perf_compile_result"]) // The "miopen_perf_compile_result" list generated
                                           // by miopen_perf_compile operation
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
            const std::string solver_name = kinder["solver_name"];
            std::cerr << "Processing solver: " << solver_name << std::endl;
            const auto solver_id    = miopen::solver::Id{solver_name};
            const auto& s           = solver_id.GetSolver();
            const auto algo         = solver_id.GetAlgo(conv_dir);
            res_item["solver_name"] = solver_name;
            res_item["algorithm"]   = algo;
            std::string params      = "";
            json kern_objs;

            if(solver_id.ToString() == "ConvBiasActivAsm1x1U" ||
               solver_id.ToString().find("Fused") != std::string::npos)
            {
                res_item["reason"] = "Skip Fused";
                std::cerr << "Skipping fused solvers" << std::endl;
                return false;
            }
            if(s.IsEmpty())
            {
                res_item["reason"] = "Empty Solver";
                std::cerr << "Skipping invalid solver: " << solver_id.ToString() << std::endl;
                return false;
            }
            if(!s.IsApplicable(ctx))
            {
                res_item["reason"] = "Not Applicable";
                std::cerr << "Solver inapplicable: " << solver_name << std::endl;
                throw std::runtime_error(
                    "InApplicable solver was sent to fin, check Tuna for errors");
                return false;
            }

            res_item["tunable"] = true;
            // allowing non-tunable solvers to enter here for fdb generation
            if(!s.IsTunable())
                res_item["tunable"] = false;

            // eg when ConvOclDirectFwd has no kernels FindSolution memory faults
            if(s.IsTunable() and kinder["kernel_objects"].empty())
            {
                res_item["reason"] = "No Kernels";
                return false;
            }

            std::cerr << solver_name << " is applicable" << std::endl;
            // Get the binary
            std::cerr << "loading binaries from fin input" << std::endl;
            for(const auto& kernel_obj : kinder["kernel_objects"])
            {
                const auto size          = kernel_obj["uncompressed_size"];
                const auto md5_sum       = kernel_obj["md5_sum"];
                const auto encoded_hsaco = kernel_obj["blob"];
                const auto decoded_hsaco = base64_decode(encoded_hsaco);
                const auto hsaco         = miopen::decompress(decoded_hsaco, size);

                std::string kernel_file_no_ext = kernel_obj["kernel_file"];
                std::string kernel_file        = kernel_file_no_ext + ".o";
                std::string comp_opts          = kernel_obj["comp_options"];
                // LoadProgram doesn't add -mcpu for mlir
                if(!miopen::EndsWith(kernel_file_no_ext, ".mlir"))
                {
                    comp_opts += " -mcpu=" + h.GetDeviceName();
                }

                if(miopen::md5(hsaco) == md5_sum)
                {
                    try
                    {
                        std::cerr << "Make Program: " << kernel_file << "; args: " << comp_opts
                                  << std::endl;
                        auto p = miopen::Program{kernel_file, hsaco};
                        std::cerr << "Add Program: " << kernel_file << "; args: " << comp_opts
                                  << std::endl;
                        h.AddProgram(p, kernel_file, comp_opts);
                    }
                    catch(const std::exception& e)
                    {
                        res_item["reason"] = std::string("Make Program exception: ") + e.what();
                        return false;
                    }

                    // SaveBinary adds ".o" to kernel_file
                    miopen::SaveBinary(hsaco,
                                       h.GetTargetProperties(),
                                       h.GetMaxComputeUnits(),
                                       kernel_file_no_ext,
                                       comp_opts,
                                       false);
                }
                else
                {
                    res_item["reason"] = "Corrupt Binary";
                    std::cerr << "Corrupt Binary Object" << std::endl;
                    throw std::runtime_error("Corrupt binary object");
                    return false;
                }
            }

            miopen::solver::ConvSolution solution;
            solution              = s.FindSolution(ctx, db, {}); // auto tune is not expected here
            res_item["workspace"] = solution.workspace_sz;

            std::cerr << "Checking for workspace: " << solution.workspace_sz << std::endl;
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

            std::cerr << "Preparing invokers" << std::endl;
            try
            {
                float time    = 0.0f;
                ctx.do_search = true;
                ctx.db_update = true;

                // This is required because DataInvokeParams switches tensor order due to
                // direction and it does not have a
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

                    std::cerr << solver_name << " Begin Search FWD" << std::endl;
                    solution = s.FindSolution(ctx, db, invoke_ctx); // forcing search here
                    std::cerr << solver_name << " Finished Search FWD" << std::endl;
                    kern_objs = BuildJsonKernelList(h, solution.construction_params);
                    SolutionHasProgram(h, solution);
                    params = s.GetPerfCfgParams(ctx, db);

                    const auto invoker =
                        h.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
                    invoker(h, invoke_ctx);
                    time = h.GetKernelTime();
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

                    solution = s.FindSolution(ctx, db, invoke_ctx); // forcing search here
                    std::cerr << solver_name << " Finished Search BWD" << std::endl;
                    kern_objs = BuildJsonKernelList(h, solution.construction_params);
                    SolutionHasProgram(h, solution);
                    params = s.GetPerfCfgParams(ctx, db);

                    const auto invoker =
                        h.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
                    invoker(h, invoke_ctx);
                    time = h.GetKernelTime();
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

                    solution = s.FindSolution(ctx, db, invoke_ctx); // forcing search here
                    std::cerr << solver_name << " Finished Search WRW" << std::endl;
                    kern_objs = BuildJsonKernelList(h, solution.construction_params);
                    SolutionHasProgram(h, solution);
                    params = s.GetPerfCfgParams(ctx, db);

                    const auto invoker =
                        h.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
                    invoker(h, invoke_ctx);
                    time = h.GetKernelTime();
                }
                else
                {
                    ss.str("");
                    ss << "Invalid Direction: solver " << solver_name << ", dir "
                       << static_cast<int>(conv_dir);
                    throw std::runtime_error(ss.str());
                }

                res_item["params"]         = params;
                res_item["time"]           = time;
                res_item["layout"]         = ctx.problem.in_layout;
                res_item["data_type"]      = ctx.problem.in_data_type;
                res_item["direction"]      = conv_dir;
                res_item["bias"]           = ctx.problem.bias;
                res_item["reason"]         = "Success";
                res_item["kernel_objects"] = kern_objs;
            }
            catch(const std::exception& e)
            {
                res_item["reason"] = std::string("Invoker exception: ") + e.what();
                return false;
            }

            return true;
        };

        auto res              = process_solver();
        res_item["evaluated"] = res;
        perf_result.push_back(res_item);
    }
    output["miopen_perf_eval_result"] = perf_result;
    return 1;
#else
    throw std::runtime_error("Unsupported feature");
    return 0;
#endif
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

    const auto network_config   = ctx.problem.BuildConfKey();
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
    bool dynamic_only = false;
    if(job.contains("dynamic_only"))
        dynamic_only = job["dynamic_only"];

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
            const std::string solver_name = kinder["solver_name"];
            std::cerr << "Processing solver: " << solver_name << std::endl;
            const auto solver_id    = miopen::solver::Id{solver_name};
            const auto& s           = solver_id.GetSolver();
            const auto algo         = solver_id.GetAlgo(conv_dir);
            res_item["solver_name"] = solver_name;
            res_item["algorithm"]   = algo;

            if(s.IsEmpty())
            {
                res_item["reason"] = "Empty Solver";
                std::cerr << "Skipping invalid solver: " << solver_id.ToString() << std::endl;
                return false;
            }
            if(!s.IsApplicable(ctx))
            {
                res_item["reason"] = "Not Applicable";
                std::cerr << "Solver inapplicable: " << solver_name << std::endl;
                throw std::runtime_error(
                    "InApplicable solver was sent to fin, check Tuna for errors");
                return false;
            }
            if(dynamic_only && !s.IsDynamic())
            {
                res_item["reason"] = "Not Dynamic";
                std::cerr << "Skipping static solver: " << solver_id.ToString() << std::endl;
                return false;
            }

            res_item["params"]  = s.GetPerfCfgParams(ctx, db);
            res_item["tunable"] = false;
            if(s.IsTunable())
                res_item["tunable"] = true;

            std::cerr << solver_name << " is applicable" << std::endl;
            // Get the binary
            std::cerr << "loading binaries from fin input" << std::endl;
            for(const auto& kernel_obj : kinder["kernel_objects"])
            {
                const auto size          = kernel_obj["uncompressed_size"];
                const auto md5_sum       = kernel_obj["md5_sum"];
                const auto encoded_hsaco = kernel_obj["blob"];
                const auto decoded_hsaco = base64_decode(encoded_hsaco);
                const auto hsaco         = miopen::decompress(decoded_hsaco, size);

                std::string kernel_file_no_ext = kernel_obj["kernel_file"];
                std::string kernel_file        = kernel_file_no_ext + ".o";
                std::string comp_opts          = kernel_obj["comp_options"];
                // LoadProgram doesn't add -mcpu for mlir
                if(!miopen::EndsWith(kernel_file_no_ext, ".mlir"))
                {
                    comp_opts += " -mcpu=" + h.GetDeviceName();
                }

                if(miopen::md5(hsaco) == md5_sum)
                {
                    try
                    {
                        std::cerr << "Make Program: " << kernel_file << "; args: " << comp_opts
                                  << std::endl;
                        auto p = miopen::Program{kernel_file, hsaco};
                        std::cerr << "Add Program: " << kernel_file << "; args: " << comp_opts
                                  << std::endl;
                        h.AddProgram(p, kernel_file, comp_opts);
                    }
                    catch(const std::exception& e)
                    {
                        res_item["reason"] = std::string("Make Program exception: ") + e.what();
                        return false;
                    }
                }
                else
                {
                    std::cerr << "Corrupt Binary Object" << std::endl;
                    throw std::runtime_error("Corrupt binary object");
                    return false;
                }
            }

            auto solution         = s.FindSolution(ctx, db, {}); // auto tune is not expected here
            res_item["workspace"] = solution.workspace_sz;
            SolutionHasProgram(h, solution);

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

    const auto network_config   = ctx.problem.BuildConfKey();
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
    assert(arch == job["arch"]);
    const size_t num_cu = h.GetMaxComputeUnits();
    assert(num_cu == job["num_cu"]);
    for(const auto& solver_id :
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
    {
        json res_item;
        auto process_solver = [&]() -> bool {
            res_item["solver_name"] = solver_id.ToString();
            const auto& s           = solver_id.GetSolver();
            const auto algo         = solver_id.GetAlgo(conv_dir);
            res_item["algorithm"]   = algo;
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
                std::string comp_opts = k.comp_options;
                if(!miopen::EndsWith(k.kernel_file, ".mlir"))
                {
                    comp_opts = k.comp_options + " -mcpu=" + arch;
                }

                const auto hsaco =
                    miopen::LoadBinary(tgt_props, num_cu, k.kernel_file, comp_opts, false);
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
    // cppcheck-suppress unreadVariable
    auto ctx = miopen::ConvolutionContext{problem};
    // cppcheck-suppress unreadVariable
    auto handle = miopen::Handle{};
#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif

    ctx.SetStream(&handle);
    ctx.DetectRocm();
    ctx.SetupFloats();
    const auto network_config = ctx.problem.BuildConfKey();
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
int ConvFin<Tgpu, Tref>::TestPerfDbEntries(
    const std::string config_id,
    const miopen::ConvolutionContext& ctx,
    const std::map<std::string, std::unordered_map<std::string, std::string>>& perf_ids,
    const std::unordered_map<std::string, miopen::DbRecord>& records,
    std::vector<std::map<std::string, std::string>>& err_list,
    std::vector<std::string>& pdb_id)
{
    bool ret = true;

    // iterate over pdb entries
    for(auto pdb_it = perf_ids.begin(); pdb_it != perf_ids.end(); pdb_it++)
    {
        auto perf_id   = pdb_it->first;
        auto solver_nm = pdb_it->second.find("solver")->second;
        auto params    = pdb_it->second.find("params")->second;
        auto record    = records.find(perf_id)->second;

        auto slv_id = miopen::solver::Id(solver_nm);
        auto solver = slv_id.GetSolver();
        std::stringstream stat_str;
        stat_str << "config_id: " << config_id << ", solver_nm " << solver_nm
                 << ", key: " << ctx.problem;

        // check if valid pdb parameters
        std::map<std::string, std::string> err;
        bool success = false;
        try
        {
            success = solver.TestSysDbRecord(ctx, record);
        }
        catch(const std::exception& e)
        {
            err["reason"] = e.what();
            std::cerr << "Error in db test: " << e.what() << std::endl;
        }
        if(!success)
        {
            err["perfdb_id"] = perf_id;
            err["config"]    = config_id;
            err["solver"]    = solver_nm;
            err["params"]    = params;
            err_list.push_back(err);
            ret = false;
            pdb_id.push_back(perf_id);

            std::cerr << stat_str.str() << ", failed" << std::endl;
        }
        else
            std::cerr << stat_str.str() << ", passed" << std::endl;
    }

    return ret;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::TestPerfDbValid()
{

    bool ret            = true;
    namespace fs        = boost::filesystem;
    bool spec_arch      = (job["arch"].size() > 0 and job["num_cu"].size() > 0);
    std::string db_path = miopen::GetSystemDbPath();

    if(job.contains("db_path"))
        db_path = job["db_path"];
    std::cout << db_path << std::endl;

    std::vector<fs::path> contents;
    std::copy(
        fs::directory_iterator(db_path), fs::directory_iterator(), std::back_inserter(contents));
    for(auto const& db_file : contents)
    {
        std::string pathstr = db_file.native();
        std::string filestr = db_file.filename().native();
        std::string db_arch;
        size_t db_num_cu = 0;

        // test if a db file
        if(filestr.compare(filestr.size() - 3, 3, ".db") != 0)
            continue;

        std::cerr << pathstr << std::endl;
        // get arch and num_cu from filename
        size_t delim;
        if((delim = filestr.find('_')) != std::string::npos)
        {
            db_arch   = filestr.substr(0, delim);
            db_num_cu = static_cast<int>(std::stoi(filestr.substr(delim + 1, filestr.size() - 3)));
        }
        else
        {
            // num_cu should be last 2 hex numbers before .db
            delim   = filestr.size() - 5;
            db_arch = filestr.substr(0, delim);
            db_num_cu =
                static_cast<int>(std::strtol(filestr.substr(delim, 2).c_str(), nullptr, 16));
        }
        std::cerr << db_arch << " " << db_num_cu << std::endl;
        BaseFin::VerifyDevProps(db_arch, db_num_cu);

        if(spec_arch)
        {
            std::stringstream str_cu;
            str_cu << job["num_cu"];

            if(db_arch.compare(job["arch"]) != 0)
                continue;
            if(db_num_cu != job["num_cu"])
                continue;
        }

        std::cerr << "processing: " << pathstr << std::endl;

        // setting system to false allows writing the db
        auto sql = miopen::SQLite{pathstr, false};

        // set handle to type of db under test
        auto handle = miopen::Handle{};
        BaseFin::InitNoGpuHandle(handle, db_arch, db_num_cu);

        // cfg -> pdb_id -> values_dict
        std::map<std::string, std::map<std::string, std::unordered_map<std::string, std::string>>>
            perfdb_entries;
        // pdb_id -> record
        std::unordered_map<std::string, miopen::DbRecord> records;
        std::vector<std::map<std::string, std::string>> err_list;
        std::vector<std::string> pdb_id;
        auto select_query = "SELECT config, solver, params, id FROM perf_db;";
        auto stmt         = miopen::SQLite::Statement{sql, select_query};
        while(true)
        {
            auto rc = stmt.Step(sql);
            if(rc == SQLITE_ROW)
            {
                const auto config_id = stmt.ColumnText(0);
                const auto solver_nm = stmt.ColumnText(1);
                const auto params    = stmt.ColumnText(2);
                const auto perf_id   = stmt.ColumnText(3);

                auto slv_id = miopen::solver::Id(solver_nm);
                if(!slv_id.IsValid())
                {
                    std::map<std::string, std::string> err;
                    err["perfdb_id"] = perf_id;
                    err["config"]    = config_id;
                    err["solver"]    = solver_nm;
                    err["params"]    = params;
                    err_list.push_back(err);
                    ret = false;
                    pdb_id.push_back(perf_id);
                    continue;
                }

                records[perf_id].SetValues(solver_nm, ParamString(params));
                perfdb_entries[config_id][perf_id]["solver"] = solver_nm;
                perfdb_entries[config_id][perf_id]["params"] = params;
            }
            else if(rc == SQLITE_DONE)
                break;
            else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
                MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
        }

        // iterate through each config
        for(auto cfg_it = perfdb_entries.begin(); cfg_it != perfdb_entries.end(); cfg_it++)
        {
            auto config_id = cfg_it->first;
            miopen::ConvolutionContext ctx;

            std::cerr << "building context" << std::endl;
            ctx = BuildContext(sql, config_id);
            ctx.SetStream(&handle);
            ctx.DetectRocm();
            ctx.SetupFloats();

            std::cerr << "test pdb" << std::endl;
            bool success = true;
            success = TestPerfDbEntries(config_id, ctx, cfg_it->second, records, err_list, pdb_id);
            if(not success)
                ret = false;
        }
        output[filestr]["errors"] = err_list;

        if(job.contains("cleanup") && job["cleanup"])
        {
            std::ostringstream id_str, del_query;
            for(auto it = pdb_id.begin(); it != pdb_id.end(); it++)
            {
                if(it != pdb_id.begin())
                    id_str << ",";
                id_str << *it;
            }
            del_query << "DELETE from perf_db where id in (" << id_str.str() << ");";
            stmt    = miopen::SQLite::Statement{sql, del_query.str()};
            auto rc = stmt.Step(sql);
            std::cerr << "delete status: " << rc << std::endl;

            output[filestr]["sql_del"]    = del_query.str();
            output[filestr]["del_status"] = rc;
        }
    }

    if(ret)
        output["clear"] = "true";

    return ret;
}

template <typename Tgpu, typename Tref>
int ConvFin<Tgpu, Tref>::SearchPreCompiledKernels()
{
    json find_result;
    // cppcheck-suppress unreadVariable
    auto handle = miopen::Handle{};

#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "for SearchPreCompiledKernels");
#endif

    // extract numcu and arch details from handle
    const auto& tgt_props  = handle.GetTargetProperties();
    const size_t num_cu    = handle.GetMaxComputeUnits();
    const std::string arch = tgt_props.Name();

    namespace fs = boost::filesystem;

    // to fetch the kdb folder location
    // ex:  /opt/rocm/miopen/share/miopen/db
    auto pathstr = miopen::GetCachePath(true);

    // append the json input arch and numcu values to file
    boost::filesystem::path sys_path =
        pathstr / (miopen::Handle::GetDbBasename(tgt_props, num_cu) + ".kdb");
    std::cout << "System KernDB path = " << sys_path << std::endl;

    // checks the file present in shared folder
    if(boost::filesystem::exists(sys_path))
    {
        std::cout << "KernDB file Present  =  " << sys_path << std::endl;

        json file_chk;
        file_chk["kdb_file"]       = sys_path.string().c_str();
        file_chk["kdb_file_found"] = true;
        find_result.push_back(file_chk);

// sets the values specific to Tensor from the json i/p file.
#if MIOPEN_MODE_NOGPU
        GetandSetData();
#endif

        // following methods are used to set the
        // problem description, directionm context etc.
        const auto conv_dir = GetDirection();
        const miopen::ProblemDescription problem(
            inputTensor.desc, weightTensor.desc, outputTensor.desc, convDesc, conv_dir);
        auto ctx = miopen::ConvolutionContext{problem};

        ctx.SetStream(&handle);
        ctx.DetectRocm();
        ctx.SetupFloats();

        // const auto network_config = ctx.problem.BuildConfKey();
        std::ostringstream ss;
        problem.Serialize(ss);

        // create handle, which holds information about kernel/solver/solution etc
        auto db_obj = GetDb(ctx);

        // get the solver ids, this is populated for default ids.
        for(const auto& solver_id :
            miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
        {

            json res_item;
            bool retvalue;
            // to extract solver id ,context,solution
            auto process_solver = [&]() -> bool {
                res_item["solver_id"] = solver_id.ToString();
                const auto s          = solver_id.GetSolver();
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

                // we need to do this to avoid perf db search/update.
                // scenario is get the solver id specific solution.
                ctx.do_search             = false;
                ctx.disable_perfdb_access = false;

                // find solution for solver id.
                const auto default_solution = s.FindSolution(ctx, db_obj, {});

                if(default_solution.Succeeded() && default_solution.construction_params.empty())
                {
                    std::cout << "Internal error in solver: " << solver_id.ToString() << std::endl;
                    res_item["reason"] = "Solver Id Error";
                    return false;
                }
                json cdobj_list = json::array();
                // check the presence of precompiled kernel code object present
                // in memory ?
                for(const auto& k : default_solution.construction_params)
                {
                    json cdobj_result;
                    auto comp_opts   = k.comp_options;
                    const auto hsaco = miopen::LoadBinary(
                        tgt_props, num_cu, k.kernel_file, comp_opts + " -mcpu=" + arch, false);
                    if(hsaco.empty())
                    {
                        std::cout << "!!!FAILURE !!! - Kernel Db is not present" << std::endl;
                        cdobj_result["kernel_db_access"] = false;
                        retvalue                         = false;
                    }
                    else
                    {
                        std::cout << "!!!Sucess!!! - Kernel Db is present" << std::endl;
                        cdobj_result["kernel_db_access"] = true;

                        // create the Program object
                        auto proObj = miopen::HIPOCProgram{comp_opts + " -mcpu=" + arch, hsaco};

                        // check the code object presence?
                        const auto c_hsaco = proObj.IsCodeObjectInMemory();
                        if(c_hsaco)
                        {
                            std::cout << "!!!Success!!!Kernel Code Objet present in memory"
                                      << std::endl;
                            cdobj_result["code_object_in_memory"] = true;
                            retvalue                              = true;
                        }
                        else
                        {
                            std::cout << "!!! FAILURE!!!Code Objet is not in memory" << std::endl;
                            cdobj_result["code_object_in_memory"] = false;
                            retvalue                              = false;
                        }
                    } // else
                    cdobj_list.push_back(cdobj_result);
                } // for
                res_item["kerenel_objects_list"] = cdobj_list;
                return retvalue;
            };
            auto result                     = process_solver();
            res_item["code_obj_chk_result"] = result;
            find_result.push_back(res_item);
        } // for-sloverlist
    }     // if( file exisits?)
    else
    {
        std::cout << " Kernel Database= " << sys_path << " Does not exist in the system path"
                  << std::endl;
        json err_result;
        err_result["kdb_file"]       = sys_path.string().c_str();
        err_result["kdb_file_found"] = false;
        find_result.push_back(err_result);
    }
    output["chk_pre_compiled_kernels"] = find_result;
    return true;
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
        return TestApplicability();
    if(step_name == "perf_db_test")
        return TestPerfDbValid();
    if(step_name == "miopen_find")
        return MIOpenFind();
    if(step_name == "miopen_find_compile")
        return MIOpenFindCompile();
    if(step_name == "miopen_find_eval")
        return MIOpenFindEval();
    if(step_name == "chk_pre_compiled_kernels")
    {
        return SearchPreCompiledKernels();
    }
    if(step_name == "miopen_perf_compile")
        return MIOpenPerfCompile();
    if(step_name == "miopen_perf_eval")
        return MIOpenPerfEval();
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
        // cppcheck-suppress unreadVariable
        in_spatial_lens[0] = command["in_h"];
        // cppcheck-suppress unreadVariable
        in_spatial_lens[1] = command["in_w"];
    }
    else if(spatial_dim == 3)
    {
        // cppcheck-suppress unreadVariable
        in_spatial_lens[0] = command["in_d"];
        // cppcheck-suppress unreadVariable
        in_spatial_lens[1] = command["in_h"];
        // cppcheck-suppress unreadVariable
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
        // cppcheck-suppress unreadVariable
        wei_spatial_lens[0] = command["fil_h"];
        // cppcheck-suppress unreadVariable
        wei_spatial_lens[1] = command["fil_w"];
    }
    else if(spatial_dim == 3)
    {
        // cppcheck-suppress unreadVariable
        wei_spatial_lens[0] = command["fil_d"];
        // cppcheck-suppress unreadVariable
        wei_spatial_lens[1] = command["fil_h"];
        // cppcheck-suppress unreadVariable
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
miopen::ConvolutionContext ConvFin<Tgpu, Tref>::GetCmdConvContext(json _command)
{
    command         = _command;
    command["bias"] = 0;
    // timing is always enabled
    is_fwd = (command["direction"].get<std::string>().compare("F") == 0);
    is_bwd = (command["direction"].get<std::string>().compare("B") == 0);
    is_wrw = (command["direction"].get<std::string>().compare("W") == 0);
    SetConvDescriptor();

    // set tensors with command data
    GetandSetData();

    // initialize context
    const auto conv_dir = GetDirection();
    const miopen::ProblemDescription problem(
        inputTensor.desc, weightTensor.desc, outputTensor.desc, convDesc, conv_dir);
    auto ctx = miopen::ConvolutionContext{problem};

    return ctx;
}

template <typename Tgpu, typename Tref>
miopen::ConvolutionContext ConvFin<Tgpu, Tref>::BuildContext(miopen::SQLite& sql,
                                                             std::string config_id)
{
    std::ostringstream ss;
    ss << "SELECT in_d, in_h, in_w, fil_d, fil_h, fil_w, pad_d, pad_h, pad_w, "
          "conv_stride_d, conv_stride_h, conv_stride_w, dilation_d, dilation_h, "
          "dilation_w, spatial_dim, layout, data_type, direction, "
          "out_channels, in_channels, batchsize, group_count, bias "
          "FROM config WHERE id=";
    ss << config_id << ";";
    auto cfg_query = ss.str();
    auto stmt      = miopen::SQLite::Statement{sql, cfg_query};
    stmt.Step(sql);

    // initialize command with query results
    command["in_d"]          = stmt.ColumnInt64(0);
    command["in_h"]          = stmt.ColumnInt64(1);
    command["in_w"]          = stmt.ColumnInt64(2);
    command["fil_d"]         = stmt.ColumnInt64(3);
    command["fil_h"]         = stmt.ColumnInt64(4);
    command["fil_w"]         = stmt.ColumnInt64(5);
    command["pad_d"]         = stmt.ColumnInt64(6);
    command["pad_h"]         = stmt.ColumnInt64(7);
    command["pad_w"]         = stmt.ColumnInt64(8);
    command["conv_stride_d"] = stmt.ColumnInt64(9);
    command["conv_stride_h"] = stmt.ColumnInt64(10);
    command["conv_stride_w"] = stmt.ColumnInt64(11);
    command["dilation_d"]    = stmt.ColumnInt64(12);
    command["dilation_h"]    = stmt.ColumnInt64(13);
    command["dilation_w"]    = stmt.ColumnInt64(14);
    command["spatial_dim"]   = stmt.ColumnInt64(15);
    command["direction"]     = stmt.ColumnText(18);
    command["out_channels"]  = stmt.ColumnInt64(19);
    command["in_channels"]   = stmt.ColumnInt64(20);
    command["batchsize"]     = stmt.ColumnInt64(21);
    command["group_count"]   = stmt.ColumnInt64(22);
    command["bias"]          = stmt.ColumnInt64(23);
    command["conv_mode"]     = "conv";

    command["in_layout"]  = stmt.ColumnText(16);
    command["wei_layout"] = stmt.ColumnText(16);
    command["out_layout"] = stmt.ColumnText(16);
    std::string data_type = stmt.ColumnText(17);

    miopen::ConvolutionContext ctx;
    if(data_type == "FP32")
    {
        ctx = fin::ConvFin<float, float>().GetCmdConvContext(command);
    }
    else if(data_type == "FP16")
    {
        ctx = fin::ConvFin<float16, float>().GetCmdConvContext(command);
    }
    else if(data_type == "BF16")
    {
        ctx = fin::ConvFin<bfloat16, float>().GetCmdConvContext(command);
    }
    else if(data_type == "INT8")
    {
        ctx = fin::ConvFin<int8_t, float>().GetCmdConvContext(command);
    }
    else
    {
        std::cerr << "other type: " << data_type << std::endl;
    }

    return ctx;
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

    using namespace miopen;
    const auto dir     = is_wrw   ? conv::Direction::BackwardWeights
                         : is_bwd ? conv::Direction::BackwardData
                                  : conv::Direction::Forward;
    const auto ctx     = ExecutionContext{&GetHandle()}.DetectRocm();
    const auto problem = conv::ProblemDescription{
        inputTensor.desc, weightTensor.desc, outputTensor.desc, convDesc, dir};

    auto& result = is_wrw ? ws_sizeof_find_wrw : is_bwd ? ws_sizeof_find_bwd : ws_sizeof_find_fwd;

    result = convDesc.GetWorkSpaceSize(ctx, problem);

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
