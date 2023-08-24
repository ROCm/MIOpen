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

#include <miopen/solution.hpp>

#include <miopen/any_solver.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/kernel.hpp>

#include <nlohmann/json.hpp>

#include <boost/hof/match.hpp>

namespace miopen::debug {
// Todo: This should be updated when a separate driver command is implemented
void LogCmdConvolution(const miopen::TensorDescriptor& x,
                       const miopen::TensorDescriptor& w,
                       const miopen::ConvolutionDescriptor& conv,
                       const miopen::TensorDescriptor& y,
                       miopenProblemDirection_t dir,
                       std::optional<uint64_t> solver_id);
} // namespace miopen::debug

namespace miopen {

void Solution::Run(Handle& handle,
                   const std::unordered_map<miopenTensorArgumentId_t, RunInput>& inputs,
                   Data_t workspace,
                   std::size_t workspace_size)
{
    if(workspace_size < workspace_required)
        MIOPEN_THROW(miopenStatusBadParm,
                     GetSolver().ToString() + " requires at least " +
                         std::to_string(workspace_required) + " workspace, while " +
                         std::to_string(workspace_size) + " was provided");

    const auto run = boost::hof::match([&](const ConvolutionDescriptor& op_desc) {
        RunImpl(handle, inputs, workspace, workspace_size, op_desc);
    });

    std::visit(run, problem.GetOperatorDescriptor());
}

void Solution::LogDriverCommand() const
{
    const auto log_function = boost::hof::match(
        [&](const ConvolutionDescriptor& op_desc) { return LogDriverCommand(op_desc); });

    std::visit(log_function, problem.GetOperatorDescriptor());
}

void Solution::LogDriverCommand(const ConvolutionDescriptor& conv_desc) const
{
    const auto& x_desc =
        problem.GetTensorDescriptorChecked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto& w_desc =
        problem.GetTensorDescriptorChecked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    const auto& y_desc =
        problem.GetTensorDescriptorChecked(miopenTensorConvolutionY, "miopenTensorConvolutionY");
    miopen::debug::LogCmdConvolution(
        x_desc, w_desc, conv_desc, y_desc, problem.GetDirection(), solver.Value());
}

void Solution::RunImpl(Handle& handle,
                       const std::unordered_map<miopenTensorArgumentId_t, RunInput>& inputs,
                       Data_t workspace,
                       std::size_t workspace_size,
                       const ConvolutionDescriptor& conv_desc)
{
    const auto get_input_checked = [&](auto name, const std::string& name_str) {
        const auto& found = inputs.find(name);
        if(found == inputs.end())
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Problem is missing " + name_str + " tensor descriptor.");
        auto ret = found->second;
        if(!ret.descriptor.has_value())
            ret.descriptor = GetProblem().GetTensorDescriptorChecked(name, name_str);
        return ret;
    };

    auto x       = get_input_checked(miopenTensorConvolutionX, "miopenTensorConvolutionX");
    const auto w = get_input_checked(miopenTensorConvolutionW, "miopenTensorConvolutionW");
    auto y       = get_input_checked(miopenTensorConvolutionY, "miopenTensorConvolutionY");

    const auto problem_ =
        conv_desc.mode == miopenTranspose ? Transpose(GetProblem(), &x, w, &y) : GetProblem();

    if(problem_.GetDirection() == miopenProblemDirectionBackward &&
       y.descriptor->GetLengths()[1] != w.descriptor->GetLengths()[0])
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(miopen::CheckNumericsEnabled())
    {
        if(problem_.GetDirection() != miopenProblemDirectionBackward)
            miopen::checkNumericsInput(handle, *x.descriptor, x.buffer);
        if(problem_.GetDirection() != miopenProblemDirectionBackwardWeights)
            miopen::checkNumericsInput(handle, *w.descriptor, w.buffer);
        if(problem_.GetDirection() != miopenProblemDirectionForward)
            miopen::checkNumericsInput(handle, *y.descriptor, y.buffer);
    }

    Problem::ValidateGroupCount(*x.descriptor, *w.descriptor, conv_desc);

    const auto invoke_ctx =
        MakeInvokeParams(problem_, conv_desc, x, w, y, workspace, workspace_size);

    const auto checkNumericsOutput_ = [&]() {
        if(miopen::CheckNumericsEnabled())
        {
            if(problem_.GetDirection() == miopenProblemDirectionBackward)
                miopen::checkNumericsOutput(handle, *x.descriptor, x.buffer);
            if(problem_.GetDirection() == miopenProblemDirectionBackwardWeights)
                miopen::checkNumericsOutput(handle, *w.descriptor, w.buffer);
            if(problem_.GetDirection() == miopenProblemDirectionForward)
                miopen::checkNumericsOutput(handle, *y.descriptor, y.buffer);
        }
    };

    if(invoker)
    {
        (*invoker)(handle, invoke_ctx);
        checkNumericsOutput_();
        return;
    }

    const auto conv_problem = problem_.AsConvolution();

    if(!kernels.empty())
    {
        const auto legacy_problem = ProblemDescription{conv_problem};
        auto conv_ctx             = ConvolutionContext{{&handle}};
        conv_problem.SetupFloats(conv_ctx);
        const auto invoker_factory = GetSolver().GetSolver().GetInvokeFactory(
            conv_ctx, legacy_problem, perf_cfg.value_or(""));

        auto kernel_handles = std::vector<Kernel>{};

        std::transform(
            std::begin(kernels),
            std::end(kernels),
            std::back_inserter(kernel_handles),
            [](const KernelInfo& ki) {
                return Kernel{ki.program, ki.kernel_name, ki.local_work_dims, ki.global_work_dims};
            });

        invoker = invoker_factory(kernel_handles);
        (*invoker)(handle, invoke_ctx);
        checkNumericsOutput_();
        return;
    }

    const auto net_cfg       = conv_problem.BuildConfKey();
    const auto found_invoker = handle.GetInvoker(net_cfg, GetSolver());

    if(found_invoker)
    {
        invoker = *found_invoker;
        (*found_invoker)(handle, invoke_ctx);
        checkNumericsOutput_();
        return;
    }

    const auto legacy_problem = ProblemDescription{conv_problem};
    auto conv_ctx             = ConvolutionContext{{&handle}};
    conv_problem.SetupFloats(conv_ctx);

    decltype(auto) db        = GetDb(conv_ctx);
    const auto conv_solution = GetSolver().GetSolver().FindSolution(
        conv_ctx, legacy_problem, db, invoke_ctx, perf_cfg.value_or(""));

    invoker =
        handle.PrepareInvoker(*conv_solution.invoker_factory, conv_solution.construction_params);
    handle.RegisterInvoker(*invoker, net_cfg, GetSolver().ToString());
    (*invoker)(handle, invoke_ctx);
    checkNumericsOutput_();
}

AnyInvokeParams Solution::MakeInvokeParams(const Problem& problem_,
                                           const ConvolutionDescriptor& conv_desc,
                                           const RunInput& x,
                                           const RunInput& w,
                                           const RunInput& y,
                                           Data_t workspace,
                                           size_t workspace_size)
{
    switch(problem_.GetDirection())
    {
    case miopenProblemDirectionForward:
        return conv::DataInvokeParams(
            {*x.descriptor, x.buffer, *w.descriptor, w.buffer, *y.descriptor, y.buffer},
            workspace,
            workspace_size,
            conv_desc.attribute.gfx90aFp16alt.GetFwd());
    case miopenProblemDirectionBackward:
        return conv::DataInvokeParams(
            {*y.descriptor, y.buffer, *w.descriptor, w.buffer, *x.descriptor, x.buffer},
            workspace,
            workspace_size,
            conv_desc.attribute.gfx90aFp16alt.GetBwd());
    case miopenProblemDirectionBackwardWeights:
        return conv::WrWInvokeParams{
            {*y.descriptor, y.buffer, *x.descriptor, x.buffer, *w.descriptor, w.buffer},
            workspace,
            workspace_size,
            conv_desc.attribute.gfx90aFp16alt.GetWrW()};
    default: MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

Problem Solution::Transpose(const Problem& problem, RunInput* x, const RunInput& w, RunInput* y)
{
    auto transposed = problem.MakeTransposed();

    std::swap(*x, *y);

    if(x->descriptor)
        transposed.RegisterTensorDescriptor(miopenTensorConvolutionX, *x->descriptor);
    if(w.descriptor)
        transposed.RegisterTensorDescriptor(miopenTensorConvolutionW, *w.descriptor);
    if(y->descriptor)
        transposed.RegisterTensorDescriptor(miopenTensorConvolutionY, *y->descriptor);

    return transposed;
}

namespace fields {
namespace header {
inline constexpr const char* Validation = "validation";
inline constexpr const char* Version    = "version";
} // namespace header
inline constexpr const char* Header    = "header";
inline constexpr const char* Time      = "time";
inline constexpr const char* Workspace = "workspace";
inline constexpr const char* Solver    = "solver";
inline constexpr const char* Problem   = "problem";
inline constexpr const char* PerfCfg   = "perf_cfg";
inline constexpr const char* Binaries  = "binaries";
inline constexpr const char* Kernels   = "kernels";
namespace kernels {
inline constexpr const char* Name           = "name";
inline constexpr const char* File           = "file";
inline constexpr const char* Program        = "program";
inline constexpr const char* LocalWorkDims  = "local_work_dims";
inline constexpr const char* GlobalWorkDims = "global_work_dims";
} // namespace kernels
} // namespace fields

void to_json(nlohmann::json& json, const Solution::SerializationMetadata& metadata)
{
    json = nlohmann::json{
        {fields::header::Validation, metadata.validation_number},
        {fields::header::Version, metadata.version},
    };
}
void from_json(const nlohmann::json& json, Solution::SerializationMetadata& metadata)
{
    json.at(fields::header::Validation).get_to(metadata.validation_number);
    json.at(fields::header::Version).get_to(metadata.version);
}

struct SerializedSolutionKernelInfo
{
    int program;
    std::vector<size_t> local_work_dims;
    std::vector<size_t> global_work_dims;
    std::string kernel_name;
    std::string program_name;

    friend void to_json(nlohmann::json& json, const SerializedSolutionKernelInfo& kernel_info)
    {
        json = nlohmann::json{
            {fields::kernels::Program, kernel_info.program},
            {fields::kernels::Name, kernel_info.kernel_name},
            {fields::kernels::File, kernel_info.program_name},
            {fields::kernels::LocalWorkDims, kernel_info.local_work_dims},
            {fields::kernels::GlobalWorkDims, kernel_info.global_work_dims},
        };

        MIOPEN_LOG_I2("Serialized solution kernel info <" << kernel_info.program_name << ":"
                                                          << kernel_info.kernel_name << ", binary "
                                                          << kernel_info.program << ">");
    }

    friend void from_json(const nlohmann::json& json, SerializedSolutionKernelInfo& kernel_info)
    {
        json.at(fields::kernels::Program).get_to(kernel_info.program);
        json.at(fields::kernels::Name).get_to(kernel_info.kernel_name);
        json.at(fields::kernels::File).get_to(kernel_info.program_name);
        json.at(fields::kernels::LocalWorkDims).get_to(kernel_info.local_work_dims);
        json.at(fields::kernels::GlobalWorkDims).get_to(kernel_info.global_work_dims);

        MIOPEN_LOG_I2("Deserialized solution kernel info <"
                      << kernel_info.program_name << ":" << kernel_info.kernel_name << ", binary "
                      << kernel_info.program << ">");
    }
};

void to_json(nlohmann::json& json, const Solution& solution)
{
    json = nlohmann::json{
        {fields::Header, Solution::SerializationMetadata::Current()},
        {fields::Time, solution.time},
        {fields::Workspace, solution.workspace_required},
        {fields::Solver, solution.solver.ToString()},
        {fields::Problem, solution.problem},
    };

    if(solution.perf_cfg.has_value())
        json[fields::PerfCfg] = *solution.perf_cfg;

    if(solution.kernels.empty())
    {
        MIOPEN_LOG_I2("Solution lacks kernels information. This would slowdown the first "
                      "miopenRunSolution call after miopenLoadSolution.");
        return;
    }

    {
        const auto& first_program = solution.kernels.front().program;
        if(!first_program.IsCodeObjectInMemory() && !first_program.IsCodeObjectInFile())
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Subsequent serialization of a deserialized solution is not supported.");
    }

    auto programs         = std::vector<Program>{};
    auto prepared_kernels = std::vector<SerializedSolutionKernelInfo>{};

    std::transform(solution.kernels.begin(),
                   solution.kernels.end(),
                   std::back_inserter(programs),
                   [](const Solution::KernelInfo& sol) { return sol.program; });

    constexpr auto sorter = [](auto&& l, auto&& r) { return l.impl.get() < r.impl.get(); };
    std::sort(programs.begin(), programs.end(), sorter);
    programs.erase(std::unique(programs.begin(), programs.end()), programs.end());

    for(auto i = 0; i < solution.kernels.size(); ++i)
    {
        const auto& kernel               = solution.kernels[i];
        const auto program_it            = std::find(programs.begin(), programs.end(), programs[i]);
        auto prepared_kernel             = SerializedSolutionKernelInfo{};
        prepared_kernel.program          = std::distance(programs.begin(), program_it);
        prepared_kernel.kernel_name      = kernel.kernel_name;
        prepared_kernel.program_name     = kernel.program_name;
        prepared_kernel.global_work_dims = kernel.global_work_dims;
        prepared_kernel.local_work_dims  = kernel.local_work_dims;
        prepared_kernels.emplace_back(std::move(prepared_kernel));
    }

    json[fields::Kernels] = prepared_kernels;
    auto programs_json    = nlohmann::json{};

    for(const auto& program : programs)
    {
        auto binary = nlohmann::json::binary_t{};

        if(program.IsCodeObjectInMemory())
        {
            // With disabled cache programs after build would be attached as a char vector. Same for
            // the sqlite cache.

            const auto& chars = program.GetCodeObjectBlobAsVector();
            binary.resize(chars.size());
            std::memcpy(binary.data(), chars.data(), chars.size());

            MIOPEN_LOG_I2("Serialized binary to solution blob, " << chars.size() << " bytes");
        }
        else if(program.IsCodeObjectInFile())
        {
            // Programs that have been loaded from file cache are internally interpreted
            // as read from file with a correct path.

            using Iterator      = std::istream_iterator<uint8_t>;
            constexpr auto mode = std::ios::binary | std::ios::ate;
            const auto path     = program.GetCodeObjectPathname();
            auto file           = std::ifstream(path, mode);
            const auto filesize = file.tellg();

            file.unsetf(std::ios::skipws);
            file.seekg(0, std::ios::beg);
            binary.reserve(filesize);
            binary.insert(binary.begin(), Iterator{file}, Iterator{});

            MIOPEN_LOG_I2("Serialized binary to solution blob, " << filesize << " bytes");
        }
        else
        {
            MIOPEN_THROW(miopenStatusInternalError);
        }

        programs_json.emplace_back(std::move(binary));
    }

    json[fields::Binaries] = std::move(programs_json);
}

void from_json(const nlohmann::json& json, Solution& solution)
{
    {
        const auto header = json.at(fields::Header).get<Solution::SerializationMetadata>();
        constexpr const auto check_header = Solution::SerializationMetadata::Current();

        if(header.validation_number != check_header.validation_number)
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Invalid buffer has been passed to the solution deserialization.");
        if(header.version != check_header.version)
            MIOPEN_THROW(
                miopenStatusVersionMismatch,
                "Data from wrong version has been passed to the solution deserialization.");
    }

    json.at(fields::Time).get_to(solution.time);
    json.at(fields::Workspace).get_to(solution.workspace_required);
    solution.solver = json.at(fields::Solver).get<std::string>();
    json.at(fields::Problem).get_to(solution.problem);

    const auto perf_cfg_json = json.find(fields::PerfCfg);
    solution.perf_cfg        = perf_cfg_json != json.end()
                                   ? std::optional{perf_cfg_json->get<std::string>()}
                                   : std::nullopt;

    auto programs = std::vector<HIPOCProgram>{};

    if(const auto binaries_json = json.find(fields::Binaries); binaries_json != json.end())
    {
        for(const auto& bin : *binaries_json)
        {
            const auto& binary = bin.get_ref<const nlohmann::json::binary_t&>();
            MIOPEN_LOG_I2("Derializing binary from solution blob, " << binary.size() << " bytes");
            programs.emplace_back(HIPOCProgram{"", binary});
        }
    }

    auto& kernel_infos = json.at(fields::Kernels).get<std::vector<SerializedSolutionKernelInfo>>();

    solution.kernels.clear();
    solution.kernels.reserve(kernel_infos.size());

    for(auto&& serialized_kernel_info : kernel_infos)
    {
        auto kernel_info             = Solution::KernelInfo{};
        kernel_info.program          = programs[serialized_kernel_info.program];
        kernel_info.local_work_dims  = std::move(serialized_kernel_info.local_work_dims);
        kernel_info.global_work_dims = std::move(serialized_kernel_info.global_work_dims);
        kernel_info.kernel_name      = std::move(serialized_kernel_info.kernel_name);
        kernel_info.program_name     = std::move(serialized_kernel_info.program_name);
        solution.kernels.emplace_back(std::move(kernel_info));
    }
}
} // namespace miopen
