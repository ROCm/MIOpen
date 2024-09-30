/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#pragma once

#include <miopen/datatype.hpp>
#include <miopen/subbuffers.hpp>
#include <miopen/tensor_layout.hpp>

namespace miopen {
namespace solver {

template <class Element = std::size_t>
inline static std::array<Element, 5> GetNCDHW(const std::vector<std::size_t>& values)
{
    const auto cast = [](auto v) { return static_cast<Element>(v); };
    std::size_t n = 1, c = 1, d = 1, h = 1, w = 1;

    switch(values.size())
    {
    case 5: std::tie(n, c, d, h, w) = tien<5>(values); break;
    case 4: std::tie(n, c, h, w) = tien<4>(values); break;
    default: MIOPEN_THROW(miopenStatusBadParm);
    }

    return {cast(n), cast(c), cast(d), cast(h), cast(w)};
}

struct TransposeProblem
{
    TensorDescriptor input;
    const char* layout;
};

using OldStyleTransposeProblem = std::tuple<const ExecutionContext*, const TransposeProblem*>;

struct TransposeInvokeParams : InvokeParams
{
    ConstData_t in;
    Data_t out;
    TensorDescriptor in_desc;
    TensorDescriptor out_desc;

    TransposeInvokeParams(ConstData_t in_,
                          Data_t out_,
                          TensorDescriptor in_desc_,
                          TensorDescriptor out_desc_)
        : in(in_), out(out_), in_desc(in_desc_), out_desc(out_desc_)
    {
    }

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

struct TransposePseudoSolver
{
    virtual ~TransposePseudoSolver()                                        = default;
    virtual std::string GetTranspose() const                                = 0;
    virtual ConvSolution GetSolution(const ExecutionContext& ctx,
                                     const TransposeProblem& problem) const = 0;

protected:
    TransposePseudoSolver()                             = default;
    TransposePseudoSolver(const TransposePseudoSolver&) = default;
};

template <class Derived, class Interface>
struct AnyImplementation
{
    AnyImplementation() : buffer(), copy(nullptr), p(nullptr) {}

    template <class Implementation>
    AnyImplementation(const Implementation& impl)
    {
        static_assert(sizeof(Implementation) == sizeof(Interface),
                      "Implementation must be stateless");
        static_assert(std::is_base_of<Interface, Implementation>{},
                      "Not derived class of the interface");
        copy = +[](const Storage& src, Storage& dst, Interface** interface) {
            new(std::addressof(dst)) Implementation(*StorageCast<const Implementation>(src));
            *interface = static_cast<Interface*>(StorageCast<Implementation>(dst));
        };

        new(std::addressof(buffer)) Implementation(impl);
        p = static_cast<Interface*>(StorageCast<Implementation>(buffer));
    }

    AnyImplementation(const Derived& rhs) : buffer(), copy(rhs.copy), p(nullptr)
    {
        copy(rhs.buffer, buffer, &p);
    }

    AnyImplementation& operator=(const AnyImplementation& rhs)
    {
        if(&rhs != this)
        {
            if(p)
                p->~Interface();
            copy(rhs.buffer, buffer, &p);
        }
        return *this;
    }

    const Interface* get() const noexcept { return p; }
    const Interface& operator*() const noexcept { return *get(); }
    const Interface* operator->() const noexcept { return get(); }

    ~AnyImplementation() noexcept
    {
        if(p)
            p->~Interface();
    }

private:
    using Storage = std::aligned_storage_t<sizeof(Interface), alignof(Interface)>;
    using Cloner  = void (*)(const Storage&, Storage&, Interface**);

    template <class T, class S>
    static T* StorageCast(S&& s)
    {
        return reinterpret_cast<T*>(std::addressof(s));
    }

    Storage buffer;
    Cloner copy;
    Interface* p;
};

struct AnyTransposePseudoSolver : AnyImplementation<AnyTransposePseudoSolver, TransposePseudoSolver>
{
    AnyTransposePseudoSolver() = default;

    template <class Transpose>
    AnyTransposePseudoSolver(const Transpose& s)
        : AnyImplementation<AnyTransposePseudoSolver, TransposePseudoSolver>(s)
    {
    }

    AnyTransposePseudoSolver(const AnyTransposePseudoSolver& rhs)
        : AnyImplementation<AnyTransposePseudoSolver, TransposePseudoSolver>(rhs)
    {
    }
};

struct UniversalTransposeSolver : TransposePseudoSolver
{
    std::string GetTranspose() const override { return "*-*"; }

    ConvSolution GetSolution(const ExecutionContext& ctx,
                             const TransposeProblem& problem) const override
    {
        auto sln = ConvSolution{};

        {
            auto transposeKernel = KernelInfo{};

            const auto tensor_space = problem.input.GetElementSize();
            const auto cus          = ctx.GetStream().GetMaxComputeUnits();
            const auto group_size   = std::min(tensor_space, cus);

            const auto build_params = GetDataTypeKBP(problem.input.GetType());

            transposeKernel.kernel_file  = "UniversalTranspose.cl";
            transposeKernel.kernel_name  = "UniversalTranspose";
            transposeKernel.g_wk         = {group_size * 16, 1, 1};
            transposeKernel.l_wk         = {group_size * 16, 1, 1};
            transposeKernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

            sln.construction_params.emplace_back(std::move(transposeKernel));
        }

        sln.invoker_factory = [](const std::vector<Kernel>& kernels) {
            const auto kernel = kernels.front();
            return [kernel](const Handle& handle, const AnyInvokeParams& any_params) {
                const auto& params      = any_params.CastTo<TransposeInvokeParams>();
                const auto& lens        = GetNCDHW<uint64_t>(params.in_desc.GetLengths());
                const auto& in_strides  = GetNCDHW<uint64_t>(params.in_desc.GetStrides());
                const auto& out_strides = GetNCDHW<uint64_t>(params.out_desc.GetStrides());

                // clang-format off
                handle.Run(kernel)(
                    params.in, params.out,
                    lens[0],        lens[1],        lens[2],        lens[3],        lens[4],
                    in_strides[0],  in_strides[1],  in_strides[2],  in_strides[3],  in_strides[4],
                    out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4]
                );
                // clang-format on
            };
        };

        return sln;
    }
};

class SegmentedGpuBuffer
{
public:
    SegmentedGpuBuffer(const Handle& handle_, Data_t memory_, std::size_t offset_ = 0)
        : handle(&handle_), memory(memory_), offset(offset_)
    {
        assert(handle);
    }

    SegmentedGpuBuffer(SegmentedGpuBuffer&)  = delete;
    SegmentedGpuBuffer(SegmentedGpuBuffer&&) = delete;
    SegmentedGpuBuffer& operator=(SegmentedGpuBuffer&) = delete;
    SegmentedGpuBuffer& operator=(SegmentedGpuBuffer&&) = delete;

    miopen::shared<Data_t> operator()(std::size_t size)
    {
        const auto align = GetSubbufferAlignment(handle);
        offset += (align - offset) % align;

        auto subbuffer = handle->CreateSubBuffer(memory, offset, size);
        offset += size;

        return subbuffer;
    }

private:
    const Handle* handle;
    Data_t memory;
    std::size_t offset;
};

inline std::string SyncLayoutDims(const char* from, const char* to)
{
    if(strlen(from) < 5)
        return ReplaceString(to, "D", "");
    return to;
}

template <class Problem, class InvokeParams>
struct ProblemTensorTransposeDescriptor
{
    using DescriptorGetter      = TensorDescriptor& (Problem::*)();
    using ConstDescriptorGetter = const TensorDescriptor& (Problem::*)() const;

    DescriptorGetter descriptor;
    ConstDescriptorGetter cdescriptor;
    TensorDescriptor InvokeParams::*rt_descriptor;

    union
    {
        ConstData_t InvokeParams::*as_input;
        Data_t InvokeParams::*as_output;
    };

    const char* to;
    bool is_input;

    template <class Problem_> // to deal with constParameter invalid warning
    inline void Transpose(const Problem& src, Problem_& dest) const
    {
        const auto& desc_from = (src.*cdescriptor)();
        auto& desc_to         = (dest.*descriptor)();
        desc_to               = Transpose(desc_from);
    }

    inline void Transpose(const InvokeParams& src, InvokeParams& dest) const
    {
        const auto& desc_from = src.*rt_descriptor;
        auto& desc_to         = dest.*rt_descriptor;
        desc_to               = Transpose(desc_from);
    }

    inline TensorDescriptor Transpose(const TensorDescriptor& in) const
    {
        const auto labels    = tensor_layout_get_default(in.GetNumDims());
        auto derived_strides = std::vector<size_t>{};
        tensor_layout_to_strides(
            in.GetLengths(), labels, SyncLayoutDims(labels.c_str(), to), derived_strides);
        return {in.GetType(), in.GetLengths(), derived_strides};
    }
};

class ProblemTensorTransposeInvoke
{
public:
    template <class Problem, class InvokeParams>
    ProblemTensorTransposeInvoke(
        SegmentedGpuBuffer& allocator,
        const ProblemTensorTransposeDescriptor<Problem, InvokeParams>& descriptor,
        const Invoker& invoker_,
        const InvokeParams& invoke_params,
        InvokeParams& transposed_params)
        : invoker(invoker_)
    {
        // Transpose runtime tensor descriptor
        descriptor.Transpose(invoke_params, transposed_params);

        const auto& orig_descriptor       = invoke_params.*(descriptor.rt_descriptor);
        const auto& transposed_descriptor = transposed_params.*(descriptor.rt_descriptor);

        // Allocate subbuffer in the workspace
        const auto e_size      = get_data_size(transposed_descriptor.GetType());
        const auto buffer_size = transposed_descriptor.GetElementSpace() * e_size;
        buffer                 = allocator(buffer_size);

        if(descriptor.is_input)
            transposed_params.*(descriptor.as_input) = buffer.get();
        else
            transposed_params.*(descriptor.as_output) = buffer.get();

        if(!descriptor.is_input)
        {
            // Prepare output transpose invoker
            const auto& out = invoke_params.*(descriptor.as_output);

            transpose_params =
                TransposeInvokeParams{buffer.get(), out, transposed_descriptor, orig_descriptor};
            return;
        }

        // Transpose input tensor
        const auto& in = invoke_params.*(descriptor.as_input);

        transpose_params =
            TransposeInvokeParams{in, buffer.get(), orig_descriptor, transposed_descriptor};
    }

    void operator()(const Handle& handle) const
    {
        const auto time = handle.GetKernelTime();
        invoker(handle, transpose_params);
        handle.AccumKernelTime(time);
    }

private:
    miopen::shared<Data_t> buffer;
    Invoker invoker;
    AnyInvokeParams transpose_params;
};

class ProblemTensorTransposeGroup
{
public:
    template <class Problem, class InvokeParams>
    ProblemTensorTransposeGroup(
        const Handle& handle_,
        SegmentedGpuBuffer& allocator,
        const std::vector<
            std::tuple<ProblemTensorTransposeDescriptor<Problem, InvokeParams>, Invoker>>& inputs_,
        const std::vector<
            std::tuple<ProblemTensorTransposeDescriptor<Problem, InvokeParams>, Invoker>>& outputs_,
        const InvokeParams& invoke_params,
        InvokeParams& transposed_params)
        : handle(&handle_)
    {
        std::transform(
            inputs_.begin(), inputs_.end(), std::back_inserter(inputs), [&](auto&& params) {
                return ProblemTensorTransposeInvoke(allocator,
                                                    std::get<0>(params),
                                                    std::get<1>(params),
                                                    invoke_params,
                                                    transposed_params);
            });

        std::transform(
            outputs_.begin(), outputs_.end(), std::back_inserter(outputs), [&](auto&& params) {
                return ProblemTensorTransposeInvoke(allocator,
                                                    std::get<0>(params),
                                                    std::get<1>(params),
                                                    invoke_params,
                                                    transposed_params);
            });

        MIOPEN_LOG_I2("Executing the input transpose");
        for(const auto& transpose : inputs)
            transpose(*handle);
    }

    ProblemTensorTransposeGroup(ProblemTensorTransposeGroup&)  = delete;
    ProblemTensorTransposeGroup(ProblemTensorTransposeGroup&&) = delete;
    ProblemTensorTransposeGroup& operator=(ProblemTensorTransposeGroup&) = delete;
    ProblemTensorTransposeGroup& operator=(ProblemTensorTransposeGroup&&) = delete;

    ~ProblemTensorTransposeGroup()
    {
        MIOPEN_LOG_I2("Executing the output transpose");
        for(const auto& transpose : outputs)
            transpose(*handle);
    }

private:
    const Handle* handle;
    std::vector<ProblemTensorTransposeInvoke> inputs;
    std::vector<ProblemTensorTransposeInvoke> outputs;
};

template <class Derived, class Base, class Problem, class InvokeParams, class Inner>
struct TransposingSolver : Base
{
    using TransposeDescriptor = ProblemTensorTransposeDescriptor<Problem, InvokeParams>;

    static std::vector<AnyTransposePseudoSolver> GetTransposeSolvers()
    {
        return {UniversalTransposeSolver{}};
    }

    static std::unordered_map<std::string, AnyTransposePseudoSolver> GetTransposeSolversMap()
    {
        auto ret = std::unordered_map<std::string, AnyTransposePseudoSolver>{};
        for(const auto& transpose : Derived::GetTransposeSolvers())
            ret.emplace(transpose->GetTranspose(), std::move(transpose));
        return ret;
    }

    bool IsApplicable(const ExecutionContext& ctx, const Problem& problem) const override
    {
        const auto transpose_solvers    = Derived::GetTransposeSolversMap();
        const auto skip_transpose_check = transpose_solvers.find("*-*") != transpose_solvers.end();
        auto any_difference             = false;

        for(auto transpose : Derived::GetTransposes())
        {
            decltype(auto) descriptor = (problem.*(transpose.cdescriptor))();
            const auto layout         = descriptor.GetLayout_str();
            const auto to             = SyncLayoutDims(layout.c_str(), transpose.to);

            auto specific_pair = layout + "-";
            specific_pair.append(to);

            if(!skip_transpose_check &&
               transpose_solvers.find(specific_pair) == transpose_solvers.end() &&
               transpose_solvers.find(layout + "-*") == transpose_solvers.end() &&
               transpose_solvers.find(std::string("*-") + to) == transpose_solvers.end())
                return false;

            any_difference |= layout != to;
        }

        return any_difference && Inner{}.IsApplicable(ctx, Transpose(problem));
    }

    std::size_t GetWorkspaceSize(const ExecutionContext& ctx, const Problem& problem) const override
    {
        const auto transposed_problem = Transpose(problem);
        auto ws_size                  = Inner{}.GetWorkspaceSize(ctx, transposed_problem);

        for(const auto& transpose : Derived::GetTransposes())
        {
            const auto& descriptor = (transposed_problem.*(transpose.cdescriptor))();
            ws_size += descriptor.GetElementSpace();
        }

        return ws_size;
    }

    ConvSolution GetSolution(const ExecutionContext& ctx, const Problem& problem) const override
    {
        auto transposed_problem      = Transpose(problem);
        ConvSolution sln             = Inner{}.GetSolution(ctx, transposed_problem);
        auto old_factory             = *sln.invoker_factory;
        const auto old_kernels_end   = sln.construction_params.size();
        const auto transpose_solvers = Derived::GetTransposeSolversMap();

        std::vector<std::tuple<TransposeDescriptor, InvokerFactory>> in_transpose_ifs,
            out_transpose_ifs;

        for(auto transpose : Derived::GetTransposes())
        {
            const auto& descriptor = (problem.*(transpose.cdescriptor))();
            const auto layout      = descriptor.GetLayout_str();
            const auto to          = SyncLayoutDims(layout.c_str(), transpose.to);

            if(layout == to)
                continue;

            auto specific_pair = layout + "-";
            specific_pair.append(to);

            auto transpose_solver = transpose_solvers.find(specific_pair);
            if(transpose_solver == transpose_solvers.end())
            {
                transpose_solver = transpose_solvers.find(layout + "-*");
                if(transpose_solver == transpose_solvers.end())
                {
                    transpose_solver = transpose_solvers.find(std::string("*-") + to);
                    if(transpose_solver == transpose_solvers.end())
                        transpose_solver = transpose_solvers.find("*-*");
                    assert(transpose_solver != transpose_solvers.end());
                }
            }

            const auto transpose_problem = TransposeProblem{descriptor, layout.c_str()};
            auto transpose_sln = transpose_solver->second->GetSolution(ctx, transpose_problem);

            const auto kernels_begin = sln.construction_params.size();
            sln.construction_params.insert(sln.construction_params.end(),
                                           transpose_sln.construction_params.begin(),
                                           transpose_sln.construction_params.end());
            const auto kernels_end         = sln.construction_params.size();
            const auto raw_invoker_factory = transpose_sln.invoker_factory;

            auto transpose_invoker_factory = [kernels_begin, kernels_end, raw_invoker_factory](
                                                 const std::vector<Kernel>& kernels) {
                auto segment = std::vector<Kernel>{};
                segment.reserve(kernels_end - kernels_begin);
                for(auto i = kernels_begin; i < kernels_end; ++i)
                    segment.push_back(kernels[i]);
                return (*raw_invoker_factory)(segment);
            };

            (transpose.is_input ? in_transpose_ifs : out_transpose_ifs)
                .emplace_back(transpose, std::move(transpose_invoker_factory));
        }

        if(in_transpose_ifs.size() + out_transpose_ifs.size() == 0)
            return sln;

        const auto ws_size = Inner{}.GetWorkspaceSize(ctx, transposed_problem);

        sln.invoker_factory =
            [old_factory, old_kernels_end, ws_size, in_transpose_ifs, out_transpose_ifs](
                const std::vector<Kernel>& kernels) {
                const auto inner_kernels =
                    std::vector<Kernel>{kernels.begin(), kernels.begin() + old_kernels_end};
                std::vector<std::tuple<TransposeDescriptor, Invoker>> in_transpose_invokers,
                    out_transpose_invokers;

                std::transform(in_transpose_ifs.begin(),
                               in_transpose_ifs.end(),
                               std::back_inserter(in_transpose_invokers),
                               [&](const auto& params) {
                                   return std::make_tuple(std::get<0>(params),
                                                          std::get<1>(params)(kernels));
                               });

                std::transform(out_transpose_ifs.begin(),
                               out_transpose_ifs.end(),
                               std::back_inserter(out_transpose_invokers),
                               [&](const auto& params) {
                                   return std::make_tuple(std::get<0>(params),
                                                          std::get<1>(params)(kernels));
                               });

                auto invoker = old_factory(inner_kernels);

                return [invoker, in_transpose_invokers, out_transpose_invokers, ws_size](
                           const Handle& handle, const AnyInvokeParams& any_params) {
                    const auto& invoke_params = any_params.CastTo<InvokeParams>();
                    auto transposed_params    = invoke_params;

                    handle.ResetKernelTime();

                    SegmentedGpuBuffer allocator{handle, invoke_params.workspace, ws_size};

                    ProblemTensorTransposeGroup transposeGroup{handle,
                                                               allocator,
                                                               in_transpose_invokers,
                                                               out_transpose_invokers,
                                                               invoke_params,
                                                               transposed_params};

                    // Execute the invoker provided by the inner solver
                    MIOPEN_LOG_I2("Executing the inner solver invoker");
                    const auto time = handle.GetKernelTime();
                    invoker(handle, transposed_params);
                    handle.AccumKernelTime(time);
                };
            };

        return sln;
    }

private:
    inline static Problem Transpose(const Problem& problem)
    {
        auto transposed_problem = problem;
        for(const auto& transpose : Derived::GetTransposes())
            transpose.Transpose(problem, transposed_problem);
        return transposed_problem;
    }
};

} // namespace solver
} // namespace miopen
