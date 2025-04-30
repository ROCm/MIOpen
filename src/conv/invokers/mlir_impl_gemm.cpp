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

#include <miopen/conv/invokers/mlir_impl_gemm.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>

#include <Miir.h>

#include <boost/any.hpp>
#include <boost/range/adaptors.hpp>

#if MIIR_VERSION_FLAT >= 6
#define MIIR_BARE_POINTER_ABI 1
#else
#define MIIR_BARE_POINTER_ABI 0
#endif

namespace miopen {
namespace conv {
namespace {
#if MIIR_BARE_POINTER_ABI
struct MlirConvArgs
{
    void* filter;
    void* input;
    void* output;
    void* workspace;
};
#else
struct MlirConvArgs
{
    StridedMemRef5D filter;
    StridedMemRef5D input;
    StridedMemRef5D output;
    StridedMemRef5D workspace;
};
#endif

// Note: Below macros are required for opencl backend only because
// opencl backend requires to invoke clSetKernelArg on every kernel
// arguments whereas hip can implicitly linearize the struct to
// kernel arguments
#if MIOPEN_BACKEND_OPENCL
#define EXPAND_ARRAY_5(x) ((x)[0]), ((x)[1]), ((x)[2]), ((x)[3]), ((x)[4])
#define EXPAND_MLIR_CONV_ARGS(x) (x).offset, EXPAND_ARRAY_5((x).sizes), EXPAND_ARRAY_5((x).strides)
#endif

#if MIIR_BARE_POINTER_ABI
void ComputeMlirDimsStrides(const ProblemDescription& problem,
                            std::vector<size_t>& in_dims,
                            std::vector<size_t>& in_strides,
                            std::vector<size_t>& weights_dims,
                            std::vector<size_t>& weights_strides,
                            std::vector<size_t>& out_dims,
                            std::vector<size_t>& out_strides)
{
    // The bare pointer ABI doesn't require this info, so this is a noop.
    (void)problem;
    (void)in_dims;
    (void)in_strides;
    (void)weights_dims;
    (void)weights_strides;
    (void)out_dims;
    (void)out_strides;
}

MlirConvArgs MakeMlirConvArgs(const std::vector<size_t>& in_dims,
                              const std::vector<size_t>& in_strides,
                              const std::vector<size_t>& weights_dims,
                              const std::vector<size_t>& weights_strides,
                              const std::vector<size_t>& out_dims,
                              const std::vector<size_t>& out_strides,
                              size_t workspace_req)
{
    // The bare pointer ABI doesn't pass dimensions or strides,
    // so the arguments are present for backward compatibility
    // and are mentioned here to silence unused argument warnings.
    (void)in_dims;
    (void)in_strides;
    (void)weights_dims;
    (void)weights_strides;
    (void)out_dims;
    (void)out_strides;
    (void)workspace_req;
    return {nullptr, nullptr, nullptr, nullptr};
}
#else
// Rearrange strides correctly
// In MLIR: the layout, sizes and strides are coherent. The layout information is not
// embedded into the permutation of strides.
// - For NCHW, sizes = {N, C, H, W}; strides = {C*H*W, H*W, W, 1}
// - For NHWC, sizes = {N, H, W, C}; strides = {C*H*W, W*C, C, 1}

// In MIOpen however, size and strides are not aligned. Permutation of the strides are used to
// infer actual layout
// - For NCHW, sizes = {N, C, H, W}; strides = {C*H*W, H*W, W, 1}
// - For NHWC, sizes = {N, C, H, W}; strides = {C*H*W, 1, W*C, C}
void PermuteDimsStrides(std::vector<size_t>& dims, std::vector<size_t>& strides)
{
    auto sorted_dims    = dims;
    auto sorted_strides = strides;
    auto p              = TensorDescriptor::find_permutation(dims, strides);
    std::transform(p.begin(), p.end(), sorted_dims.begin(), [&](auto i) { return dims[i]; });
    std::transform(p.begin(), p.end(), sorted_strides.begin(), [&](auto i) { return strides[i]; });
    dims    = sorted_dims;
    strides = sorted_strides;
};

void InsertGToDimsStrides(const std::string& layout,
                          char dim,
                          int group_count,
                          std::vector<size_t>& dims,
                          std::vector<size_t>& strides)
{
    std::size_t index = layout.find(dim);
    if(index == std::string::npos)
        MIOPEN_THROW("Failed to find channel in the layout");

    // For dimensions,
    //    Insert an additional dimension g before channel
    //    Amend the channel size to be channel_size / group_count
    dims[index] /= group_count;
    dims.insert(dims.begin() + index, group_count);

    // For strides,
    //   The channel stride remain the same
    //   Insert an additional group stride to be channel_stride * new_channel_size
    strides.insert(strides.begin() + index, strides[index] * dims[index + 1]);
}

void ComputeMlirDimsStrides(const ProblemDescription& problem,
                            std::vector<size_t>& in_dims,
                            std::vector<size_t>& in_strides,
                            std::vector<size_t>& weights_dims,
                            std::vector<size_t>& weights_strides,
                            std::vector<size_t>& out_dims,
                            std::vector<size_t>& out_strides)
{
    auto group_count = problem.GetGroupCount();

    TensorDescriptor in;
    if(problem.GetDirection() == conv::Direction::Forward)
        in = problem.GetIn();
    else
        in = problem.GetOut();

    in_dims    = in.GetLengths();
    in_strides = in.GetStrides();
    PermuteDimsStrides(in_dims, in_strides);
    // Add a virtual group dimension before input channel.
    InsertGToDimsStrides(in.GetLayout("NCHW"), 'C', group_count, in_dims, in_strides);

    // Add a virtual group dimension before output channel.
    const TensorDescriptor& weights = problem.GetWeights();
    weights_dims                    = weights.GetLengths();
    weights_strides                 = weights.GetStrides();
    PermuteDimsStrides(weights_dims, weights_strides);
    InsertGToDimsStrides(
        weights.GetLayout("NCHW"), 'N', group_count, weights_dims, weights_strides);

    TensorDescriptor out;
    if(problem.GetDirection() == conv::Direction::Forward)
        out = problem.GetOut();
    else
        out = problem.GetIn();

    out_dims    = out.GetLengths();
    out_strides = out.GetStrides();
    PermuteDimsStrides(out_dims, out_strides);
    // Add a virtual group dimension before output channel.
    InsertGToDimsStrides(out.GetLayout("NCHW"), 'C', group_count, out_dims, out_strides);
}

MlirConvArgs MakeMlirConvArgs(const std::vector<size_t>& in_dims,
                              const std::vector<size_t>& in_strides,
                              const std::vector<size_t>& weights_dims,
                              const std::vector<size_t>& weights_strides,
                              const std::vector<size_t>& out_dims,
                              const std::vector<size_t>& out_strides,
                              size_t workspace_req)
{
    auto initDimStrides = [](const std::vector<size_t>& dims,
                             const std::vector<size_t>& strides,
                             StridedMemRef5D& target) {
        std::copy(dims.cbegin(), dims.cend(), &target.sizes[0]);
        std::copy(strides.cbegin(), strides.cend(), &target.strides[0]);
    };

    StridedMemRef5D filter{nullptr, nullptr, 0, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    initDimStrides(weights_dims, weights_strides, filter);
    StridedMemRef5D input{nullptr, nullptr, 0, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    initDimStrides(in_dims, in_strides, input);
    StridedMemRef5D output{nullptr, nullptr, 0, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    initDimStrides(out_dims, out_strides, output);

    StridedMemRef5D workspace{nullptr, nullptr, 0, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    if(workspace_req > 0)
    {
        initDimStrides(weights_dims, weights_strides, workspace);
    }

    return {filter, input, output, workspace};
}
#endif // MIIR_BARE_POINTER_ABI = 0
// Note: This does not work for opencl backend because it is impossible
// to extract the device pointer out from a ocl memory object. The only
// way around is to call clSetKernelArg on a oclMemory object to pass
// the device pointer to the kernel
#if MIOPEN_BACKEND_HIP
void SetMlirConvArgsPtr(ConstData_t in, ConstData_t out, ConstData_t w, MlirConvArgs& args)
{
    void* filter = nullptr;
    void* input  = nullptr;
    void* output = nullptr;
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    filter = const_cast<void*>(w);
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    input = const_cast<void*>(in);
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    output = const_cast<void*>(out);

    if((filter == nullptr) || (input == nullptr) || (output == nullptr))
    {
        MIOPEN_THROW("Invalid device pointers");
    }
#if MIIR_BARE_POINTER_ABI
    args.filter = filter;
    args.input  = input;
    args.output = output;
#else
    args.filter.basePtr    = filter;
    args.filter.data       = filter;
    args.input.basePtr     = input;
    args.input.data        = input;
    args.output.basePtr    = output;
    args.output.data       = output;
#endif
}

void SetMlirConvArgsPtr(
    ConstData_t in, ConstData_t out, ConstData_t w, ConstData_t wk, MlirConvArgs& args)
{
    SetMlirConvArgsPtr(in, out, w, args);
    void* workspace = nullptr;
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    workspace = const_cast<void*>(wk);

    if(workspace == nullptr)
    {
        MIOPEN_THROW("Invalid device pointers for workspace");
    }

#if MIIR_BARE_POINTER_ABI
    args.workspace = workspace;
#else
    args.workspace.basePtr = workspace;
    args.workspace.data    = workspace;
#endif
}
#endif // MIOPEN_BACKEND_HIP
} // Anonymous namespace

InvokerFactory MakeMlirFwdInvokerFactory(const ProblemDescription& problem)
{
    assert((problem.IsDirectionForward()));

    std::vector<size_t> in_dims, in_strides;
    std::vector<size_t> weights_dims, weights_strides;
    std::vector<size_t> out_dims, out_strides;
    ComputeMlirDimsStrides(
        problem, in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides);

    MlirConvArgs args = MakeMlirConvArgs(
        in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides, 0);

    const auto& conv             = problem.GetConv();
    const auto& lowp_quant       = conv.lowp_quant;
    const auto& outDesc          = problem.GetOut();
    TensorDescriptor outConvDesc = outDesc;
    // outConvDesc is only functional when this is a int8 convolution. It allows the output type to
    // be cast to a different than int32_t. This gives the solver a wider applicable range and
    // mimics the behavior of the gemm solver.
    bool needs_output_cast = false;
    if(problem.GetIn().GetType() == miopenInt8 && problem.GetWeights().GetType() == miopenInt8 &&
       problem.GetOut().GetType() != miopenInt32)
    {
        needs_output_cast = true;
        outConvDesc = TensorDescriptor(miopenInt32, outDesc.GetLengths(), outDesc.GetStrides());
    }

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            const auto& forward_invoke_params =
                primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors = forward_invoke_params.tensors;

#if MIOPEN_BACKEND_OPENCL
#if MIIR_BARE_POINTER_ABI
            handle.Run(kernels[0])(tensors.w, tensors.in, tensors.out);
            (void)args;
#else
            handle.Run(kernels[0])(tensors.w,
                                   tensors.w,
                                   EXPAND_MLIR_CONV_ARGS(args.filter),
                                   tensors.in,
                                   tensors.in,
                                   EXPAND_MLIR_CONV_ARGS(args.input),
                                   tensors.out,
                                   tensors.out,
                                   EXPAND_MLIR_CONV_ARGS(args.output));
#endif // MIIR_BARE_POINTER_ABI
#elif MIOPEN_BACKEND_HIP
            SetMlirConvArgsPtr(tensors.in, tensors.out, tensors.w, args);
            handle.Run(kernels[0])(args);
#endif
            if(needs_output_cast)
            {
                CastTensor(handle,
                           &lowp_quant,
                           true,
                           outConvDesc,
                           tensors.out,
                           tensors.outDesc,
                           tensors.out,
                           0,
                           0);
            }
        };
    };
}

InvokerFactory MakeMlirBwdInvokerFactory(const ProblemDescription& problem)
{
    assert(problem.IsDirectionBackwardData());

    std::vector<size_t> in_dims, in_strides;
    std::vector<size_t> weights_dims, weights_strides;
    std::vector<size_t> out_dims, out_strides;
    ComputeMlirDimsStrides(
        problem, in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides);
    MlirConvArgs args = MakeMlirConvArgs(
        in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides, 0);

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            float elapsed        = 0.f;
            const auto& data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;

#if MIOPEN_BACKEND_OPENCL
            for(const auto& k : kernels)
            {
#if MIIR_BARE_POINTER_ABI
                handle.Run(k)(tensors.w, tensors.out, tensors.in);
                (void)args;
#else
                handle.Run(k)(tensors.w,
                              tensors.w,
                              EXPAND_MLIR_CONV_ARGS(args.filter),
                              tensors.out,
                              tensors.out,
                              EXPAND_MLIR_CONV_ARGS(args.output),
                              tensors.in,
                              tensors.in,
                              EXPAND_MLIR_CONV_ARGS(args.input));
#endif
                elapsed += handle.GetKernelTime();
            }
#elif MIOPEN_BACKEND_HIP
            SetMlirConvArgsPtr(tensors.out, tensors.in, tensors.w, args);
            for(const auto& k : kernels)
            {
                handle.Run(k)(args);
                elapsed += handle.GetKernelTime();
            }
#endif

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

InvokerFactory MakeMlirWrWInvokerFactory(const ProblemDescription& problem, size_t workspace_req)
{
    assert((problem.IsDirectionBackwardWrW()));

    std::vector<size_t> in_dims, in_strides;
    std::vector<size_t> weights_dims, weights_strides;
    std::vector<size_t> out_dims, out_strides;
    ComputeMlirDimsStrides(
        problem, in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides);
    MlirConvArgs args = MakeMlirConvArgs(
        in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides, workspace_req);

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            float elapsed                 = 0.f;
            const auto& wrw_invoke_params = primitive_parameters.CastTo<conv::WrWInvokeParams>();
            const auto& tensors           = wrw_invoke_params.tensors;

            if(workspace_req > 0)
            {
                const auto& workspace    = wrw_invoke_params.workSpace;
                const auto workspaceSize = wrw_invoke_params.workSpaceSize;

                if((workspace == nullptr) || (workspaceSize < workspace_req))
                {
                    MIOPEN_THROW("Not enough workspace for MLIR WRW (" +
                                 std::to_string(workspaceSize) + " provided, " +
                                 std::to_string(workspace_req) + " required)");
                }

                TensorDescriptor workspaceDesc(
                    miopenFloat, tensors.dwDesc.GetLengths(), tensors.dwDesc.GetStrides());

#if MIOPEN_BACKEND_OPENCL
                for(const auto& k : kernels)
                {
#if MIIR_BARE_POINTER_ABI
                    handle.Run(k)(tensors.dw, tensors.x, tensors.dy, workspace);
                    (void)args;
#else
                    handle.Run(k)(tensors.dw,
                                  tensors.dw,
                                  EXPAND_MLIR_CONV_ARGS(args.filter),
                                  tensors.x,
                                  tensors.x,
                                  EXPAND_MLIR_CONV_ARGS(args.input),
                                  tensors.dy,
                                  tensors.dy,
                                  EXPAND_MLIR_CONV_ARGS(args.output),
                                  workspace,
                                  workspace,
                                  EXPAND_MLIR_CONV_ARGS(args.workspace));
#endif
                    elapsed += handle.GetKernelTime();
                }
#elif MIOPEN_BACKEND_HIP
                SetMlirConvArgsPtr(tensors.x, tensors.dy, tensors.dw, workspace, args);
                for(const auto& k : kernels)
                {
                    handle.Run(k)(args);
                    elapsed += handle.GetKernelTime();
                }
#endif
            }
            else
            {
#if MIOPEN_BACKEND_OPENCL
                for(const auto& k : kernels)
                {
#if MIIR_BARE_POINTER_ABI
                    handle.Run(k)(tensors.dw, tensors.x, tensors.dy);
#else
                    handle.Run(k)(tensors.dw,
                                  tensors.dw,
                                  EXPAND_MLIR_CONV_ARGS(args.filter),
                                  tensors.x,
                                  tensors.x,
                                  EXPAND_MLIR_CONV_ARGS(args.input),
                                  tensors.dy,
                                  tensors.dy,
                                  EXPAND_MLIR_CONV_ARGS(args.output));
#endif
                    elapsed += handle.GetKernelTime();
                }
#elif MIOPEN_BACKEND_HIP
                SetMlirConvArgsPtr(tensors.x, tensors.dy, tensors.dw, args);
                for(const auto& k : kernels)
                {
                    handle.Run(k)(args);
                    elapsed += handle.GetKernelTime();
                }
#endif
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

} // namespace conv
} // namespace miopen
