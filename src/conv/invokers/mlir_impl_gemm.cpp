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
#include <miopen/memref.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>

#include <boost/any.hpp>
#include <boost/range/adaptors.hpp>

namespace miopen {
namespace conv {

namespace {
using MemRef4DGeneric = StridedMemRefType<void, 4>;

struct MlirConvArgs
{
    MemRef4DGeneric filter;
    MemRef4DGeneric input;
    MemRef4DGeneric output;
};

// Rearrange strides correctly
// In MLIR: the layout, sizes and strides are coherent. The layout information is not
// embedded into the permutation of strides.
// - For NCHW, sizes = {N, C, H, W}; strides = {C*H*W, H*W, W, 1}
// - For NHWC, sizes = {N, H, W, C}; strides = {C*H*W, W*C, C, 1}

// In MIOpen however, size and strides are not aligned. Permutation of the strides are used to
// infer actual layout
// - For NCHW, sizes = {N, C, H, W}; strides = {C*H*W, H*W, W, 1}
// - For NHWC, sizes = {N, C, H, W}; strides = {C*H*W, 1, W*C, C}
auto permuteDimsStrides(const std::vector<size_t>& dims, const std::vector<size_t>& strides)
{
    auto sorted_dims    = dims;
    auto sorted_strides = strides;
    auto p              = TensorDescriptor::find_permutation(dims, strides);
    std::transform(p.begin(), p.end(), sorted_dims.begin(), [&](auto i) { return dims[i]; });
    std::transform(p.begin(), p.end(), sorted_strides.begin(), [&](auto i) { return strides[i]; });
    return std::make_tuple(sorted_dims, sorted_strides);
};

void permuteDimStridesAllDir(const conv::ProblemDescription& conv_problem,
                             std::vector<size_t>& in_dims,
                             std::vector<size_t>& in_strides,
                             std::vector<size_t>& weights_dims,
                             std::vector<size_t>& weights_strides,
                             std::vector<size_t>& out_dims,
                             std::vector<size_t>& out_strides)
{
    const TensorDescriptor& in = conv_problem.GetIn();
    std::make_tuple(in_dims, in_strides) = permuteDimsStrides(in.GetLengths(), in.GetStrides());

    const TensorDescriptor& weights = conv_problem.GetWeights();
    std::make_tuple(weights_dims, weights_strides) =
        permuteDimsStrides(weights.GetLengths(), weights.GetStrides());

    const TensorDescriptor& out = conv_problem.GetOut();
    std::make_tuple(out_dims, out_strides) = permuteDimsStrides(out.GetLengths(), out.GetStrides());
}

MlirConvArgs makeMlirConvArgs(const std::vector<size_t>& in_dims,
                              const std::vector<size_t>& in_strides,
                              const std::vector<size_t>& weights_dims,
                              const std::vector<size_t>& weights_strides,
                              const std::vector<size_t>& out_dims,
                              const std::vector<size_t>& out_strides)
{
    auto initDimStrides = [](const std::vector<size_t>& dims,
                             const std::vector<size_t>& strides,
                             MemRef4DGeneric& target) {
        std::copy(dims.cbegin(), dims.cend(), &target.sizes[0]);
        std::copy(strides.cbegin(), strides.cend(), &target.strides[0]);
    };

    MemRef4DGeneric filter{nullptr, nullptr, 0, {0, 0, 0, 0}, {0, 0, 0, 0}};
    initDimStrides(weights_dims, weights_strides, filter);
    MemRef4DGeneric input{nullptr, nullptr, 0, {0, 0, 0, 0}, {0, 0, 0, 0}};
    initDimStrides(in_dims, in_strides, input);
    MemRef4DGeneric output{nullptr, nullptr, 0, {0, 0, 0, 0}, {0, 0, 0, 0}};
    initDimStrides(out_dims, out_strides, output);

    return {filter, input, output};
}

void setMlirConvArgsPtr(const ConvDataTensors& tensors, MlirConvArgs& args)
{
    void* filter = nullptr;
    void* input  = nullptr;
    void* output = nullptr;
#if MIOPEN_BACKEND_OPENCL
    clGetMemObjectInfo(tensors.w, CL_MEM_HOST_PTR, sizeof(filter), &filter, nullptr);
    clGetMemObjectInfo(tensors.in, CL_MEM_HOST_PTR, sizeof(input), &input, nullptr);
    clGetMemObjectInfo(tensors.out, CL_MEM_HOST_PTR, sizeof(output), &output, nullptr);
#elif MIOPEN_BACKEND_HIP
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    filter = const_cast<void*>(tensors.w);
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    input = const_cast<void*>(tensors.in);
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    output = const_cast<void*>(tensors.out);
#endif

    if((filter == nullptr) || (input == nullptr) || (output == nullptr))
        MIOPEN_THROW("Invalid device pointers");

    args.filter.basePtr = filter;
    args.filter.data    = filter;
    args.input.basePtr  = input;
    args.input.data     = input;
    args.output.basePtr = output;
    args.output.data    = output;
}
} // Anonymous namespace

InvokerFactory MakeMlirFwdInvokerFactory(const ConvolutionContext& ctx)
{
    assert((ctx.direction.IsForward()));

    std::vector<size_t> in_dims, in_strides;
    std::vector<size_t> weights_dims, weights_strides;
    std::vector<size_t> out_dims, out_strides;
    permuteDimStridesAllDir(ctx.conv_problem,
                            in_dims,
                            in_strides,
                            weights_dims,
                            weights_strides,
                            out_dims,
                            out_strides);

    MlirConvArgs args =
        makeMlirConvArgs(in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides);

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            const auto& forward_invoke_params =
                primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors = forward_invoke_params.tensors;

            setMlirConvArgsPtr(tensors, args);
            handle.Run(kernels[0])(args);
        };
    };
}

InvokerFactory MakeMlirBwdInvokerFactory(const ConvolutionContext& ctx)
{
    assert(ctx.direction.IsBackwardData());

    std::vector<size_t> in_dims, in_strides;
    std::vector<size_t> weights_dims, weights_strides;
    std::vector<size_t> out_dims, out_strides;
    permuteDimStridesAllDir(ctx.conv_problem,
                            in_dims,
                            in_strides,
                            weights_dims,
                            weights_strides,
                            out_dims,
                            out_strides);
    MlirConvArgs args =
        makeMlirConvArgs(in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides);

    const auto& conv  = ctx.conv_problem.GetConv();
    const auto& wDesc = ctx.conv_problem.GetWeights();

    const bool isFilter1Padding0Stride1 = [&]() {
        const auto wei_spatial =
            boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + conv.GetSpatialDimension());
        return miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
               miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
               miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; });
    }();

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            float elapsed        = 0;
            const auto& data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;

            if(!isFilter1Padding0Stride1)
            {
                float zero = 0.f;
                SetTensor(handle, tensors.outDesc, tensors.out, &zero);
            }

            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();

            setMlirConvArgsPtr(tensors, args);
            handle.Run(kernels[0])(args);
            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

} // namespace conv
} // namespace miopen
