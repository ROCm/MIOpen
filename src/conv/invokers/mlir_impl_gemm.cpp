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

namespace miopen {
namespace conv {

namespace {
struct MlirConvArgs
{
    StridedMemRef5D filter;
    StridedMemRef5D input;
    StridedMemRef5D output;
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

void ComputeMlirDimsStrides(const conv::ProblemDescription& conv_problem,
                            std::vector<size_t>& in_dims,
                            std::vector<size_t>& in_strides,
                            std::vector<size_t>& weights_dims,
                            std::vector<size_t>& weights_strides,
                            std::vector<size_t>& out_dims,
                            std::vector<size_t>& out_strides)
{
    auto group_count = conv_problem.GetGroupCount();

    TensorDescriptor in;
    if(conv_problem.GetDirection() == conv::Direction::Forward)
        in = conv_problem.GetIn();
    else
        in = conv_problem.GetOut();

    in_dims    = in.GetLengths();
    in_strides = in.GetStrides();
    PermuteDimsStrides(in_dims, in_strides);
    // Add a virtual group dimension before input channel.
    InsertGToDimsStrides(in.GetLayout("NCHW"), 'C', group_count, in_dims, in_strides);

    // Add a virtual group dimension before output channel.
    const TensorDescriptor& weights = conv_problem.GetWeights();
    weights_dims                    = weights.GetLengths();
    weights_strides                 = weights.GetStrides();
    PermuteDimsStrides(weights_dims, weights_strides);
    InsertGToDimsStrides(
        weights.GetLayout("NCHW"), 'N', group_count, weights_dims, weights_strides);

    TensorDescriptor out;
    if(conv_problem.GetDirection() == conv::Direction::Forward)
        out = conv_problem.GetOut();
    else
        out = conv_problem.GetIn();

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
                              const std::vector<size_t>& out_strides)
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

    return {filter, input, output};
}

void SetMlirConvArgsPtr(ConstData_t in, ConstData_t out, ConstData_t w, MlirConvArgs& args)
{
    void* filter = nullptr;
    void* input  = nullptr;
    void* output = nullptr;
#if MIOPEN_BACKEND_OPENCL
    clGetMemObjectInfo(w, CL_MEM_HOST_PTR, sizeof(filter), &filter, nullptr);
    clGetMemObjectInfo(in, CL_MEM_HOST_PTR, sizeof(input), &input, nullptr);
    clGetMemObjectInfo(out, CL_MEM_HOST_PTR, sizeof(output), &output, nullptr);
#elif MIOPEN_BACKEND_HIP
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    filter = const_cast<void*>(w);
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    input = const_cast<void*>(in);
    // NOLINTNEXTLINE (cppcoreguidelines-pro-type-const-cast)
    output = const_cast<void*>(out);
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
    ComputeMlirDimsStrides(ctx.conv_problem,
                           in_dims,
                           in_strides,
                           weights_dims,
                           weights_strides,
                           out_dims,
                           out_strides);

    MlirConvArgs args =
        MakeMlirConvArgs(in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides);

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            const auto& forward_invoke_params =
                primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors = forward_invoke_params.tensors;

            SetMlirConvArgsPtr(tensors.in, tensors.out, tensors.w, args);
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
    ComputeMlirDimsStrides(ctx.conv_problem,
                           in_dims,
                           in_strides,
                           weights_dims,
                           weights_strides,
                           out_dims,
                           out_strides);
    MlirConvArgs args =
        MakeMlirConvArgs(in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides);

    const auto& conv  = ctx.conv_problem.GetConv();
    const auto& wDesc = ctx.conv_problem.GetWeights();

    bool every_pixel_is_written = true;
    for(int i = 0; i < conv.GetSpatialDimension(); ++i)
    {
        const auto conv_stride   = conv.GetConvStrides()[i];
        const auto conv_dilation = conv.GetConvDilations()[i];
        const auto filter_size   = wDesc.GetLengths()[2 + i];

        // TODO: This can be relaxed.
        if(!(conv_dilation == 1 && conv_stride <= filter_size))
            every_pixel_is_written = false;
    }

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            float elapsed        = 0;
            const auto& data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;

            if(!every_pixel_is_written)
            {
                float zero = 0.f;
                SetTensor(handle, tensors.outDesc, tensors.out, &zero);

                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
            }

            SetMlirConvArgsPtr(tensors.out, tensors.in, tensors.w, args);
            for(const auto& k : kernels)
            {
                handle.Run(k)(args);
                elapsed += handle.GetKernelTime();
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

InvokerFactory MakeMlirWrWInvokerFactory(const ConvolutionContext& ctx)
{
    assert((ctx.direction.IsBackwardWrW()));

    std::vector<size_t> in_dims, in_strides;
    std::vector<size_t> weights_dims, weights_strides;
    std::vector<size_t> out_dims, out_strides;
    ComputeMlirDimsStrides(ctx.conv_problem,
                           in_dims,
                           in_strides,
                           weights_dims,
                           weights_strides,
                           out_dims,
                           out_strides);
    MlirConvArgs args =
        MakeMlirConvArgs(in_dims, in_strides, weights_dims, weights_strides, out_dims, out_strides);

    return [=](const std::vector<Kernel>& kernels) mutable {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) mutable {
            const auto& wrw_invoke_params = primitive_parameters.CastTo<conv::WrWInvokeParams>();
            const auto& tensors           = wrw_invoke_params.tensors;

            SetMlirConvArgsPtr(tensors.x, tensors.dy, tensors.dw, args);
            handle.Run(kernels[0])(args);
        };
    };
}

} // namespace conv
} // namespace miopen
