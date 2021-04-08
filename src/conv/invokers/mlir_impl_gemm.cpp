#include <miopen/conv/invokers/mlir_impl_gemm.hpp>
#include <miopen/memref.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>

#include <boost/any.hpp>

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
    auto p              = TensorDescriptor::sort_permutation(strides, std::greater<>{});
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

MlirConvArgs MakeMlirConvArgs(ConstData_t in,
                              const std::vector<size_t>& in_dims,
                              const std::vector<size_t>& in_strides,
                              ConstData_t w,
                              const std::vector<size_t>& weights_dims,
                              const std::vector<size_t>& weights_strides,
                              ConstData_t out,
                              const std::vector<size_t>& out_dims,
                              const std::vector<size_t>& out_strides)
{
    auto cpyToMemRef = [](const void* ptr,
                          const std::vector<size_t>& dims,
                          const std::vector<size_t>& strides,
                          MemRef4DGeneric& target) {
        target.basePtr = const_cast<void*>(ptr); // NOLINT (cppcoreguidelines-pro-type-const-cast)
        target.data    = const_cast<void*>(ptr); // NOLINT (cppcoreguidelines-pro-type-const-cast)
        target.offset  = 0;
        std::copy(dims.cbegin(), dims.cend(), &target.sizes[0]);
        std::copy(strides.cbegin(), strides.cend(), &target.strides[0]);
    };

    MlirConvArgs args{};
    cpyToMemRef(w, weights_dims, weights_strides, args.filter);
    cpyToMemRef(in, in_dims, in_strides, args.input);
    cpyToMemRef(out, out_dims, out_strides, args.output);
    return args;
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

    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;

            MlirConvArgs args = MakeMlirConvArgs(tensors.in,
                                                 in_dims,
                                                 in_strides,
                                                 tensors.w,
                                                 weights_dims,
                                                 weights_strides,
                                                 tensors.out,
                                                 out_dims,
                                                 out_strides);
            handle.Run(kernels[0])(args);
            if(handle.IsProfilingEnabled())
            {
                float elapsed = 0;
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
}

} // namespace conv
} // namespace miopen
