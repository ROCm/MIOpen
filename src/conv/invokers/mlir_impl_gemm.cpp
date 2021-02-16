#include <miopen/conv/invokers/mlir_impl_gemm.hpp>
#include <miopen/memref.hpp>

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>

#include <boost/any.hpp>

namespace miopen {
namespace conv {

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

InvokerFactory MakeMlirFwdInvokerFactory(const ConvolutionContext& ctx)
{
    assert(ctx.direction.IsForward());

    // Rearrange strides correctly
    // In MLIR: the layout, sizes and strides are coherent. The layout information is not
    // embedded into the permutation of strides.
    // - For NCHW, sizes = {N, C, H, W}; strides = {C*H*W, H*W, W, 1}
    // - For NHWC, sizes = {N, H, W, C}; strides = {C*H*W, W*C, C, 1}

    // In MIOpen however, size and strides are not aligned. Permutation of the strides are used to
    // infer actual layout
    // - For NCHW, sizes = {N, C, H, W}; strides = {C*H*W, H*W, W, 1}
    // - For NHWC, sizes = {N, C, H, W}; strides = {C*H*W, 1, W*C, C}
    auto permuteDimsStrides = [](const std::vector<size_t>& dims,
                                 const std::vector<size_t>& strides) {
        auto sorted_dims    = dims;
        auto sorted_strides = strides;
        auto p              = TensorDescriptor::sort_permutation(strides, std::greater<>{});
        std::transform(p.begin(), p.end(), sorted_dims.begin(), [&](auto i) { return dims[i]; });
        std::transform(
            p.begin(), p.end(), sorted_strides.begin(), [&](auto i) { return strides[i]; });
        return std::make_tuple(sorted_dims, sorted_strides);
    };

    TensorDescriptor in      = ctx.conv_problem.GetIn();
    TensorDescriptor weights = ctx.conv_problem.GetWeights();
    TensorDescriptor out     = ctx.conv_problem.GetOut();

    std::vector<size_t> in_dims, in_strides;
    std::make_tuple(in_dims, in_strides) = permuteDimsStrides(in.GetLengths(), in.GetStrides());

    std::vector<size_t> weights_dims, weights_strides;
    std::make_tuple(weights_dims, weights_strides) =
        permuteDimsStrides(weights.GetLengths(), weights.GetStrides());

    std::vector<size_t> out_dims, out_strides;
    std::make_tuple(out_dims, out_strides) = permuteDimsStrides(out.GetLengths(), out.GetStrides());

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
        };
    };
}

} // namespace conv
} // namespace miopen
