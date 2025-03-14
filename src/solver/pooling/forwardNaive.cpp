/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/pooling.hpp>
#include <miopen/pooling/invoke_params.hpp>
#include <miopen/pooling/solvers.hpp>

#define WORKAROUND_ISSUE_MIFIN_80 1 // https://github.com/ROCm/MIFin/issues/80

namespace miopen {

namespace solver {

namespace pooling {

namespace {

#if !MIOPEN_NDEBUG && !WORKAROUND_ISSUE_MIFIN_80
template <typename T>
bool IsPower2(T v)
{
    return (v != 0) && ((v & (v - 1)) == 0);
}
#endif

template <typename T>
T RoundUpNearestPower2Positive(T v) = delete;

inline uint32_t RoundUpNearestPower2Positive(uint32_t v)
{
    assert(v > 0);
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return std::max(++v, 1U); // Shut clang-tidy.
}

} // namespace

bool PoolingForwardNaive::IsApplicable(const ExecutionContext&,
                                       const miopen::pooling::ProblemDescription& problem) const
{
    return problem.GetDirection() == miopen::pooling::Direction::Forward           //
           && problem.GetXDesc().GetType() == problem.GetYDesc().GetType()         //
           && (problem.GetXDesc().GetType() == miopenFloat                         //
               || problem.GetXDesc().GetType() == miopenHalf)                      //
           && (problem.GetPooling().GetMode() == miopenPoolingMax                  //
               || problem.GetPooling().GetMode() == miopenPoolingAverage           //
               || problem.GetPooling().GetMode() == miopenPoolingAverageInclusive) //
           && (                                                                    //
                  (problem.GetXDesc().GetNumDims() == 5                            //
                   && problem.GetXDesc().IsPossibleLayout4D5D("NCDHW")             //
                   && problem.GetYDesc().IsPossibleLayout4D5D("NCDHW"))            //
                  ||                                                               //
                  (problem.GetXDesc().GetNumDims() == 4                            //
                   && problem.GetXDesc().IsPossibleLayout4D5D("NCHW")              //
                   && problem.GetYDesc().IsPossibleLayout4D5D("NCHW"))             //
              );
}

ConvSolution
PoolingForwardNaive::GetSolution(const ExecutionContext& context,
                                 const miopen::pooling::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const auto bot  = problem.GetXDesc();
    const auto top  = problem.GetYDesc();
    const bool is2d = (bot.GetNumDims() == 4);

    // To compact code:
    const auto& pooling = problem.GetPooling();
    const auto& lengths = pooling.GetLengths();
    const auto& strides = pooling.GetStrides();
    const auto& pads    = pooling.GetPads();

    // This also deduces 3D (DHW) parameters from 2D (HW) descriptor.
    const uint32_t filter_w        = lengths[is2d ? 1 : 2];
    const uint32_t filter_h        = lengths[is2d ? 0 : 1];
    const uint32_t filter_d        = is2d ? 1 : lengths[0];
    const uint32_t filter_w_stride = strides[is2d ? 1 : 2];
    const uint32_t filter_h_stride = strides[is2d ? 0 : 1];
    const uint32_t filter_d_stride = is2d ? (filter_h_stride * filter_d) : strides[0];
    const uint32_t filter_w_pad    = pads[is2d ? 1 : 2];
    const uint32_t filter_h_pad    = pads[is2d ? 0 : 1];
    const uint32_t filter_d_pad    = is2d ? 0 : pads[0];

    const int pooling_method = (pooling.GetMode() == miopenPoolingMax) ? MLO_POOLING_OP_MAX
                               : (pooling.GetMode() == miopenPoolingAverage)
                                   ? MLO_POOLING_OP_AVE
                                   : MLO_POOLING_OP_AVE_INCLUSIVE;

    const auto save_index = problem.SaveIndex();
    const auto index_mode = pooling.GetWorkspaceIndexMode();
    const auto index_type = pooling.GetIndexType();

    /// \anchor multiply_dims_overflow_assumption
    ///
    /// Preventing overflow during dimension-related computations:
    /// Let's assume that multiplication of three dims always fits into 32 bits (unsigned).
    /// Then let's use size_t when we need to multiply more than three dims.
    /// For example, in NCDHW layout, the N and C strides are results of multiplication
    /// of >= 3 dims, so we have to use size_t for storing them.
    ///
    /// We need to pay special attention to muls of D stride with some other dims.
    /// The D stride is a result of 2 muls. Therefore (d_stride * dim) does
    /// not require widening to size_t prior mul, but (d_stride * dim * dim)
    /// requires it because the total number of muls is 4.

    const auto spatial_dim = is2d ? 2U : 3U;
    uint32_t all_n, all_c, bot_d, bot_h, bot_w;
    std::tie(all_n, all_c, bot_d, bot_h, bot_w) = miopen::GetNCDHW(spatial_dim, bot.GetLengths());
    uint32_t bot_w_stride, bot_h_stride, bot_d_stride;
    size_t bot_c_stride, bot_n_stride;
    std::tie(bot_n_stride, bot_c_stride, bot_d_stride, bot_h_stride, bot_w_stride) =
        miopen::GetNCDHW(spatial_dim, bot.GetStrides());

    uint32_t top_d, top_h, top_w;
    std::tie(std::ignore, std::ignore, top_d, top_h, top_w) =
        miopen::GetNCDHW(spatial_dim, top.GetLengths());
    uint32_t top_w_stride, top_h_stride, top_d_stride;
    size_t top_c_stride, top_n_stride;
    std::tie(top_n_stride, top_c_stride, top_d_stride, top_h_stride, top_w_stride) =
        miopen::GetNCDHW(spatial_dim, top.GetStrides());

    // Mask data is always NCDHW
    const uint32_t mask_w_stride = 1;
    const uint32_t mask_h_stride = mask_w_stride * top_w;
    const uint32_t mask_d_stride = mask_h_stride * top_h;
    const size_t mask_c_stride   = static_cast<size_t>(mask_d_stride) * top_d;
    const size_t mask_n_stride   = mask_c_stride * all_c;

    /// About optimal grid size. The simplest way is to map the problem onto grid is 1:1 mapping of
    /// N,C and top.D onto grid dimensions.
    ///
    /// However, this would waste 1 dimension of grid for 2D convolutions, i.e. the grid size would
    /// be N*C*1, which might be too small and lead to under-utilization of GPU. If we exchange D
    /// with H then the grid size for 2D problem would be N*C*H, but for 3D problem the kernel will
    /// access memory in a scattered way, which would affect performance again. Current design
    /// choice is using separate 2D and 3D kernels (via build-time parameter) and N*C*H grid for 2D.
    ///
    /// \anchor naive_pooling_max_grid_size
    /// * Assumption: Max grid size is >= 2^32-1 (4G-1) i.e. std::max<unint32_t>.
    ///   Currently this limitation is valid for both ROCm HIP and OCL runtimes.
    ///
    /// Another problem with this simple approach is finding out the optimal workgroup size.
    /// The trivial solution is {1,1,1}, but this would lead to under-utilization of GPU, because
    /// in this case only 1 thread out of the 64/32 available in the wavefront will be used.
    /// Let's use workgroup size (w0*w1*w2) = WAVESIZE.
    ///
    /// We have to use workroup which is, after multiplication by some integer, gives exactly grid
    /// (w0*I == g0, w1*J == g1, w2*K == g2). In order to simplify computation of the workgroup
    /// sizes, let's round grid dims to be a power of 2. The extra positions in the grid (due to
    /// rounding) are to be skipped by the kernels.
    /// * Assumption: WAVESIZE is power of 2.
    ///
    /// The workgroup size does not have the restrictions imposed by synchronization between
    /// workitems because the kernel does not require synchronization.

#if WORKAROUND_ISSUE_MIFIN_80
    const uint32_t wavesize = 64;
    std::ignore             = context;
#else
    const auto wavesize = static_cast<uint32_t>(context.GetStream().GetWavefrontWidth());
    assert(IsPower2(wavesize));
#endif

    const auto is2d_kernel = (top_d == 1); // For 2D + optimize for 3D where the 1st dim is 1.
    const auto g0          = RoundUpNearestPower2Positive(all_n);
    const auto g1          = RoundUpNearestPower2Positive(all_c);
    const auto g2          = RoundUpNearestPower2Positive(is2d_kernel ? top_h : top_d);

    auto work_left = wavesize / 1;
    const auto w0  = (g0 < work_left) ? g0 : work_left;
    work_left /= w0;
    const auto w1 = (g1 < work_left) ? g1 : work_left;
    work_left /= w1;
    const auto w2 = (g2 < work_left) ? g2 : work_left;

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPoolingForwardNaive.cl";
        kernel.kernel_name = "mloPoolingForwardNaive";

        auto build_params = KernelBuildParameters{
            {"MLO_POOLING_OP_ID", pooling_method}, // We need this at compile time in order to
                                                   // engage mixed precision only when necessary.
            {"MLO_POOLING_INDEX_TYPE", get_pooling_index_type_name(index_type)},
            {"MLO_POOLING_IS2D_KERNEL", static_cast<int>(is2d_kernel)},
        };
        build_params << GetDataTypeKBP(bot.GetType());
        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        // [Informative] The total number of kernels required to cover the whole
        // forward pooling problem space is 3*4*2*2 = 48. The solver is dynamic.
        // * 3: the number of supported operations
        // * 4: the number of supported index types
        // * 2: the number of supported data types
        // * 2: 2D and 3D kernels (optimization)

        kernel.g_wk.push_back(g0);
        kernel.g_wk.push_back(g1);
        kernel.g_wk.push_back(g2);
        kernel.l_wk.push_back(w0);
        kernel.l_wk.push_back(w1);
        kernel.l_wk.push_back(w2);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::FwdInvokeParams>();

            kernel(params.x,
                   params.y,
                   params.workspace,
                   save_index,
                   index_mode,
                   filter_d,
                   filter_h,
                   filter_w,
                   filter_d_stride,
                   filter_h_stride,
                   filter_w_stride,
                   filter_d_pad,
                   filter_h_pad,
                   filter_w_pad,
                   all_n,
                   all_c,
                   bot_d,
                   bot_h,
                   bot_w,
                   bot_n_stride,
                   bot_c_stride,
                   bot_d_stride,
                   bot_h_stride,
                   bot_w_stride,
                   top_d,
                   top_h,
                   top_w,
                   top_n_stride,
                   top_c_stride,
                   top_d_stride,
                   top_h_stride,
                   top_w_stride,
                   mask_n_stride,
                   mask_c_stride,
                   mask_d_stride,
                   mask_h_stride,
                   mask_w_stride);
        };
    };
    return result;
}

std::size_t
PoolingForwardNaive::GetWorkspaceSize(const ExecutionContext&,
                                      const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax || !problem.SaveIndex())
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen
