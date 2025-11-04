/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "tensor_op_helpers.hpp"
#include <miopen/tensorOp/solvers.hpp>
#include <miopen/tensorOp/invoke_params.hpp>
#include <miopen/tensor.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/datatype.hpp>

namespace miopen {

namespace solver {

namespace tensorOp {

bool Op5dTensorGeneric::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                     const miopen::tensorOp::ProblemDescription& problem) const
{
    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& alens       = aTensorDesc.GetLengths();
    auto asize              = alens.size();

    if(asize == 5)
    {
        return true;
    }

    return false;
}

std::size_t Op5dTensorGeneric::GetWorkspaceSize(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::tensorOp::ProblemDescription& problem) const
{
    return 0;
}

ConvSolution
Op5dTensorGeneric::GetSolution([[maybe_unused]] const ExecutionContext& context,
                               const miopen::tensorOp::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const auto& aTensorDesc = problem.GetATensorDesc();
    const auto& bTensorDesc = problem.GetBTensorDesc();
    const auto& cTensorDesc = problem.GetCTensorDesc();

    const auto& alens = aTensorDesc.GetLengths();
    const auto& blens = bTensorDesc.GetLengths();
    const auto& clens = cTensorDesc.GetLengths();

    std::array<size_t, 5> astrides;
    std::array<size_t, 5> bstrides;
    std::array<size_t, 5> cstrides;
    std::tie(astrides[0], astrides[1], astrides[2], astrides[3], astrides[4]) =
        miopen::tien<5>(aTensorDesc.GetStrides());
    std::tie(bstrides[0], bstrides[1], bstrides[2], bstrides[3], bstrides[4]) =
        miopen::tien<5>(bTensorDesc.GetStrides());
    std::tie(cstrides[0], cstrides[1], cstrides[2], cstrides[3], cstrides[4]) =
        miopen::tien<5>(cTensorDesc.GetStrides());

    auto bstrides_fix = bstrides;
    auto astrides_fix = astrides;
    for(int i = 0; i < 5; ++i){
        if(blens[i]==1) bstrides_fix[i] = 0;
        if(alens[i] == 1) astrides_fix[i] = 0;
    }

    KernelBuildParameters build_params = KernelBuildParameters{};
    GetCommonParams(build_params, problem, false);
    build_params.Define("USE_5D_TENSOR_GENERIC");

    // PACK_T - how many consecutive elements each thread handles per inner iteration.
    // declaration + define as main source, in kernel defined as fallback - only
    constexpr size_t k_PACK_T = 8;
    build_params.Define("PACK_T", k_PACK_T);

    // helper lambda for max offset calc - needed to ensure
    // that our adressing would fit in 32-bit integers or we
    // have to use 64-bit ones
    auto max_off = [&](const auto& lens, const auto& strides) {
        unsigned long long m = 0;
        for(int i=0;i<5;++i)
            if(lens[i] > 0) m += (unsigned long long)(lens[i]-1) * (unsigned long long)strides[i];
        return m;
    };

    const unsigned long long a_max = max_off(alens, astrides_fix);
    const unsigned long long b_max = max_off(blens, bstrides_fix);
    const unsigned long long c_max = max_off(clens, cstrides);

    const auto fits_u32 = [](auto x) { return x <= 0xFFFFFFFFull; };

    auto all_fit_u32 = [&](const auto& lens, const auto& strides) {
        for(int i=0;i<5;++i)
            if(!(fits_u32(lens[i]) && fits_u32(strides[i])))
                return false;
        return true;
    };

    bool use_i32 = (a_max < 0x7fffffffULL) && (b_max < 0x7fffffffULL) && (c_max < 0x7fffffffULL);
    use_i32 = use_i32 && all_fit_u32(alens, astrides_fix) && all_fit_u32(blens, bstrides_fix) && all_fit_u32(clens, cstrides);

    // check if total work fits in u32 or not
    auto mul_u64_sat = [](uint64_t a, uint64_t b) -> uint64_t {
        if(a == 0 || b == 0) return uint64_t{0};
        if(a > (std::numeric_limits<uint64_t>::max() / b)) return std::numeric_limits<uint64_t>::max();
        return a * b;
    };

    uint64_t total_work_u64 = 1;
    for(int i = 0; i < 5; ++i) total_work_u64 = mul_u64_sat(total_work_u64, static_cast<uint64_t>(clens[i]));
    const bool total_work_fits_u32 = (total_work_u64 <= uint64_t{0xFFFFFFFFu});

    use_i32 = use_i32 && total_work_fits_u32;
    if(use_i32) build_params.Define("USE_INDEX32");

    auto kernel = KernelInfo{};
    miopenDataType_t data_type = bTensorDesc.GetType();

    // size of warp (simultaneously executed threads) + total work
    constexpr size_t warp_size = 64;
    const size_t total_elems   = clens[0] * clens[1] * clens[2] * clens[3] * clens[4];
    const size_t total_packets = (total_elems + k_PACK_T - 1) / k_PACK_T;

    // use max compute units for task
    const size_t cu_count = context.GetStream().GetMaxComputeUnits();

    // Heuristic cap for the number of work-groups to launch, proportional to the number of
    // compute units (CU). The goal is to keep enough warps resident to hide memory latency,
    // while avoiding a cascade of tiny WGs on small problems (scheduler overhead & tail waste).
    //
    // On AMD, a wavefront is 64 threads. With local_threads = 256, each WG has 256/64 = 4 wavefronts.
    // The multipliers below roughly target a desired number of wavefronts per CU:
    //
    // cu_count * 32  -> approx 32 WGs/CU × 4 waves/WG approx 128 waves/CU   (for larger problems)
    // cu_count * 16  -> approx 16 WGs/CU × 4 waves/WG approx  64 waves/CU   (for mid-size problems)
    // cu_count *  8  -> aprox WGs/CU × 4 waves/WG approx  32 waves/CU   (for small problems)
    //
    // We downscale the cap when total_packets is small to reduce launch/dispatch overhead and avoid
    // over-fragmentation, yet still keep enough waves per CU to hide latency. For the special case
    // b_w == 1 (W-broadcast on B), memory access patterns tend to be less favorable; we slightly raise
    // the cap to allow more WGs in flight and better latency hiding.
    size_t max_num_wg_hw = cu_count * 32;
    if(total_packets < 512)  max_num_wg_hw = cu_count * 16;
    if(total_packets < 128)  max_num_wg_hw = cu_count * 8;

    size_t local_threads = warp_size * 4;

    // Small adjustments for narrow problems and W-broadcast cases:
    // For small total packet counts - reduce local_threads to avoid spawning many half-empty waves
    // For b_w == 1 - i.e. "heavy broadcast", limit local_threads to 128 to allow more WGs per CU.
    const bool bw_is_one = (blens[4] == 1);

    if(total_packets < 512)      local_threads = 128;
    if(total_packets < 128)      local_threads = 64;
    if(bw_is_one && local_threads > 128) local_threads = 128;

    // For W-broadcast, slightly increase the upper WG cap to better hide memory latency.
    if(bw_is_one) max_num_wg_hw = std::max(max_num_wg_hw, cu_count * 48);

    // Required number of WGs given the chosen local_threads
    const size_t need_wg = (total_packets + local_threads - 1) / local_threads;

    // Software-level safety cap on top of the hardware heuristic limit
    const size_t max_num_wg_sw = (total_elems < 1048576) ? 1024 : 4096;

    // Final cap
    const size_t max_num_wg = std::min(max_num_wg_hw, max_num_wg_sw);

    /// Final number of WGs to launch
    const size_t num_wg = std::clamp(need_wg, static_cast<size_t>(1), max_num_wg);

    const size_t global_threads = num_wg * local_threads;
    const std::array<size_t, 3> vld{local_threads, 1, 1};
    const std::array<size_t, 3> vgd{global_threads, 1, 1};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    kernel.kernel_file  = "MIOpenTensorKernelsHip.cpp";

    const bool is_contig = [&]{
        uint64_t exp = 1;
        for(int i = 4; i >= 0; --i) {
            if(static_cast<uint64_t>(astrides[i]) != exp ||
               static_cast<uint64_t>(bstrides[i]) != exp ||
               static_cast<uint64_t>(cstrides[i]) != exp)
                return false;
            exp *= static_cast<uint64_t>(clens[i]);
        }
        return true;
    }();

    const bool shapes_equal = (alens == blens) && (blens == clens);
    const bool use_fast = shapes_equal && is_contig;

    // kernel selector
    if(use_fast)
        kernel.kernel_name = "Op5dTensorGenericContiguous";
    else
        kernel.kernel_name = "Op5dTensorGeneric";

    using std::begin, std::end;
    kernel.l_wk.insert(end(kernel.l_wk), begin(vld), end(vld));
    kernel.g_wk.insert(end(kernel.g_wk), begin(vgd), end(vgd));

    result.invoker_factory =
        [data_type, blens, clens, cstrides, astrides_fix, bstrides_fix, kernel_name = kernel.kernel_name, mul_u64_sat](
            const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
                auto kernel = handle.Run(kernels.front());
                auto params = raw_params.CastTo<miopen::tensorOp::InvokeParams>();

                visit_float(data_type, [&](auto as_float) {
                    const float alpha0f = *static_cast<const float*>(params.alpha0);
                    const float alpha1f = *static_cast<const float*>(params.alpha1);
                    const float betaf   = *static_cast<const float*>(params.beta);
                    const auto alpha0   = as_float(alpha0f);
                    const auto alpha1   = as_float(alpha1f);
                    const auto beta     = as_float(betaf);

                    uint64_t total_work = 1;
                    for (int i = 0; i < 5; ++i) total_work = mul_u64_sat(total_work, static_cast<uint64_t>(clens[i]));

                    if(kernel_name == "Op5dTensorGenericContiguous")
                    {
                        kernel(params.ATensor,
                               params.BTensor,
                               params.CTensor,
                               static_cast<uint64_t>(params.Aoffset),
                               static_cast<uint64_t>(params.Boffset),
                               static_cast<uint64_t>(params.Coffset),
                               uint64_t(blens[0]),
                               uint64_t(blens[1]),
                               uint64_t(blens[2]),
                               uint64_t(blens[3]),
                               uint64_t(blens[4]),
                               uint64_t(clens[0]),
                               uint64_t(clens[1]),
                               uint64_t(clens[2]),
                               uint64_t(clens[3]),
                               uint64_t(clens[4]),
                               alpha0,
                               alpha1,
                               beta,
                               total_work,
                               !float_equal(beta, 0.0f));
                    }
                    else // Op5dTensorGeneric - case
                    {
                        kernel(params.ATensor,
                               params.BTensor,
                               params.CTensor,
                               uint64_t(params.Aoffset),
                               uint64_t(params.Boffset),
                               uint64_t(params.Coffset),
                               uint64_t(blens[0]),
                               uint64_t(blens[1]),
                               uint64_t(blens[2]),
                               uint64_t(blens[3]),
                               uint64_t(blens[4]),
                               uint64_t(clens[0]),
                               uint64_t(clens[1]),
                               uint64_t(clens[2]),
                               uint64_t(clens[3]),
                               uint64_t(clens[4]),
                               uint64_t(astrides_fix[0]),
                               uint64_t(astrides_fix[1]),
                               uint64_t(astrides_fix[2]),
                               uint64_t(astrides_fix[3]),
                               uint64_t(astrides_fix[4]),
                               uint64_t(bstrides_fix[0]),
                               uint64_t(bstrides_fix[1]),
                               uint64_t(bstrides_fix[2]),
                               uint64_t(bstrides_fix[3]),
                               uint64_t(bstrides_fix[4]),
                               uint64_t(cstrides[0]),
                               uint64_t(cstrides[1]),
                               uint64_t(cstrides[2]),
                               uint64_t(cstrides[3]),
                               uint64_t(cstrides[4]),
                               alpha0,
                               alpha1,
                               beta,
                               total_work,
                               !float_equal(beta, 0.0f));
                    }
                });
            };
        };
    result.construction_params.push_back(kernel);
    return result;
}

} // namespace tensorOp

} // namespace solver

} // namespace miopen
