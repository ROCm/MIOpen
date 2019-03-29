/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <miopen/config.h>
#include <miopen/convolution.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/db.hpp>
#include <miopen/env.hpp>
#include <miopen/find_db.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/solver.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/tensor.hpp>
#include <miopen/util.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/algorithm.hpp>

#if MIOPEN_USE_GEMM
#include <miopen/gemm_v2.hpp>
#endif

#include <cassert>
#include <type_traits>

#include <boost/range/adaptors.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING)

struct AutoEnableProfiling
{
    AutoEnableProfiling(Handle& x) : h(x)
    {
        prev_state = h.IsProfilingEnabled();
        h.EnableProfiling();
    }

    ~AutoEnableProfiling()
    {
        h.EnableProfiling(prev_state);
        h.ResetKernelTime();
    }

    private:
    Handle& h;
    bool prev_state;
};

static inline void AddKernels(Handle& handle,
                              const std::string& algorithm_name,
                              const std::string& network_config,
                              const miopen::solver::ConvSolution& s,
                              std::vector<KernelInvoke>* const kernels)
{
    if(!algorithm_name.empty() && !network_config.empty())
    {
        handle.ClearKernels(algorithm_name, network_config);
    }
    else
    {
        assert(algorithm_name.empty() && network_config.empty());
    }
    int i = 0;
    for(auto& k : s.construction_params)
    {
        MIOPEN_LOG_I2(k.kernel_name);
        auto kernel = handle.AddKernel(algorithm_name,
                                       network_config,
                                       k.kernel_file,
                                       k.kernel_name,
                                       k.l_wk,
                                       k.g_wk,
                                       k.comp_options,
                                       i);
        if(kernels != nullptr)
        {
            kernels->push_back(kernel);
        }
        ++i;
    }
}

static inline void ValidateGroupCount(const TensorDescriptor& xDesc,
                                      const TensorDescriptor& wDesc,
                                      const ConvolutionDescriptor& conv)
{
    if(conv.group_count == 1)
    {
        if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1])
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");
    }
    if(conv.group_count > 1)
    {
        if(xDesc.GetLengths()[1] % conv.group_count != 0 ||
           wDesc.GetLengths()[0] % conv.group_count != 0 ||
           conv.group_count > xDesc.GetLengths()[1] || conv.group_count > wDesc.GetLengths()[0] ||
           conv.group_count < 1)
            MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
        if(xDesc.GetLengths()[1] / conv.group_count != wDesc.GetLengths()[1])
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");
    }
}

template <typename T>
inline int EvaluateDataDirectSolution(Handle& handle,
                                      const miopen::solver::ConvSolution& solution,
                                      const ExtraKernelArgs& extraArgs,
                                      ConstData_t in, // Fwd: x, Bwd: dy
                                      ConstData_t weights,
                                      Data_t out, // Fwd: y, Bwd: dx
                                      const TensorDescriptor& outDesc,
                                      Data_t workSpace,
                                      const size_t workSpaceSize,
                                      T padding_val,
                                      float& elapsed)
{
    // Fail if required workspace is not provided.
    if(solution.workspce_sz != 0)
    {
        if(workSpace == nullptr || workSpaceSize < solution.workspce_sz)
            return -1;
    }
    std::vector<KernelInvoke> kernels;
    AddKernels(handle, "", "", solution, &kernels);
    if(kernels.size() > 2)
        return -2;

    bool with_subsample = false;
    bool with_upsample  = false;
    for(auto& k : kernels)
    {
        if(k.GetName() == "SubSample")
            with_subsample = true;
        else if(k.GetName() == "UpSample")
            with_upsample = true;
    }
    assert(!(with_subsample && with_upsample));
    if(with_subsample && with_upsample)
        return -3;

    // Note implicit conversion: Data_t to ConstData_t (workSpace).
    ConstData_t conv_in = with_subsample ? workSpace : in;
    Data_t conv_out     = with_upsample ? workSpace : out;

    elapsed = 0.0f;
    for(auto& k : kernels)
    {

        if(k.GetName() == "SubSample")
        {
            k(in, workSpace);
        }
        else if(k.GetName() == "UpSample")
        {
            {
                /// \todo Initialization is required for upsampling. This leads to small perf drop.
                /// 1: Add kernel (from SetTensor) to the Solution in the Solver.
                /// 2: Fix UpSample kernel, probably by means of conditional compilation.
                float zero = 0.f;
                SetTensor(handle, outDesc, out, &zero);
                elapsed += handle.GetKernelTime();
            }
            k(workSpace, out);
        }
        else if(k.GetName() == "gcnAsmConv1x1U")
        {
            int unused       = 0;
            int* return_addr = nullptr;
            int N, C, H, W, K, n_groups, out_H, out_W;
            std::tie(N, C, H, W, K, n_groups, out_H, out_W) = extraArgs;
            int conv_H = (with_subsample ? out_H : H); // Trick; see respective Solver.
            int conv_W = (with_subsample ? out_W : W);
            k(N,
              C,
              conv_H,
              conv_W,
              K,
              n_groups,
              unused,
              unused,
              conv_in,
              weights,
              conv_out,
              return_addr);
        }
        else
        {
            k(conv_in, weights, conv_out, padding_val);
        }
        elapsed += handle.GetKernelTime();
    }
    return 0;
}

template <typename T>
int ConvolutionDescriptor::FindWinogradKernel(Handle& handle,
                                              const TensorDescriptor& xDesc,
                                              const TensorDescriptor& wDesc,
                                              const TensorDescriptor& yDesc,
                                              WinogradKernelParams& k_p,
                                              KernelInvoke& kernel,
                                              std::string& solver_id,
                                              int direction,
                                              std::string* kcache_key) const
{
    assert((std::is_same<T, mlo_construct_winograd>::value ||
            std::is_same<T, mlo_construct_winograd_wrw>::value));
    try
    {
        T construct_params(xDesc, wDesc, yDesc, *this, direction);
        construct_params.setStream(&handle);

        const auto solution = FindFirstSolution(construct_params);
        if(!solution.Succeeded())
            return -1;
        const auto& kernels_info = solution.construction_params;
        const auto& k_info       = kernels_info[0];

        solver_id = solution.solver_id;
        std::string network_config;
        construct_params.mloBuildConf_Key(network_config);

        if(kcache_key != nullptr)
            *kcache_key = network_config;

        const bool is_wrw = std::is_same<T, mlo_construct_winograd_wrw>::value;

        const std::string algorithm = is_wrw ? "miopenConvolutionBwdWeightsAlgoWinograd"
                                             : (direction == 1)
                                                   ? "miopenConvolutionFwdAlgoWinograd"
                                                   : "miopenConvolutionBwdDataAlgoWinograd";
        handle.ClearKernels(algorithm, network_config);
        kernel = handle.AddKernel(algorithm,
                                  network_config,
                                  k_info.kernel_file,
                                  k_info.kernel_name,
                                  k_info.l_wk,
                                  k_info.g_wk,
                                  k_info.comp_options);
        int N, C, H, W, K, n_groups, out_H, out_W, R, S, pad_H, pad_W;
        construct_params.getCompiledInParameters(
            &N, &C, &H, &W, &K, &n_groups, &out_H, &out_W, &R, &S, &pad_H, &pad_W);
        k_p = std::make_tuple(N,
                              C,
                              H,
                              W,
                              K,
                              n_groups,
                              out_H,
                              out_W,
                              R,
                              S,
                              pad_H,
                              pad_W,
                              k_info.kernel_name == "sp3AsmConvRxSU");
        return 0;
    }
    catch(miopen::Exception&)
    {
        return -1;
    }
}

std::vector<miopen::solver::ConvSolution>
ConvolutionDescriptor::FindDataDirectSolutions(Handle& handle,
                                               const TensorDescriptor& xDesc,
                                               const TensorDescriptor& wDesc,
                                               const TensorDescriptor& yDesc,
                                               bool exhaustiveSearch,
                                               bool isForward,
                                               std::string& network_config,
                                               ExtraKernelArgs& extraArgs) const
{

    if(GetSpatialDimension() != 2 || miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
        return {};

    mlo_construct_direct2D construct_params(xDesc, wDesc, yDesc, *this, isForward ? 1 : 0);
    construct_params.setDoSearch(exhaustiveSearch);
    construct_params.saveSearchRequest(true);
    construct_params.setGeneralCompOptions("");
    construct_params.setStream(&handle);
    construct_params.detectRocm();

    if(IsWinograd3x3Supported(handle, isForward, wDesc, (isForward ? xDesc : yDesc)) &&
       construct_params.mloIsFastBinaryWinograd3x3U() && construct_params.usesBinaryKernel())
        return {};

    try
    {
        int N, C, H, W, K, n_groups, out_H, out_W;
        construct_params.getCompiledInParameters(&N, &C, &H, &W, &K, &n_groups, &out_H, &out_W);
        extraArgs = std::make_tuple(N, C, H, W, K, n_groups, out_H, out_W);
        construct_params.mloBuildConf_Key(network_config);
        return FindAllSolutions(construct_params);
    }
    catch(miopen::Exception&)
    {
        return {};
    }
}

static void DirConvFindCore(Handle& handle,
                            const TensorDescriptor& xDesc,
                            ConstData_t x,
                            const TensorDescriptor& wDesc,
                            ConstData_t w,
                            const TensorDescriptor& yDesc,
                            Data_t workSpace,
                            size_t workSpaceSize,
                            const ConvolutionDescriptor& conv,
                            bool exhaustiveSearch,
                            DbRecord& record)
{
    AutoEnableProfiling enableProfiling{handle};

    // create a dummy buffer for use as output for the kernel calls
    // because kernels are called purely for timing purposes
    auto tmp_y = handle.Create(yDesc.GetElementSize() * GetTypeSize(yDesc.GetType()));

    {
        ValidateGroupCount(xDesc, wDesc, conv);

#if MIOPEN_USE_GEMM
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
        { // GEMM algo
            std::size_t in_n, in_c;
            std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

            std::size_t wei_k = wDesc.GetLengths()[0];

            std::size_t spatial_dim = conv.GetSpatialDimension();

            auto in_spatial  = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
            auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
            auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);

            float time_gemm           = 0;
            const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));
            // Use transpose path if input ht and width <= 14 for 1x1_stride=1 convolutions OR
            // for 1x1_stride=2
            if(conv.GetSpatialDimension() == 2 &&
               (miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
                miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; })) &&
               ((miopen::all_of(in_spatial, [](auto v) { return v <= 14; }) &&
                 miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; })) ||
                miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; })))
            {
                size_t workspace_req = conv.ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc);
                if(workSpace != nullptr && workSpaceSize >= workspace_req)
                {
                    if(conv.group_count > 1)
                    {
                        MIOPEN_LOG_FUNCTION("groupconv, 1x1, h14xw14 || u2xv2");
                    }
                    else
                    {
                        MIOPEN_LOG_FUNCTION("convolution, 1x1, h14xw14 || u2xv2");
                    }

                    // y = CNHW2NCHW(w * NCHW2CNHW(x))
                    transpose_NCHW2CNHW(handle,
                                        in_n,
                                        in_c,
                                        in_spatial[0],
                                        in_spatial[1],
                                        out_spatial[0],
                                        out_spatial[1],
                                        x,
                                        workSpace,
                                        0,
                                        0,
                                        conv.GetConvStrides()[0],
                                        conv.GetConvStrides()[1],
                                        xDesc.GetType());
                    time_gemm = handle.GetKernelTime();

                    std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                                   out_spatial.end(),
                                                                   std::size_t(1),
                                                                   std::multiplies<std::size_t>());

                    std::size_t x_t_size = in_n * in_c * out_spatial_size;

                    std::size_t wksp_offset = 0;
                    if(wDesc.GetType() == miopenInt8)
                    {
                        wksp_offset = x_t_size;
                        transpose_packed_MN2NM(handle,
                                               in_c,
                                               static_cast<int>(in_n * out_spatial_size),
                                               0,
                                               wksp_offset,
                                               workSpace,
                                               workSpace,
                                               xDesc.GetType());

                        time_gemm += handle.GetKernelTime();

                        x_t_size *= 2;
                    }
                    if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
                       (yDesc.GetType() == miopenInt32 || yDesc.GetType() == miopenFloat))
                        x_t_size /= 4;

                    std::string kcache_key;

                    GemmDescriptor gemm_desc =
                        conv.group_count > 1 ? CreateGemmDescriptorGroupConvCNHWFwd(
                                                   wDesc, xDesc, yDesc, conv.group_count)
                                             : CreateGemmDescriptorConvCNHWFwd(wDesc, xDesc, yDesc);

                    miopenStatus_t gemm_status = CallGemmTimeMeasure(
                        handle,
                        gemm_desc,
                        w,
                        0,
                        workSpace,
                        wksp_offset,
                        workSpace,
                        x_t_size,
                        &kcache_key,
                        time_precision,
                        conv.group_count > 1 ? callGemmStridedBatched : callGemm);

                    time_gemm += handle.GetKernelTime();

                    transpose_CNHW2NCHW(handle,
                                        in_n,
                                        wei_k,
                                        out_spatial[0],
                                        out_spatial[1],
                                        out_spatial[0],
                                        out_spatial[1],
                                        workSpace,
                                        tmp_y.get(),
                                        x_t_size,
                                        0,
                                        1,
                                        1,
                                        yDesc.GetType());
                    time_gemm += handle.GetKernelTime();

                    if(wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4)
                    {
                        TensorDescriptor ygemmDesc(
                            miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                        CastTensor(handle,
                                   &conv.lowp_quant,
                                   ygemmDesc,
                                   tmp_y.get(),
                                   yDesc,
                                   tmp_y.get(),
                                   0,
                                   0);
                        time_gemm += handle.GetKernelTime();
                    }

                    if(gemm_status == miopenStatusSuccess)
                        record.SetValues("miopenConvolutionFwdAlgoGEMM",
                                         FindDbData{"gemm",
                                                    time_gemm,
                                                    workspace_req,
                                                    kcache_key}); // Todo: gemm solver id?
                }
            }
            // 1x1_stride=1 with GEMM and zero workspace
            else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
                    miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
                    miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }))
            {
                if(conv.group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1");
                }

                // y = w * x
                std::string kcache_key;
                miopenStatus_t gemm_status = miopenStatusNotInitialized;

                if(wDesc.GetType() == miopenInt8)
                {
                    GemmDescriptor gemm_desc = CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

                    std::size_t out_offset      = 0;
                    std::size_t in_offset       = 0;
                    std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                                  in_spatial.end(),
                                                                  std::size_t(1),
                                                                  std::multiplies<std::size_t>());
                    transpose_packed_MN2NM(
                        handle, in_c, in_spatial_size, in_offset, 0, x, workSpace, xDesc.GetType());

                    time_gemm += (in_n * handle.GetKernelTime());

                    gemm_status = CallGemmTimeMeasure(handle,
                                                      gemm_desc,
                                                      w,
                                                      0,
                                                      workSpace,
                                                      0,
                                                      tmp_y.get(),
                                                      out_offset,
                                                      &kcache_key,
                                                      time_precision,
                                                      callGemm);

                    time_gemm += (in_n * handle.GetKernelTime());
                }
                else
                {
                    GemmDescriptor gemm_desc =
                        conv.group_count > 1
                            ? CreateGemmDescriptorGroupConvFwd(
                                  wDesc, xDesc, yDesc, conv.group_count)
                            : CreateGemmStridedBatchedDescriptorConv1x1Fwd(wDesc, xDesc, yDesc);

                    gemm_status = CallGemmTimeMeasure(handle,
                                                      gemm_desc,
                                                      w,
                                                      0,
                                                      x,
                                                      0,
                                                      tmp_y.get(),
                                                      0,
                                                      &kcache_key,
                                                      time_precision,
                                                      callGemmStridedBatched);

                    time_gemm = handle.GetKernelTime();
                    if(conv.group_count > 1)
                        time_gemm *= in_n;
                }

                if(wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4)
                {
                    TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                    CastTensor(
                        handle, &conv.lowp_quant, ygemmDesc, tmp_y.get(), yDesc, tmp_y.get(), 0, 0);
                    time_gemm += handle.GetKernelTime();
                }

                if(gemm_status == miopenStatusSuccess)
                    record.SetValues(
                        "miopenConvolutionFwdAlgoGEMM",
                        FindDbData{"gemm", time_gemm, 0, kcache_key}); // Todo: gemm solver id?
            }
            // if not 1x1
            else if(workSpace != nullptr &&
                    workSpaceSize >=
                        (conv.ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc) * conv.group_count))
            {
                if(conv.group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, non 1x1");
                }

                // y = w * Im2Col(x)
                float time_im2col = 0;
                int in_offset     = 0;
                time_im2col       = Im2ColGPU(handle,
                                        conv.GetSpatialDimension(),
                                        x,
                                        in_offset,
                                        in_c,
                                        in_spatial,
                                        wei_spatial,
                                        out_spatial,
                                        conv.GetConvPads(),
                                        conv.GetConvStrides(),
                                        conv.GetConvDilations(),
                                        workSpace,
                                        xDesc.GetType());

                std::size_t wksp_offset = 0;
                if(wDesc.GetType() == miopenInt8)
                {
                    std::size_t wei_spatial_size = std::accumulate(wei_spatial.begin(),
                                                                   wei_spatial.end(),
                                                                   std::size_t(1),
                                                                   std::multiplies<std::size_t>());

                    std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                                   out_spatial.end(),
                                                                   std::size_t(1),
                                                                   std::multiplies<std::size_t>());

                    wksp_offset = in_c * wei_spatial_size * out_spatial_size;

                    transpose_packed_MN2NM(handle,
                                           static_cast<int>(in_c * wei_spatial_size),
                                           out_spatial_size,
                                           0,
                                           wksp_offset,
                                           workSpace,
                                           workSpace,
                                           xDesc.GetType());
                    time_gemm += (in_n * handle.GetKernelTime());
                }

                std::string kcache_key;

                GemmDescriptor gemm_desc =
                    conv.group_count > 1
                        ? CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, conv.group_count)
                        : CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

                miopenStatus_t gemm_status =
                    CallGemmTimeMeasure(handle,
                                        gemm_desc,
                                        w,
                                        0,
                                        workSpace,
                                        wksp_offset,
                                        tmp_y.get(),
                                        0,
                                        &kcache_key,
                                        time_precision,
                                        conv.group_count > 1 ? callGemmStridedBatched : callGemm,
                                        (conv.group_count > 1 || wDesc.GetType() == miopenInt8 ||
                                         wDesc.GetType() == miopenInt8x4)
                                            ? GemmBackend_t::rocblas
                                            : GemmBackend_t::miopengemm);

                time_gemm += (in_n * (time_im2col + handle.GetKernelTime()));

                if(wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4)
                {
                    TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                    CastTensor(
                        handle, &conv.lowp_quant, ygemmDesc, tmp_y.get(), yDesc, tmp_y.get(), 0, 0);
                    time_gemm += handle.GetKernelTime();
                }

                if(gemm_status == miopenStatusSuccess)
                    record.SetValues("miopenConvolutionFwdAlgoGEMM",
                                     FindDbData{"gemm",
                                                time_gemm,
                                                (conv.ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc) *
                                                 conv.group_count),
                                                kcache_key}); // Todo: gemm solver id?
            }
        }
#else
        (void)workSpace;     // Suppress warning
        (void)workSpaceSize; // Suppress warning
#endif

        if(conv.GetSpatialDimension() == 2)
        {
            // Winograd algo
            WinogradKernelParams k_p;
            KernelInvoke kernel_wino;
            std::string network_config;
            std::string solver_id;
            if(conv.FindWinogradKernel<mlo_construct_winograd>(
                   handle, xDesc, wDesc, yDesc, k_p, kernel_wino, solver_id, 1, &network_config) ==
               0)
            { // TODO: be more graceful
                // Execute the winograd kernel
                // Invocation of winograd does not depend on input bitness (FP32 or FP16)
                float time_wino  = 0;
                int flags        = 0;
                int reserved     = 0;
                int* return_addr = nullptr;
                bool isRxS;
                int N, C, H, W, K, n_groups, out_H, out_W, R, S, unused;
                std::tie(N, C, H, W, K, n_groups, out_H, out_W, R, S, unused, unused, isRxS) = k_p;
                // clang-format off
                MIOPEN_LOG_I2(" N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                    << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                    << " pad_h=" << conv.GetConvPads()[0] << " pad_w=" << conv.GetConvPads()[1] << " out_H=" << out_H << " out_W=" << out_W); // clang-format on
                if(isRxS)
                {
                    kernel_wino(N,
                                C,
                                H,
                                W,
                                K,
                                n_groups,
                                flags,
                                reserved,
                                x,
                                w,
                                tmp_y.get(),
                                return_addr,
                                R,
                                S,
                                conv.GetConvPads()[0],
                                conv.GetConvPads()[1],
                                out_H,
                                out_W);
                }
                else
                {
                    kernel_wino(
                        N, C, H, W, K, n_groups, flags, reserved, x, w, tmp_y.get(), return_addr);
                }
                time_wino = handle.GetKernelTime();
                record.SetValues("miopenConvolutionFwdAlgoWinograd",
                                 FindDbData{solver_id, time_wino, 0, network_config});
            }

            // Direct algo
            ExtraKernelArgs eka;
            const auto all = conv.FindDataDirectSolutions(
                handle, xDesc, wDesc, yDesc, exhaustiveSearch, true, network_config, eka);
            miopen::solver::ConvSolution selected{miopenStatusUnknownError};
            float best = std::numeric_limits<float>::max();
            visit_float(xDesc.GetType(), [&](auto as_float) {
                for(const auto& sol : all)
                {
                    float elapsed = 0.0f;
                    const int rc  = EvaluateDataDirectSolution(handle,
                                                              sol,
                                                              eka,
                                                              x,
                                                              w,
                                                              tmp_y.get(),
                                                              yDesc,
                                                              workSpace,
                                                              workSpaceSize,
                                                              as_float(0.0f),
                                                              elapsed);
                    if(rc != 0)
                    {
                        MIOPEN_LOG_E(sol << " returns " << rc);
                    }
                    else
                    {
                        MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ")
                                         << best);
                        if(elapsed < best)
                        {
                            best     = elapsed;
                            selected = sol;
                        }
                    }
                }
            });
            if(selected.Succeeded())
            {
                const std::string algorithm_name = "miopenConvolutionFwdAlgoDirect";
                AddKernels(handle, algorithm_name, network_config, selected, nullptr);
                MIOPEN_LOG_I("Selected: " << selected << ": " << best << ", workspce_sz = "
                                          << selected.workspce_sz);
                record.SetValues(
                    algorithm_name,
                    FindDbData{selected.solver_id, best, selected.workspce_sz, network_config});
            }
        }

        // FFT algo
        if(conv.GetSpatialDimension() == 2 &&
           miopen::all_of(conv.GetConvDilations(), [](auto v) { return v == 1; }) &&
           conv.group_count == 1 && wDesc.GetType() != miopenInt8 &&
           wDesc.GetType() != miopenInt8x4)
        {
            std::string network_config;
            std::vector<KernelInvoke> kernels_fft;
            size_t workspace_fft = conv.ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc);
            if(conv.FindFwdFFTKernel(
                   handle, xDesc, wDesc, yDesc, workspace_fft, kernels_fft, network_config) == 0)
            {
                (void)kernels_fft; // not used now, but needed as fft coverage widens
                if(workSpace != nullptr && workSpaceSize >= workspace_fft)
                {
                    float time_fft = conv.ExecuteFwdFFTKernel(handle,
                                                              xDesc,
                                                              x,
                                                              wDesc,
                                                              w,
                                                              yDesc,
                                                              tmp_y.get(),
                                                              workSpace,
                                                              workSpaceSize,
                                                              true);
                    record.SetValues("miopenConvolutionFwdAlgoFFT",
                                     FindDbData{"fft",
                                                time_fft,
                                                workspace_fft,
                                                network_config}); // Todo: fft solver id?
                }
            }
        }
    }
}

void ConvolutionDescriptor::FindConvFwdAlgorithm(Handle& handle,
                                                 const TensorDescriptor& xDesc,
                                                 ConstData_t x,
                                                 const TensorDescriptor& wDesc,
                                                 ConstData_t w,
                                                 const TensorDescriptor& yDesc,
                                                 ConstData_t y,
                                                 const int requestAlgoCount,
                                                 int* const returnedAlgoCount,
                                                 miopenConvAlgoPerf_t* perfResults,
                                                 Data_t workSpace,
                                                 size_t workSpaceSize,
                                                 bool exhaustiveSearch) const
{
    MIOPEN_LOG_I2("");
    if(x == nullptr || w == nullptr || y == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "Buffers cannot be NULL");
    if(returnedAlgoCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "returnedAlgoCount cannot be nullptr");
    if(perfResults == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "perfResults cannot be nullptr");
    if(requestAlgoCount < 1)
        MIOPEN_THROW(miopenStatusBadParm, "requestAlgoCount cannot be < 1");

    *returnedAlgoCount = 0;

    ProblemDescription problem(xDesc, wDesc, yDesc, *this, 1);

    std::vector<PerfField> perf_db = FindDb::TryLoad(handle, problem, [&](DbRecord& record) {
        DirConvFindCore(handle,
                        xDesc,
                        x,
                        wDesc,
                        w,
                        yDesc,
                        workSpace,
                        workSpaceSize,
                        *this,
                        exhaustiveSearch,
                        record);
    });

    if(perf_db.empty())
        MIOPEN_THROW("Fwd Convolution cannot be executed due to incorrect params");

    std::sort(begin(perf_db), end(perf_db));

    for(const auto& entry : perf_db)
        MIOPEN_LOG_I(entry.name << "\t" << entry.time << "\t" << entry.workspace);

    *returnedAlgoCount = std::min(requestAlgoCount, static_cast<int>(perf_db.size()));

    for(int i = 0; i < *returnedAlgoCount; i++)
    {
        perfResults[i].fwd_algo =
            static_cast<miopenConvFwdAlgorithm_t>(FwdAlgoResolver(perf_db[i].name));
        perfResults[i].time   = perf_db[i].time;
        perfResults[i].memory = perf_db[i].workspace;
    }

    MIOPEN_LOG_I("FW Chosen Algorithm: " << perf_db[0].solver_id << " , " << perf_db[0].workspace
                                         << ", "
                                         << perf_db[0].time);
}

void ConvolutionDescriptor::ConvolutionForward(Handle& handle,
                                               const void* alpha,
                                               const TensorDescriptor& xDesc,
                                               ConstData_t x,
                                               const TensorDescriptor& wDesc,
                                               ConstData_t w,
                                               miopenConvFwdAlgorithm_t algo,
                                               const void* beta,
                                               const TensorDescriptor& yDesc,
                                               Data_t y,
                                               Data_t workSpace,
                                               size_t workSpaceSize) const
{
    MIOPEN_LOG_I2("algo = " << algo << ", workspace = " << workSpaceSize);
    if(x == nullptr || w == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != wDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if((xDesc.GetType() != yDesc.GetType() && xDesc.GetType() != miopenInt8 &&
        xDesc.GetType() != miopenInt8x4) ||
       xDesc.GetType() != wDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(algo != miopenConvolutionFwdAlgoGEMM &&
       (xDesc.GetType() == miopenInt8 || xDesc.GetType() == miopenInt8x4))
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    //    if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1]) {
    //        MIOPEN_THROW(miopenStatusBadParm);
    //    }
    if(xDesc.GetSize() < 3)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Only alpha=1 and beta=0 is supported");
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, wDesc, w);
    }

    {
        ValidateGroupCount(xDesc, wDesc, *this);

        switch(algo)
        {
        case miopenConvolutionFwdAlgoDirect:
        {
            // TODO(paul): Replicating code for now.
            mlo_construct_direct2D construct_params(xDesc, wDesc, yDesc, *this, 1); // forward
            construct_params.setStream(&handle);

            std::string network_config;
            construct_params.mloBuildConf_Key(network_config);

            auto&& kernels = handle.GetKernels("miopenConvolutionFwdAlgoDirect", network_config);
#if(!defined(__GNUC__) || defined(__clang__)) // w/a for segfault in gcc 5.4.0
            const
#endif
                auto num_kernels = kernels.size();
            if(kernels.empty())
                MIOPEN_THROW(
                    "Error running Direct Forward convolution. Was Find() executed previously?");

            auto kernel = kernels[0];

            visit_float(xDesc.GetType(), [&](auto as_float) {
                // Miminum checks. Only check what is required to select
                // proper invocation procedure & workspace sanity.
                float padding_val = 0;
                float elapsed     = 0;
                if((kernel.GetName() == "MIOpenCvFwd11x11") && num_kernels == 2)
                {
                    kernel(x, w, y, as_float(padding_val));
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();

                    kernels[1](x, w, y, as_float(padding_val));
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                else if(num_kernels == 2 && workSpace != nullptr && workSpaceSize != 0)
                {
                    assert(kernel.GetName() == "SubSample");
                    kernel(x, workSpace);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();

                    assert(kernels[1].GetName() == "gcnAsmConv1x1U");
                    int unused       = 0;
                    int* return_addr = nullptr;
                    int N, C, H, W, K, n_groups, out_H, out_W;
                    construct_params.getCompiledInParameters(
                        &N, &C, &H, &W, &K, &n_groups, &out_H, &out_W);
                    kernels[1](N,
                               C,
                               out_H,
                               out_W,
                               K,
                               n_groups,
                               unused,
                               unused,
                               workSpace,
                               w,
                               y,
                               return_addr);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                else if(num_kernels == 1)
                {
                    if(kernel.GetName() == "gcnAsmConv1x1U")
                    {
                        int unused       = 0;
                        int* return_addr = nullptr;
                        int N, C, H, W, K, n_groups;
                        construct_params.getCompiledInParameters(&N, &C, &H, &W, &K, &n_groups);
                        kernel(N, C, H, W, K, n_groups, unused, unused, x, w, y, return_addr);
                    }
                    else
                    {
                        kernel(x, w, y, as_float(padding_val));
                    }
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                }
                else
                {
                    MIOPEN_THROW("Error running Direct Forward convolution (none workspace?)");
                }
                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            });
        }
        break;

        case miopenConvolutionFwdAlgoWinograd:
        {
            if(group_count > 1)
                MIOPEN_THROW("Winograd is not supported for group conv");

            mlo_construct_winograd construct_params(xDesc, wDesc, yDesc, *this, 1); // forward
            construct_params.setStream(&handle);

            std::string network_config;
            construct_params.mloBuildConf_Key(network_config);

            std::string algorithm_name = "miopenConvolutionFwdAlgoWinograd";
            auto kernel                = handle.GetKernel(algorithm_name, network_config);

            int flags        = 0;
            int reserved     = 0;
            int* return_addr = nullptr;
            int N, C, H, W, K, n_groups, out_H, out_W, R, S, unused;
            construct_params.getCompiledInParameters(
                &N, &C, &H, &W, &K, &n_groups, &out_H, &out_W, &R, &S, &unused, &unused);
            // clang-format off
            MIOPEN_LOG_I2(" N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                << " pad_h=" << GetConvPads()[0] << " pad_w=" << GetConvPads()[1] << " out_H=" << out_H << " out_W=" << out_W); // clang-format on
            if(kernel.GetName() == "sp3AsmConvRxSU")
            {
                kernel(N,
                       C,
                       H,
                       W,
                       K,
                       n_groups,
                       flags,
                       reserved,
                       x,
                       w,
                       y,
                       return_addr,
                       R,
                       S,
                       GetConvPads()[0],
                       GetConvPads()[1],
                       out_H,
                       out_W);
            }
            else
            {
                kernel(N, C, H, W, K, n_groups, flags, reserved, x, w, y, return_addr);
            }
        }
        break;

        case miopenConvolutionFwdAlgoGEMM:
#if MIOPEN_USE_GEMM
        {
            if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
            {
                MIOPEN_THROW("GEMM convolution is disabled");
            }

            std::size_t in_n, in_c;
            std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

            std::size_t wei_k = wDesc.GetLengths()[0];

            std::size_t spatial_dim = GetSpatialDimension();

            auto in_spatial  = boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + spatial_dim);
            auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
            auto out_spatial = boost::adaptors::slice(yDesc.GetLengths(), 2, 2 + spatial_dim);

            // Use transpose path if input ht and width <= 14 for 1x1_stride=1 convolutions OR for
            // 1x1_stride=2
            if(GetSpatialDimension() == 2 &&
               (miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
                miopen::all_of(GetConvPads(), [](auto v) { return v == 0; })) &&
               ((miopen::all_of(in_spatial, [](auto v) { return v <= 14; }) &&
                 miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; })) ||
                miopen::all_of(GetConvStrides(), [](auto v) { return v == 2; })))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1, h14xw14 || u2xv2");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1, h14xw14 || u2xv2");
                }

                assert(workSpace != nullptr &&
                       workSpaceSize >= ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc));

                float t1 = 0;
                transpose_NCHW2CNHW(handle,
                                    in_n,
                                    in_c,
                                    in_spatial[0],
                                    in_spatial[1],
                                    out_spatial[0],
                                    out_spatial[1],
                                    x,
                                    workSpace,
                                    0,
                                    0,
                                    GetConvStrides()[0],
                                    GetConvStrides()[1],
                                    xDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 = handle.GetKernelTime();

                std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                               out_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>());

                std::size_t x_t_size = in_n * in_c * out_spatial_size;

                std::size_t wksp_offset = 0;
                if(wDesc.GetType() == miopenInt8)
                {
                    wksp_offset = x_t_size;

                    transpose_packed_MN2NM(handle,
                                           in_c,
                                           static_cast<int>(in_n * out_spatial_size),
                                           0,
                                           wksp_offset,
                                           workSpace,
                                           workSpace,
                                           xDesc.GetType());
                    if(handle.IsProfilingEnabled())
                        t1 += handle.GetKernelTime();

                    x_t_size *= 2;
                }
                if((wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4) &&
                   GetTypeSize(xDesc.GetType()) > 0 &&
                   GetTypeSize(yDesc.GetType()) >= GetTypeSize(xDesc.GetType()))
                    x_t_size /= GetTypeSize(yDesc.GetType()) / GetTypeSize(xDesc.GetType());

                if(group_count > 1)
                {
                    GemmDescriptor gemm_desc =
                        CreateGemmDescriptorGroupConvCNHWFwd(wDesc, xDesc, yDesc, group_count);

                    CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, workSpace, 0, workSpace, x_t_size, nullptr, false);
                }
                else
                {
                    // y = CNHW2NCHW(w * NCHW2CNHW(x))
                    GemmDescriptor gemm_desc = CreateGemmDescriptorConvCNHWFwd(wDesc, xDesc, yDesc);

                    // y = CNHW2NCHW(w * NCHW2CNHW(x))
                    CallGemm(handle,
                             gemm_desc,
                             w,
                             0,
                             workSpace,
                             wksp_offset,
                             workSpace,
                             x_t_size,
                             nullptr,
                             false);
                }
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                transpose_CNHW2NCHW(handle,
                                    in_n,
                                    wei_k,
                                    out_spatial[0],
                                    out_spatial[1],
                                    out_spatial[0],
                                    out_spatial[1],
                                    workSpace,
                                    y,
                                    x_t_size,
                                    0,
                                    1,
                                    1,
                                    yDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                if(wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4)
                {
                    TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                    CastTensor(handle, &lowp_quant, ygemmDesc, y, yDesc, y, 0, 0);
                    if(handle.IsProfilingEnabled())
                        t1 += handle.GetKernelTime();
                }

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(t1);
                }
            }
            else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
                    miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
                    miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; }))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1");

                    GemmDescriptor gemm_desc =
                        CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, group_count);
                    float time_0 = 0;

                    std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                                   out_spatial.end(),
                                                                   std::size_t(1),
                                                                   std::multiplies<std::size_t>());

                    std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                                  in_spatial.end(),
                                                                  std::size_t(1),
                                                                  std::multiplies<std::size_t>());

                    for(std::size_t i = 0; i < in_n; i++)
                    {
                        std::size_t out_offset = i * wei_k * out_spatial_size;

                        std::size_t in_offset = i * in_c * in_spatial_size;

                        CallGemmStridedBatched(
                            handle, gemm_desc, w, 0, x, in_offset, y, out_offset, nullptr, false);
                        if(handle.IsProfilingEnabled())
                        {
                            if(i == in_n - 1)
                                handle.AccumKernelTime(time_0);
                            time_0 += handle.GetKernelTime();
                        }
                    }
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1");
                    float time_0 = 0;
                    float t1     = 0;

                    if(wDesc.GetType() == miopenInt8)
                    {
                        GemmDescriptor gemm_desc = CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

                        std::size_t out_spatial_size =
                            std::accumulate(out_spatial.begin(),
                                            out_spatial.end(),
                                            std::size_t(1),
                                            std::multiplies<std::size_t>());

                        std::size_t in_spatial_size =
                            std::accumulate(in_spatial.begin(),
                                            in_spatial.end(),
                                            std::size_t(1),
                                            std::multiplies<std::size_t>());

                        for(std::size_t i = 0; i < in_n; i++)
                        {
                            std::size_t out_offset = i * wei_k * out_spatial_size;

                            std::size_t in_offset = i * in_c * in_spatial_size;

                            transpose_packed_MN2NM(handle,
                                                   in_c,
                                                   in_spatial_size,
                                                   in_offset,
                                                   0,
                                                   x,
                                                   workSpace,
                                                   xDesc.GetType());
                            if(handle.IsProfilingEnabled())
                                t1 += handle.GetKernelTime();

                            CallGemm(handle,
                                     gemm_desc,
                                     w,
                                     0,
                                     workSpace,
                                     0,
                                     y,
                                     out_offset,
                                     nullptr,
                                     false);
                            if(handle.IsProfilingEnabled())
                                time_0 += handle.GetKernelTime();
                        }
                    }
                    else
                    {
                        // y = w * x
                        GemmDescriptor gemm_desc =
                            CreateGemmStridedBatchedDescriptorConv1x1Fwd(wDesc, xDesc, yDesc);

                        // y = w * x
                        CallGemmStridedBatched(handle, gemm_desc, w, 0, x, 0, y, 0, nullptr, false);
                        if(handle.IsProfilingEnabled())
                            time_0 += handle.GetKernelTime();
                    }

                    if(wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4)
                    {
                        TensorDescriptor ygemmDesc(
                            miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                        CastTensor(handle, &lowp_quant, ygemmDesc, y, yDesc, y, 0, 0);
                        if(handle.IsProfilingEnabled())
                            handle.AccumKernelTime(t1 + time_0);
                    }
                }
            }
            // if not 1x1
            else
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, non 1x1");
                }
                assert(workSpace != nullptr &&
                       workSpaceSize >= (ForwardGetWorkSpaceSizeGEMM(wDesc, yDesc) * group_count));

                // y = w * Im2Col(x)
                GemmDescriptor gemm_desc{};
                if(group_count > 1)
                    gemm_desc = CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, group_count);
                else
                    gemm_desc = CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

                std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                               out_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>());

                std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                              in_spatial.end(),
                                                              std::size_t(1),
                                                              std::multiplies<std::size_t>());

                float time_0 = 0;
                float t1     = 0;
                for(std::size_t i = 0; i < in_n; i++)
                {
                    std::size_t out_offset = i * wei_k * out_spatial_size;

                    std::size_t in_offset = i * in_c * in_spatial_size;

                    Im2ColGPU(handle,
                              GetSpatialDimension(),
                              x,
                              in_offset,
                              in_c,
                              in_spatial,
                              wei_spatial,
                              out_spatial,
                              GetConvPads(),
                              GetConvStrides(),
                              GetConvDilations(),
                              workSpace,
                              xDesc.GetType());

                    if(handle.IsProfilingEnabled())
                        t1 = handle.GetKernelTime();

                    std::size_t wksp_offset = 0;
                    if(wDesc.GetType() == miopenInt8)
                    {
                        std::size_t wei_spatial_size =
                            std::accumulate(wei_spatial.begin(),
                                            wei_spatial.end(),
                                            std::size_t(1),
                                            std::multiplies<std::size_t>());

                        wksp_offset = in_c * wei_spatial_size * out_spatial_size;

                        transpose_packed_MN2NM(handle,
                                               static_cast<int>(in_c * wei_spatial_size),
                                               out_spatial_size,
                                               0,
                                               wksp_offset,
                                               workSpace,
                                               workSpace,
                                               xDesc.GetType());

                        if(handle.IsProfilingEnabled())
                            t1 += handle.GetKernelTime();
                    }

                    // y = w * Im2Col(x)
                    if(group_count > 1)
                        CallGemmStridedBatched(
                            handle, gemm_desc, w, 0, workSpace, 0, y, out_offset, nullptr, false);
                    else
                        CallGemm(handle,
                                 gemm_desc,
                                 w,
                                 0,
                                 workSpace,
                                 wksp_offset,
                                 y,
                                 out_offset,
                                 nullptr,
                                 false,
                                 (wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4)
                                     ? GemmBackend_t::rocblas
                                     : GemmBackend_t::miopengemm);

                    // Update times for both the kernels
                    if(handle.IsProfilingEnabled())
                    {
                        if(i == in_n - 1)
                        {
                            handle.AccumKernelTime(t1 + time_0);
                            time_0 = handle.GetKernelTime();
                        }
                        else
                        {
                            handle.AccumKernelTime(t1);
                            time_0 += handle.GetKernelTime();
                        }
                    }
                }

                if(wDesc.GetType() == miopenInt8 || wDesc.GetType() == miopenInt8x4)
                {
                    TensorDescriptor ygemmDesc(miopenInt32, yDesc.GetLengths(), yDesc.GetStrides());

                    CastTensor(handle, &lowp_quant, ygemmDesc, y, yDesc, y, 0, 0);
                    if(handle.IsProfilingEnabled())
                        handle.AccumKernelTime(time_0);
                }
            }
        }
        break;
#else
            MIOPEN_THROW("GEMM is not supported");
#endif

        case miopenConvolutionFwdAlgoFFT:
        {
            if(group_count > 1)
                MIOPEN_THROW("FFT is not supported for group conv");

            std::size_t workspace_fft = ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc);
            if(workSpace != nullptr && workSpaceSize >= workspace_fft)
            {
                bool timed  = handle.IsProfilingEnabled();
                float timev = ExecuteFwdFFTKernel(
                    handle, xDesc, x, wDesc, w, yDesc, y, workSpace, workSpaceSize, timed);
                // FIXME: Is workSpaceSize correct here? It seems that workspace_fft is.

                if(timed)
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(timev);
                }
            }
        }
        break;
        }
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
}

// FindBackwardDataAlgorithm()
//
void ConvolutionDescriptor::FindConvBwdDataAlgorithm(Handle& handle,
                                                     const TensorDescriptor& dyDesc,
                                                     ConstData_t dy,
                                                     const TensorDescriptor& wDesc,
                                                     ConstData_t w,
                                                     const TensorDescriptor& dxDesc,
                                                     ConstData_t dx,
                                                     const int requestAlgoCount,
                                                     int* const returnedAlgoCount,
                                                     miopenConvAlgoPerf_t* perfResults,
                                                     Data_t workSpace,
                                                     size_t workSpaceSize,
                                                     bool exhaustiveSearch) const
{
    MIOPEN_LOG_I2("");
    if(dx == nullptr || w == nullptr || dy == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "Buffers cannot be NULL");
    if(returnedAlgoCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "returnedAlgoCount cannot be nullptr");
    if(perfResults == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "perfResults cannot be nullptr");
    if(requestAlgoCount < 1)
        MIOPEN_THROW(miopenStatusBadParm, "requestAlgoCount cannot be < 1");
    if(wDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    *returnedAlgoCount = 0;

    // create a dummy buffer for use as output for the kernel calls
    // because kernels are called purely for timing purposes
    auto tmp_dx = handle.Create(dxDesc.GetElementSize() * GetTypeSize(dxDesc.GetType()));

    AutoEnableProfiling enableProfiling{handle};

    // < algorith_name, <time, workspace_size> >
    std::vector<PerfField> perf_db;

    std::string network_config;
    {
        if(GetSpatialDimension() == 2 && GetConvDilations()[0] == 1 && GetConvDilations()[1] == 1)
        {
            // Winograd algo
            WinogradKernelParams k_p;
            KernelInvoke kernel_wino;
            std::string solver;
            if(FindWinogradKernel<mlo_construct_winograd>(
                   handle, dxDesc, wDesc, dyDesc, k_p, kernel_wino, solver, 0) == 0)
            { // TODO: be more graceful
                float time_wino = 0;
                /// \todo Move Flags into Solution.
                /// Flags:
                ///  - Any combination of flags is allowed.
                ///  - The last two (F_FLIP_DATA_N_C, F_FLIP_OUT_N_K) are for RxS version only.
                ///
                /// Reverse indexing of r, r -> R-1-r if set.
                static const int F_REVERSE_R = 1 << 0;
                /// Reverse indexing of s, s -> S-1-s if set.
                static const int F_REVERSE_S = 1 << 1;
                /// The w ("filter_addr") to be interpreted as float F [C][K][3][3] instead of
                /// float F [K][C][3][3].
                static const int F_FLIP_K_C = 1 << 2;
                /// Causes the dy ("data_addr") to be interpreted as float D [C][N][H][W] with
                /// the following restrictions:
                ///  - Read several stacks, no restrictions when reading single C
                ///  - When reading 2x C, ((N * H * W) <= 2^28)
                /// instead of float D [N][C][H][W] with the following restrictions:
                ///  - Read several stacks, if (H * W) >= 128 not more than 2, distance at most
                ///  one
                ///    stack, else  (C * H * W) <= 2^23 and it can do 32 stacks, so
                ///    (C * H * W) <= 2^28.
                ///  - Reading 2x C at once not a problem if it can read one.
                // static const int F_FLIP_DATA_N_C = 1 << 3;
                /// Causes the dx ("output_addr") to be interpreted as
                /// float OUT[K][N][out_h][out_w] (no specific restrictions)
                /// instead of float OUT [N][K][out_h][out_w] with the
                /// following restrictions:
                ///  - (K * out_h * out_w) <= 2^28
                // static const int F_FLIP_OUT_N_K = 1 << 4;
                /// <End of Flags>
                // (void)F_FLIP_DATA_N_C;
                // (void)F_FLIP_OUT_N_K;
                int flags        = F_REVERSE_R + F_REVERSE_S + F_FLIP_K_C;
                int reserved     = 0;
                int* return_addr = nullptr;
                int N, C, H, W, K, n_groups, out_H, out_W, R, S, pad_H, pad_W;
                bool isRxS;
                std::tie(N, C, H, W, K, n_groups, out_H, out_W, R, S, pad_H, pad_W, isRxS) = k_p;
                // clang-format off
                MIOPEN_LOG_I2(" N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                    << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                    << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W); // clang-format on
                if(isRxS)
                {
                    kernel_wino(N,
                                C,
                                H,
                                W,
                                K,
                                n_groups,
                                flags,
                                reserved,
                                dy,
                                w,
                                tmp_dx.get(),
                                return_addr,
                                R,
                                S,
                                pad_H,
                                pad_W,
                                out_H,
                                out_W);
                }
                else
                {
                    kernel_wino(
                        N, C, H, W, K, n_groups, flags, reserved, dy, w, tmp_dx.get(), return_addr);
                }
                time_wino = handle.GetKernelTime();
                perf_db.push_back(
                    PerfField{"miopenConvolutionBwdDataAlgoWinograd", solver, time_wino, 0});
            }

            // Direct algo
            ExtraKernelArgs eka;
            const auto all = FindDataDirectSolutions(
                handle, dxDesc, wDesc, dyDesc, exhaustiveSearch, false, network_config, eka);
            miopen::solver::ConvSolution selected{miopenStatusUnknownError};
            float best = std::numeric_limits<float>::max();
            visit_float(dyDesc.GetType(), [&](auto as_float) {
                for(const auto& sol : all)
                {
                    float elapsed = 0.0f;
                    const int rc  = EvaluateDataDirectSolution(handle,
                                                              sol,
                                                              eka,
                                                              dy,
                                                              w,
                                                              tmp_dx.get(),
                                                              dxDesc,
                                                              workSpace,
                                                              workSpaceSize,
                                                              as_float(0.0f),
                                                              elapsed);
                    if(rc != 0)
                    {
                        MIOPEN_LOG_E(sol << " returns " << rc);
                    }
                    else
                    {
                        MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ")
                                         << best);
                        if(elapsed < best)
                        {
                            best     = elapsed;
                            selected = sol;
                        }
                    }
                }
            });
            if(selected.Succeeded())
            {
                const std::string algorithm_name = "miopenConvolutionBwdDataAlgoDirect";
                AddKernels(handle, algorithm_name, network_config, selected, nullptr);
                MIOPEN_LOG_I("Selected: " << selected << ": " << best << ", workspce_sz = "
                                          << selected.workspce_sz);
                perf_db.push_back(
                    PerfField{algorithm_name, selected.solver_id, best, selected.workspce_sz});
            }
        }
        if(GetSpatialDimension() == 2 && GetConvDilations()[0] == 1 && GetConvDilations()[1] == 1 &&
           group_count == 1)
        {
            // FFT algo
            std::vector<KernelInvoke> kernels_fft;
            size_t workspace_fft = BackwardGetWorkSpaceSizeFFT(wDesc, dyDesc, dxDesc);
            if(FindBwdFFTKernel(handle, dyDesc, wDesc, dxDesc, workspace_fft, kernels_fft) == 0)
            {
                (void)kernels_fft; // not used now, but needed as fft coverage widens
                if(workSpace != nullptr && workSpaceSize >= workspace_fft)
                {
                    float time_fft = ExecuteBwdFFTKernel(handle,
                                                         dyDesc,
                                                         dy,
                                                         wDesc,
                                                         w,
                                                         dxDesc,
                                                         tmp_dx.get(),
                                                         workSpace,
                                                         workSpaceSize,
                                                         true);
                    perf_db.push_back(PerfField{
                        "miopenConvolutionBwdDataAlgoFFT", "FFT", time_fft, workspace_fft});
                }
            }
        }

#if MIOPEN_USE_GEMM
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
        { // GEMM based
            ValidateGroupCount(dxDesc, wDesc, *this);

            const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

            std::size_t in_n, in_c;
            std::tie(in_n, in_c) = tie_pick<0, 1>()(dxDesc.GetLengths());

            std::size_t wei_k = wDesc.GetLengths()[0];

            std::size_t spatial_dim = GetSpatialDimension();

            auto in_spatial  = boost::adaptors::slice(dxDesc.GetLengths(), 2, 2 + spatial_dim);
            auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
            auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + spatial_dim);

            // 1x1 does not require col2im
            if(GetSpatialDimension() == 2 &&
               miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
               miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
               miopen::all_of(GetConvStrides(), [](auto v) { return v == 2; }) &&
               workSpace != nullptr &&
               workSpaceSize >= BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1 u2xv2");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1 u2xv2");
                }
                float time_gemm = 0;

                // Initialization required for upsampling in bwd direction
                float zero = 0.f;
                SetTensor(handle, dxDesc, tmp_dx.get(), &zero);
                time_gemm = handle.GetKernelTime();

                // dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
                transpose_NCHW2CNHW(handle,
                                    in_n,
                                    wei_k,
                                    out_spatial[0],
                                    out_spatial[1],
                                    out_spatial[0],
                                    out_spatial[1],
                                    dy,
                                    workSpace,
                                    0,
                                    0,
                                    1,
                                    1,
                                    dyDesc.GetType());
                time_gemm += handle.GetKernelTime();

                GemmDescriptor gemm_desc =
                    group_count > 1 ? CreateGemmDescriptorGroupConvCNHWBwdData(
                                          wDesc, dyDesc, dxDesc, group_count)
                                    : CreateGemmDescriptorConvCNHWBwdData(wDesc, dyDesc, dxDesc);

                miopenStatus_t gemm_status =
                    CallGemmTimeMeasure(handle,
                                        gemm_desc,
                                        w,
                                        0,
                                        workSpace,
                                        0,
                                        workSpace,
                                        dyDesc.GetElementSize(),
                                        nullptr,
                                        time_precision,
                                        group_count > 1 ? callGemmStridedBatched : callGemm);

                time_gemm += handle.GetKernelTime();

                transpose_CNHW2NCHW(handle,
                                    in_n,
                                    in_c,
                                    out_spatial[0],
                                    out_spatial[1],
                                    in_spatial[0],
                                    in_spatial[1],
                                    workSpace,
                                    tmp_dx.get(),
                                    dyDesc.GetElementSize(),
                                    0,
                                    GetConvStrides()[0],
                                    GetConvStrides()[1],
                                    dyDesc.GetType());
                time_gemm += handle.GetKernelTime();

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdDataAlgoGEMM",
                                  "GEMM",
                                  time_gemm,
                                  BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc)});
            }
            // 1x1_stride=1 convolutions use GEMM and zero workspace
            else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
                    miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
                    miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; }))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1");
                }
                // dx = transpose(w) * dy
                GemmDescriptor gemm_desc =
                    group_count > 1
                        ? CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count)
                        : CreateGemmStridedBatchedDescriptorConv1x1BwdData(wDesc, dyDesc, dxDesc);

                miopenStatus_t gemm_status = CallGemmTimeMeasure(handle,
                                                                 gemm_desc,
                                                                 w,
                                                                 0,
                                                                 dy,
                                                                 0,
                                                                 tmp_dx.get(),
                                                                 0,
                                                                 nullptr,
                                                                 time_precision,
                                                                 callGemmStridedBatched);

                float time_gemm = handle.GetKernelTime();
                if(group_count > 1)
                    time_gemm *= in_n;

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdDataAlgoGEMM", "GEMM", time_gemm, 0});
            }
            // if not 1x1
            else if(workSpace != nullptr &&
                    workSpaceSize >=
                        (BackwardDataGetWorkSpaceSizeGEMM(wDesc, dyDesc) * group_count))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, non 1x1");
                }
                float time_col2im = 0;
                int in_offset     = 0;

                // dx = transpose(w) * dy
                GemmDescriptor gemm_desc =
                    group_count > 1
                        ? CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count)
                        : CreateGemmDescriptorConvBwdData(wDesc, dyDesc, dxDesc);

                miopenStatus_t gemm_status = CallGemmTimeMeasure(
                    handle,
                    gemm_desc,
                    w,
                    0,
                    dy,
                    0,
                    workSpace,
                    0,
                    nullptr,
                    time_precision,
                    group_count > 1 ? callGemmStridedBatched : callGemm,
                    group_count > 1 ? GemmBackend_t::rocblas : GemmBackend_t::miopengemm);

                float time_gemm = in_n * handle.GetKernelTime();
                time_col2im     = Col2ImGPU(handle,
                                        GetSpatialDimension(),
                                        workSpace,
                                        out_spatial,
                                        wei_spatial,
                                        GetConvPads(),
                                        GetConvStrides(),
                                        GetConvDilations(),
                                        in_c,
                                        in_spatial,
                                        tmp_dx.get(),
                                        in_offset,
                                        dyDesc.GetType());

                time_gemm += in_n * time_col2im;

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdDataAlgoGEMM",
                                  "GEMM",
                                  time_gemm,
                                  (BackwardDataGetWorkSpaceSizeGEMM(wDesc, dyDesc) * group_count)});
            }
        }
#else
        (void)workSpace;     // Suppress warning
        (void)workSpaceSize; // Suppress warning
#endif
    }

    if(perf_db.empty())
        MIOPEN_THROW(miopenStatusUnknownError, "Backward Data Algo cannot be executed");

    std::sort(begin(perf_db), end(perf_db));

    for(const auto& entry : perf_db)
        MIOPEN_LOG_I(entry.name << "\t" << entry.time << "\t" << entry.workspace);

    *returnedAlgoCount = std::min(requestAlgoCount, static_cast<int>(perf_db.size()));

    for(int i = 0; i < *returnedAlgoCount; i++)
    {
        perfResults[i].bwd_data_algo =
            static_cast<miopenConvBwdDataAlgorithm_t>(BwdDataAlgoResolver(perf_db[i].name));
        perfResults[i].time   = perf_db[i].time;
        perfResults[i].memory = perf_db[i].workspace;
    }

    MIOPEN_LOG_I("BWD Chosen Algorithm: " << perf_db[0].solver_id << " , " << perf_db[0].workspace
                                          << ", "
                                          << perf_db[0].time);
}

// BackwardDataAlgorithm()
void ConvolutionDescriptor::ConvolutionBackwardData(Handle& handle,
                                                    const void* alpha,
                                                    const TensorDescriptor& dyDesc,
                                                    ConstData_t dy,
                                                    const TensorDescriptor& wDesc,
                                                    ConstData_t w,
                                                    miopenConvBwdDataAlgorithm_t algo,
                                                    const void* beta,
                                                    const TensorDescriptor& dxDesc,
                                                    Data_t dx,
                                                    Data_t workSpace,
                                                    size_t workSpaceSize) const
{
    MIOPEN_LOG_I2("algo = " << algo << ", workspace = " << workSpaceSize);
    if(dx == nullptr || w == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(dyDesc.GetSize() != dxDesc.GetSize() || dyDesc.GetSize() != wDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(dyDesc.GetType() != dxDesc.GetType() || dyDesc.GetType() != wDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(wDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);
    //    if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[0]) {
    //       MIOPEN_THROW(miopenStatusBadParm);
    //    }
    if(dyDesc.GetSize() < 3)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, dyDesc, dy);
        miopen::checkNumericsInput(handle, wDesc, w);
        if(!float_equal(*(static_cast<const float*>(beta)), 0))
        {
            miopen::checkNumericsInput(handle, dxDesc, dx);
        }
    }

    {
        if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        ValidateGroupCount(dxDesc, wDesc, *this);

        // Launch all kernels and store the perf, workspace limits, etc.
        switch(algo)
        {
        case miopenConvolutionBwdDataAlgoDirect:
        {
            mlo_construct_direct2D construct_params(dxDesc, wDesc, dyDesc, *this, 0); // backward
            construct_params.setStream(&handle);

            std::string network_config;
            construct_params.mloBuildConf_Key(network_config);

            auto&& kernels =
                handle.GetKernels("miopenConvolutionBwdDataAlgoDirect", network_config);
            assert(1 <= kernels.size() && kernels.size() <= 2);

            visit_float(dyDesc.GetType(), [&](auto as_float) {
                float t1 = 0;
                if(kernels[0].GetName() == "gcnAsmConv1x1U")
                {
                    int unused       = 0;
                    int* return_addr = nullptr;

                    int N, C, H, W, K, n_groups;
                    construct_params.getCompiledInParameters(&N, &C, &H, &W, &K, &n_groups);

                    kernels[0](N,
                               C,
                               H,
                               W,
                               K,
                               n_groups,
                               unused,
                               unused,
                               dy,
                               w,
                               (kernels.size() == 2) ? workSpace : dx,
                               return_addr);
                    if(handle.IsProfilingEnabled())
                        t1 += handle.GetKernelTime();

                    if(kernels.size() == 2)
                    {
                        assert(kernels[1].GetName() == "UpSample");

                        /// \todo Initialization is required for upsampling. This leads to small
                        /// perf drop.
                        /// 1: Add kernel (from SetTensor) to the Solution in the Solver.
                        /// 2: Fix UpSample kernel, probably by means of conditional
                        /// compilation.
                        float zero = 0.f;
                        SetTensor(handle, dxDesc, dx, &zero);
                        if(handle.IsProfilingEnabled())
                            t1 += handle.GetKernelTime();

                        kernels[1](workSpace, dx);
                        if(handle.IsProfilingEnabled())
                            t1 += handle.GetKernelTime();
                    }
                }
                else
                {
                    float padding_val = 0;
                    kernels[0](dy, w, dx, as_float(padding_val));
                    if(handle.IsProfilingEnabled())
                        t1 += handle.GetKernelTime();
                }
                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(t1);
                }
            });
            break;
        }

        case miopenConvolutionBwdDataAlgoWinograd:
        {
            if(group_count > 1)
                MIOPEN_THROW("Winograd is not supported for group conv");

            mlo_construct_winograd construct_params(
                dxDesc, wDesc, dyDesc, *this, 0); // backward data

            construct_params.setStream(&handle);
            std::string network_config;
            construct_params.mloBuildConf_Key(network_config);

            auto kernel = handle.GetKernel("miopenConvolutionBwdDataAlgoWinograd", network_config);
            /// \todo Copied from ConvolutionDescriptor::FindConvBwdDataAlgorithm()
            static const int F_REVERSE_R = 1 << 0;
            static const int F_REVERSE_S = 1 << 1;
            static const int F_FLIP_K_C  = 1 << 2;
            int flags                    = F_REVERSE_R + F_REVERSE_S + F_FLIP_K_C;
            int reserved                 = 0;
            int* return_addr             = nullptr;
            int N, C, H, W, K, n_groups, out_H, out_W, R, S, pad_H, pad_W;
            construct_params.getCompiledInParameters(
                &N, &C, &H, &W, &K, &n_groups, &out_H, &out_W, &R, &S, &pad_H, &pad_W);
            // clang-format off
            MIOPEN_LOG_I2(" N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                    << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                    << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W); // clang-format on
            if(kernel.GetName() == "sp3AsmConvRxSU")
            {
                kernel(N,
                       C,
                       H,
                       W,
                       K,
                       n_groups,
                       flags,
                       reserved,
                       dy,
                       w,
                       dx,
                       return_addr,
                       R,
                       S,
                       pad_H,
                       pad_W,
                       out_H,
                       out_W);
            }
            else
            {
                kernel(N, C, H, W, K, n_groups, flags, reserved, dy, w, dx, return_addr);
            }
            break;
        }

        case miopenConvolutionBwdDataAlgoGEMM:
#if MIOPEN_USE_GEMM
        {
            if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
            {
                MIOPEN_THROW("GEMM convolution is disabled");
            }

            std::size_t in_n, in_c;
            std::tie(in_n, in_c) = tie_pick<0, 1>()(dxDesc.GetLengths());

            std::size_t wei_k = wDesc.GetLengths()[0];

            std::size_t spatial_dim = GetSpatialDimension();

            auto in_spatial  = boost::adaptors::slice(dxDesc.GetLengths(), 2, 2 + spatial_dim);
            auto wei_spatial = boost::adaptors::slice(wDesc.GetLengths(), 2, 2 + spatial_dim);
            auto out_spatial = boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + spatial_dim);

            if(GetSpatialDimension() == 2 &&
               miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
               miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
               miopen::all_of(GetConvStrides(), [](auto v) { return v == 2; }))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1, u2xv2");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1, u2xv2");
                }

                float t1 = 0;
                // Initialization required for upsampling in bwd direction
                float zero = 0.f;
                SetTensor(handle, dxDesc, dx, &zero);
                if(handle.IsProfilingEnabled())
                    t1 = handle.GetKernelTime();

                assert(workSpace != nullptr &&
                       workSpaceSize >= BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc));

                transpose_NCHW2CNHW(handle,
                                    in_n,
                                    wei_k,
                                    out_spatial[0],
                                    out_spatial[1],
                                    out_spatial[0],
                                    out_spatial[1],
                                    dy,
                                    workSpace,
                                    0,
                                    0,
                                    1,
                                    1,
                                    dyDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                if(group_count > 1)
                {
                    GemmDescriptor gemm_desc = CreateGemmDescriptorGroupConvCNHWBwdData(
                        wDesc, dyDesc, dxDesc, group_count);

                    CallGemmStridedBatched(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           workSpace,
                                           0,
                                           workSpace,
                                           dyDesc.GetElementSize(),
                                           nullptr,
                                           false);
                }
                else
                {
                    // dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
                    GemmDescriptor gemm_desc =
                        CreateGemmDescriptorConvCNHWBwdData(wDesc, dyDesc, dxDesc);

                    // dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
                    CallGemm(handle,
                             gemm_desc,
                             w,
                             0,
                             workSpace,
                             0,
                             workSpace,
                             dyDesc.GetElementSize(),
                             nullptr,
                             false);
                }
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                transpose_CNHW2NCHW(handle,
                                    in_n,
                                    in_c,
                                    out_spatial[0],
                                    out_spatial[1],
                                    in_spatial[0],
                                    in_spatial[1],
                                    workSpace,
                                    dx,
                                    dyDesc.GetElementSize(),
                                    0,
                                    GetConvStrides()[0],
                                    GetConvStrides()[1],
                                    dyDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(t1);
                }
            }
            // 1x1_stride=1 convolutions use GEMM and zero workspace
            else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
                    miopen::all_of(GetConvPads(), [](auto v) { return v == 0; }) &&
                    miopen::all_of(GetConvStrides(), [](auto v) { return v == 1; }))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1");

                    GemmDescriptor gemm_desc =
                        CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count);

                    float time_0 = 0;
                    for(std::size_t i = 0; i < in_n; i++)
                    {
                        std::size_t out_spatial_size =
                            std::accumulate(out_spatial.begin(),
                                            out_spatial.end(),
                                            std::size_t(1),
                                            std::multiplies<std::size_t>());

                        std::size_t in_spatial_size =
                            std::accumulate(in_spatial.begin(),
                                            in_spatial.end(),
                                            std::size_t(1),
                                            std::multiplies<std::size_t>());

                        std::size_t out_offset = i * wei_k * out_spatial_size;

                        std::size_t in_offset = i * in_c * in_spatial_size;

                        CallGemmStridedBatched(
                            handle, gemm_desc, w, 0, dy, out_offset, dx, in_offset, nullptr, false);

                        if(handle.IsProfilingEnabled())
                        {
                            if(i == in_n - 1)
                                handle.AccumKernelTime(time_0);
                            time_0 += handle.GetKernelTime();
                        }
                    }
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1");

                    // dx = transpose(w) * dy
                    GemmDescriptor gemm_desc =
                        CreateGemmStridedBatchedDescriptorConv1x1BwdData(wDesc, dyDesc, dxDesc);

                    // dx = transpose(w) * dy
                    CallGemmStridedBatched(handle, gemm_desc, w, 0, dy, 0, dx, 0, nullptr, false);
                }
            }
            // if not 1x1
            else
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, non 1x1");
                }
                assert(workSpace != nullptr &&
                       workSpaceSize >=
                           (BackwardDataGetWorkSpaceSizeGEMM(wDesc, dyDesc) * group_count));

                // dx = transpose(w) * dy
                GemmDescriptor gemm_desc{};
                if(group_count > 1)
                    gemm_desc =
                        CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count);
                else
                    gemm_desc = CreateGemmDescriptorConvBwdData(wDesc, dyDesc, dxDesc);

                handle.ResetKernelTime();

                std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                               out_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>());

                std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                              in_spatial.end(),
                                                              std::size_t(1),
                                                              std::multiplies<std::size_t>());

                float time_0 = 0;
                float t1     = 0;
                for(std::size_t i = 0; i < in_n; i++)
                {
                    std::size_t out_offset = i * wei_k * out_spatial_size;
                    std::size_t in_offset  = i * in_c * in_spatial_size;

                    // dx = transpose(w) * dy
                    if(group_count > 1)
                        CallGemmStridedBatched(
                            handle, gemm_desc, w, 0, dy, out_offset, workSpace, 0, nullptr, false);
                    else
                        CallGemm(handle,
                                 gemm_desc,
                                 w,
                                 0,
                                 dy,
                                 out_offset,
                                 workSpace,
                                 0,
                                 nullptr,
                                 false,
                                 GemmBackend_t::miopengemm);

                    if(handle.IsProfilingEnabled())
                        t1 = handle.GetKernelTime();

                    Col2ImGPU(handle,
                              GetSpatialDimension(),
                              workSpace,
                              out_spatial,
                              wei_spatial,
                              GetConvPads(),
                              GetConvStrides(),
                              GetConvDilations(),
                              in_c,
                              in_spatial,
                              dx,
                              in_offset,
                              dyDesc.GetType());

                    // Update times for both the kernels
                    if(handle.IsProfilingEnabled())
                    {
                        if(i == in_n - 1)
                            handle.AccumKernelTime(t1 + time_0);
                        else
                            handle.AccumKernelTime(t1);
                        time_0 += handle.GetKernelTime();
                    }
                }
            }
        }
        break;
#else
            MIOPEN_THROW("GEMM is not supported");
#endif

        case miopenConvolutionBwdDataAlgoFFT:
        {
            if(group_count > 1)
                MIOPEN_THROW("FFT is not supported for group conv");

            size_t workspace_fft = BackwardGetWorkSpaceSizeFFT(wDesc, dyDesc, dxDesc);
            if(workSpace != nullptr && workSpaceSize >= workspace_fft)
            {
                bool timed  = handle.IsProfilingEnabled();
                float timev = ExecuteBwdFFTKernel(
                    handle, dyDesc, dy, wDesc, w, dxDesc, dx, workSpace, workSpaceSize, timed);

                if(timed)
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(timev);
                }
            }
        }
        break;

        case miopenTransposeBwdDataAlgoGEMM: break;
        }
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, dxDesc, dx);
    }
}

template <typename T>
inline float EvaluateWrWDirectSolution(Handle& handle,
                                       const mlo_construct_BwdWrW2D& construct_params,
                                       const miopen::solver::ConvSolution& s,
                                       ConstData_t dy,
                                       ConstData_t x,
                                       Data_t dw,
                                       Data_t workSpace,
                                       const size_t workSpaceSize,
                                       T padding_val)
{
    float elapsed            = 0;
    const auto& kernels_info = s.construction_params;
    assert((s.workspce_sz != 0 && kernels_info.size() == 2) ||
           (s.workspce_sz == 0 && kernels_info.size() == 1));
    std::vector<KernelInvoke> kernels;
    AddKernels(handle, "", "", s, &kernels);
    const auto& k_info = kernels_info[0];
    if(kernels_info.size() == 1)
    {
        if(k_info.kernel_name == "gcnAsmConv3x3WrW" || k_info.kernel_name == "gcnAsmConv1x1WrW")
        {
            int unused       = 0;
            int* return_addr = nullptr;
            int N, C, H, W, K, n_groups;
            construct_params.getCompiledInParameters(&N, &C, &H, &W, &K, &n_groups);
            kernels[0](N, C, H, W, K, n_groups, unused, unused, x, dw, dy, return_addr);
        }
        else
        {
            kernels[0](dy, x, dw, padding_val);
        }
        elapsed = handle.GetKernelTime();
    }
    else
    {
        if(workSpace != nullptr && workSpaceSize >= s.workspce_sz)
        {
            if(k_info.kernel_name == "SubSample") // bwd stride 2
            {
                kernels[0](x, workSpace);
                elapsed = handle.GetKernelTime();
                if(kernels_info[1].kernel_name == "gcnAsmConv1x1WrW")
                {
                    int unused       = 0;
                    int* return_addr = nullptr;
                    int N, C, H, W, K, n_groups;
                    construct_params.getCompiledInParameters(&N, &C, &H, &W, &K, &n_groups);
                    kernels[1](
                        N, C, H, W, K, n_groups, unused, unused, workSpace, dw, dy, return_addr);
                }
                else
                {
                    kernels[1](dy, workSpace, dw, padding_val);
                }
                elapsed += handle.GetKernelTime();
            }
            else
            {
                kernels[0](dy, x, workSpace, padding_val);
                elapsed = handle.GetKernelTime();
                kernels[1](workSpace, dw); // reduction
                elapsed += handle.GetKernelTime();
            }
        }
    }
    return elapsed;
}

// ConvolutionBackwardWeightsGetWorkSpaceSize
// FindBackwardWeightsAlgorithm()
//
void ConvolutionDescriptor::FindConvBwdWeightsAlgorithm(Handle& handle,
                                                        const TensorDescriptor& dyDesc,
                                                        ConstData_t dy,
                                                        const TensorDescriptor& xDesc,
                                                        ConstData_t x,
                                                        const TensorDescriptor& dwDesc,
                                                        ConstData_t dw,
                                                        const int requestAlgoCount,
                                                        int* const returnedAlgoCount,
                                                        miopenConvAlgoPerf_t* perfResults,
                                                        Data_t workSpace,
                                                        size_t workSpaceSize,
                                                        bool exhaustiveSearch) const
{
    MIOPEN_LOG_I2("");
    if(x == nullptr || dw == nullptr || dy == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "Buffers cannot be NULL");
    if(returnedAlgoCount == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "returnedAlgoCount cannot be nullptr");
    if(perfResults == nullptr)
        MIOPEN_THROW(miopenStatusBadParm, "perfResults cannot be nullptr");
    if(requestAlgoCount < 1)
        MIOPEN_THROW(miopenStatusBadParm, "requestAlgoCount cannot be < 1");
    if(xDesc.GetType() == miopenInt8)
        MIOPEN_THROW(miopenStatusBadParm);

    *returnedAlgoCount = 0;

    // create a dummy buffer for use as output for the kernel calls
    // because kernels are called purely for timing purposes
    auto tmp_dw = handle.Create(dwDesc.GetElementSize() * GetTypeSize(dwDesc.GetType()));

    AutoEnableProfiling enableProfiling{handle};

    // < algorith_name, <time, workspace_size> >
    std::vector<PerfField> perf_db;

    {
#if MIOPEN_USE_GEMM
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
        { // GEMM based
            const bool time_precision = (!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}));

            ValidateGroupCount(xDesc, dwDesc, *this);

            std::size_t in_n, in_c;
            std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

            auto in_spatial =
                boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + GetSpatialDimension());
            auto wei_spatial =
                boost::adaptors::slice(dwDesc.GetLengths(), 2, 2 + GetSpatialDimension());
            auto out_spatial =
                boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + GetSpatialDimension());

            size_t workspace_req =
                BackwardWeightsGetWorkSpaceSizeGEMM(dyDesc, dwDesc) * group_count;

            float time_gemm = 0;

            // if not 1x1
            if((miopen::any_of(wei_spatial, [](auto v) { return v != 1; }) ||
                miopen::any_of(GetConvPads(), [](auto v) { return v != 0; }) ||
                miopen::any_of(GetConvStrides(), [](auto v) { return v != 1; })) &&
               (workSpace != nullptr && workSpaceSize >= workspace_req))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, non 1x1");
                }
                float time_im2col = 0;
                int in_offset     = 0;
                time_im2col       = Im2ColGPU(handle,
                                        GetSpatialDimension(),
                                        x,
                                        in_offset,
                                        in_c,
                                        in_spatial,
                                        wei_spatial,
                                        out_spatial,
                                        GetConvPads(),
                                        GetConvStrides(),
                                        GetConvDilations(),
                                        workSpace,
                                        dyDesc.GetType());

                // dw = dy * transpose(Im2Col(x))
                GemmDescriptor gemm_desc =
                    group_count > 1
                        ? CreateGemmDescriptorGroupConvBwdWeight(dyDesc, xDesc, dwDesc, group_count)
                        : CreateGemmDescriptorConvBwdWeight(dyDesc, xDesc, dwDesc);

                miopenStatus_t gemm_status = CallGemmTimeMeasure(
                    handle,
                    gemm_desc,
                    dy,
                    0,
                    workSpace,
                    0,
                    tmp_dw.get(),
                    0,
                    nullptr,
                    time_precision,
                    group_count > 1 ? callGemmStridedBatched : callGemm,
                    group_count > 1 ? GemmBackend_t::rocblas : GemmBackend_t::miopengemm);

                time_gemm = in_n * (time_im2col + handle.GetKernelTime());

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(PerfField{
                        "miopenConvolutionBwdWeightsAlgoGEMM", "GEMM", time_gemm, workspace_req});
            }
            // 1x1 does not require im2col or workspace
            else if(miopen::any_of(wei_spatial, [](auto v) { return v == 1; }) &&
                    miopen::any_of(GetConvPads(), [](auto v) { return v == 0; }) &&
                    miopen::any_of(GetConvStrides(), [](auto v) { return v == 1; }))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1");
                }

                // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
                GemmDescriptor gemm_desc =
                    group_count > 1
                        ? CreateGemmDescriptorGroupConvBwdWeight(dyDesc, xDesc, dwDesc, group_count)
                        : CreateGemmStridedBatchedDescriptorConv1x1BwdWeight(dyDesc, xDesc, dwDesc);

                miopenStatus_t gemm_status = CallGemmTimeMeasure(
                    handle,
                    gemm_desc,
                    dy,
                    0,
                    x,
                    0,
                    tmp_dw.get(),
                    0,
                    nullptr,
                    time_precision,
                    group_count > 1 ? callGemmStridedBatched : callGemmStridedBatchedSequential,
                    group_count > 1 ? GemmBackend_t::rocblas : GemmBackend_t::miopengemm);

                time_gemm = handle.GetKernelTime();
                if(group_count > 1)
                    time_gemm *= in_n;

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdWeightsAlgoGEMM", "GEMM", time_gemm, 0});
            }
        }
#endif

        // direct convolution
        {
            if(GetSpatialDimension() == 2 && !miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
            {
                mlo_construct_BwdWrW2D construct_params(
                    xDesc, dwDesc, dyDesc, *this, 0); // backward with regards to weights
                construct_params.setDoSearch(exhaustiveSearch);
                construct_params.setStream(&handle);

                std::string network_config;
                construct_params.mloBuildConf_Key(network_config);
                const std::string algorithm_name = "miopenConvolutionBwdWeightsAlgoDirect";

                miopen::solver::ConvSolution selected{miopenStatusUnknownError};
                float best     = std::numeric_limits<float>::max();
                const auto all = FindAllSolutions(construct_params);

                visit_float(dyDesc.GetType(), [&](auto as_float) {
                    for(const auto& sol : all)
                    {
                        /// \todo If there is only one solution available,
                        /// we can avoid wasting time for building kernels with empty
                        /// algorithm_name and network_config.
                        float elapsed = EvaluateWrWDirectSolution(handle,
                                                                  construct_params,
                                                                  sol,
                                                                  dy,
                                                                  x,
                                                                  tmp_dw.get(),
                                                                  workSpace,
                                                                  workSpaceSize,
                                                                  as_float(0.0f));
                        MIOPEN_LOG_I(sol << ": " << elapsed << (elapsed < best ? " < " : " >= ")
                                         << best);
                        if(elapsed < best)
                        {
                            best     = elapsed;
                            selected = sol;
                        }
                    }
                });
                if(selected.Succeeded())
                {
                    AddKernels(handle, algorithm_name, network_config, selected, nullptr);
                    MIOPEN_LOG_I("Selected: " << selected << ": " << best << ", workspce_sz = "
                                              << selected.workspce_sz);
                    perf_db.push_back(
                        PerfField{algorithm_name, selected.solver_id, best, selected.workspce_sz});
                }
            }
        }

        if(GetSpatialDimension() == 2)
        {
            try
            {
                WinogradKernelParams k_p;
                KernelInvoke kernel_wino;
                std::string network_config;
                std::string solver_id;
                if(FindWinogradKernel<mlo_construct_winograd_wrw>(handle,
                                                                  xDesc,
                                                                  dwDesc,
                                                                  dyDesc,
                                                                  k_p,
                                                                  kernel_wino,
                                                                  solver_id,
                                                                  0,
                                                                  &network_config) == 0)
                {
                    float time_wino                = 0;
                    static const int F_FLIP_K_C    = 1 << 2;
                    static const int F_NKC_STRIDES = 1 << 9;
                    int flags                      = F_FLIP_K_C + F_NKC_STRIDES;
                    int reserved                   = 0;
                    int* reserved_ptr              = nullptr;
                    bool isRxS;
                    int pad_H = GetConvPads()[0];
                    int pad_W = GetConvPads()[1];
                    int N, C, H, W, K, n_groups, out_H, out_W, R, S, unused;
                    // For bwd & wrw inputs and outputs reside in k_p in reversed order.
                    std::tie(N, K, out_H, out_W, C, n_groups, H, W, R, S, unused, unused, isRxS) =
                        k_p;
                    assert(isRxS);
                    using dataType = float;
                    int d_N_stride = H * W * static_cast<int>(sizeof(dataType));
                    int d_C_stride = C * d_N_stride;
                    int f_K_stride = out_H * out_W * static_cast<int>(sizeof(dataType));
                    int f_C_stride = K * f_K_stride;
                    int o_N_stride = R * S * static_cast<int>(sizeof(dataType));
                    int o_K_stride = C * o_N_stride;
                    // clang-format off
                MIOPEN_LOG_I2(" N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                        << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                        << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                        << " d_N_stride=" << d_N_stride << " d_C_stride=" << d_C_stride
                        << " f_K_stride=" << f_K_stride << " f_C_stride=" << f_C_stride
                        << " o_N_stride=" << o_N_stride << " o_K_stride=" << o_K_stride); // clang-format on
                    kernel_wino(C,
                                N,
                                H,
                                W,
                                K,
                                n_groups,
                                flags,
                                reserved,
                                x,
                                dy,
                                tmp_dw.get(),
                                reserved_ptr, // Unused return_addr.
                                out_H,
                                out_W,
                                pad_H, // Like Fwd wino.
                                pad_W,
                                R,
                                S,
                                reserved_ptr, // Unused bias_addr.
                                reserved,     // Unused relu_alpha.
                                d_N_stride,
                                d_C_stride,
                                f_K_stride,
                                f_C_stride,
                                o_N_stride,
                                o_K_stride);
                    time_wino = handle.GetKernelTime();
                    perf_db.push_back(PerfField{
                        "miopenConvolutionBwdWeightsAlgoWinograd", solver_id, time_wino, 0});
                }
            }
            catch(const miopen::Exception& ex)
            {
                MIOPEN_LOG_W("Find Winograd WrW failed:" << ex.what());
            }
        }
    }

    if(perf_db.empty())
        MIOPEN_THROW("Bwd Weights Convolution cannot be executed due to incorrect params");

    std::sort(begin(perf_db), end(perf_db));

    for(const auto& entry : perf_db)
        MIOPEN_LOG_I(entry.name << "\t" << entry.time << "\t" << entry.workspace);

    *returnedAlgoCount = std::min(requestAlgoCount, static_cast<int>(perf_db.size()));

    for(int i = 0; i < *returnedAlgoCount; i++)
    {
        perfResults[i].bwd_weights_algo =
            static_cast<miopenConvBwdWeightsAlgorithm_t>(BwdWeightsAlgoResolver(perf_db[i].name));
        perfResults[i].time   = perf_db[i].time;
        perfResults[i].memory = perf_db[i].workspace;
    }
    MIOPEN_LOG_I("BWrW Chosen Algorithm: " << perf_db[0].solver_id << " , " << perf_db[0].workspace
                                           << ", "
                                           << perf_db[0].time);
}

// BackwardWeightsAlgorithm()
void ConvolutionDescriptor::ConvolutionBackwardWeights(Handle& handle,
                                                       const void* alpha,
                                                       const TensorDescriptor& dyDesc,
                                                       ConstData_t dy,
                                                       const TensorDescriptor& xDesc,
                                                       ConstData_t x,
                                                       miopenConvBwdWeightsAlgorithm_t algo,
                                                       const void* beta,
                                                       const TensorDescriptor& dwDesc,
                                                       Data_t dw,
                                                       Data_t workSpace,
                                                       size_t workSpaceSize) const
{
    MIOPEN_LOG_I2("algo = " << algo << ", workspace = " << workSpaceSize);
    if(x == nullptr || dw == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(dyDesc.GetSize() != dwDesc.GetSize() || dyDesc.GetSize() != xDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(dyDesc.GetType() != dwDesc.GetType() || dyDesc.GetType() != xDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetType() == miopenInt8)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(dyDesc.GetLengths()[0] != xDesc.GetLengths()[0])
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(dyDesc.GetSize() < 3)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, dyDesc, dy);
        miopen::checkNumericsInput(handle, xDesc, x);
        if(!float_equal(*(static_cast<const float*>(beta)), 0))
        {
            miopen::checkNumericsInput(handle, dwDesc, dw);
        }
    }

    {
        ValidateGroupCount(xDesc, dwDesc, *this);

        switch(algo)
        {
        case miopenConvolutionBwdWeightsAlgoGEMM: {
#if MIOPEN_USE_GEMM
            if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
            {
                MIOPEN_THROW("GEMM convolution is disabled");
            }

            std::size_t in_n, in_c;
            std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

            std::size_t wei_k = dwDesc.GetLengths()[0];

            auto in_spatial =
                boost::adaptors::slice(xDesc.GetLengths(), 2, 2 + GetSpatialDimension());
            auto wei_spatial =
                boost::adaptors::slice(dwDesc.GetLengths(), 2, 2 + GetSpatialDimension());
            auto out_spatial =
                boost::adaptors::slice(dyDesc.GetLengths(), 2, 2 + GetSpatialDimension());

            // Zeroing out the output buffer
            float zero = 0.0f;
            SetTensor(handle, dwDesc, dw, &zero);

            handle.ResetKernelTime();
            float time_0 = 0;
            if((miopen::any_of(wei_spatial, [](auto v) { return v != 1; }) ||
                miopen::any_of(GetConvPads(), [](auto v) { return v != 0; }) ||
                miopen::any_of(GetConvStrides(), [](auto v) { return v != 1; })))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, non 1x1");
                }
                assert(workSpace != nullptr &&
                       workSpaceSize >=
                           (BackwardWeightsGetWorkSpaceSizeGEMM(dyDesc, dwDesc) * group_count));

                std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                               out_spatial.end(),
                                                               std::size_t(1),
                                                               std::multiplies<std::size_t>());

                std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                              in_spatial.end(),
                                                              std::size_t(1),
                                                              std::multiplies<std::size_t>());

                float t1 = 0;

                for(std::size_t i = 0; i < in_n; i++)
                {
                    std::size_t out_offset = i * wei_k * out_spatial_size;

                    std::size_t in_offset = i * in_c * in_spatial_size;

                    Im2ColGPU(handle,
                              GetSpatialDimension(),
                              x,
                              in_offset,
                              in_c,
                              in_spatial,
                              wei_spatial,
                              out_spatial,
                              GetConvPads(),
                              GetConvStrides(),
                              GetConvDilations(),
                              workSpace,
                              dyDesc.GetType());

                    if(handle.IsProfilingEnabled())
                        t1 = handle.GetKernelTime();

                    if(group_count > 1)
                    {
                        GemmDescriptor gemm_desc = CreateGemmDescriptorGroupConvBwdWeight(
                            dyDesc, xDesc, dwDesc, group_count);
                        CallGemmStridedBatched(
                            handle, gemm_desc, dy, out_offset, workSpace, 0, dw, 0, nullptr, false);
                    }
                    else
                    {
                        // dw = dy * transpose(Im2Col(x))
                        GemmDescriptor gemm_desc =
                            CreateGemmDescriptorConvBwdWeight(dyDesc, xDesc, dwDesc);

                        // dw = dy * transpose(Im2Col(x))
                        CallGemm(handle,
                                 gemm_desc,
                                 dy,
                                 out_offset,
                                 workSpace,
                                 0,
                                 dw,
                                 0,
                                 nullptr,
                                 false,
                                 GemmBackend_t::miopengemm);
                    }
                    // Update times for both the kernels
                    if(handle.IsProfilingEnabled())
                    {
                        if(i == in_n - 1)
                            handle.AccumKernelTime(t1 + time_0);
                        else
                            handle.AccumKernelTime(t1);
                        time_0 += handle.GetKernelTime();
                    }
                }
            }
            else if(miopen::any_of(wei_spatial, [](auto v) { return v == 1; }) &&
                    miopen::any_of(GetConvPads(), [](auto v) { return v == 0; }) &&
                    miopen::any_of(GetConvStrides(), [](auto v) { return v == 1; }))
            {
                if(group_count > 1)
                {
                    MIOPEN_LOG_FUNCTION("groupconv, 1x1");

                    GemmDescriptor gemm_desc =
                        CreateGemmDescriptorGroupConvBwdWeight(dyDesc, xDesc, dwDesc, group_count);

                    std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                                   out_spatial.end(),
                                                                   std::size_t(1),
                                                                   std::multiplies<std::size_t>());

                    std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                                  in_spatial.end(),
                                                                  std::size_t(1),
                                                                  std::multiplies<std::size_t>());

                    for(std::size_t i = 0; i < in_n; i++)
                    {
                        std::size_t out_offset = i * wei_k * out_spatial_size;

                        std::size_t in_offset = i * in_c * in_spatial_size;

                        CallGemmStridedBatched(
                            handle, gemm_desc, dy, out_offset, x, in_offset, dw, 0, nullptr, false);

                        if(handle.IsProfilingEnabled())
                        {
                            if(i == in_n - 1)
                                handle.AccumKernelTime(time_0);
                            time_0 += handle.GetKernelTime();
                        }
                    }
                }
                else
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1");

                    // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
                    GemmDescriptor gemm_desc =
                        CreateGemmStridedBatchedDescriptorConv1x1BwdWeight(dyDesc, xDesc, dwDesc);

                    // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
                    CallGemmStridedBatchedSequential(handle,
                                                     gemm_desc,
                                                     dy,
                                                     0,
                                                     x,
                                                     0,
                                                     dw,
                                                     0,
                                                     nullptr,
                                                     false,
                                                     GemmBackend_t::miopengemm);
                }
            }
#else
            MIOPEN_THROW("GEMM is not supported");
#endif
        }
        break;

        case miopenConvolutionBwdWeightsAlgoDirect:
        {
            mlo_construct_BwdWrW2D construct_params(
                xDesc, dwDesc, dyDesc, *this, 0); // backward with regards to weights
            construct_params.setStream(&handle);

            visit_float(dyDesc.GetType(), [&](auto as_float) {
                std::string network_config;
                construct_params.mloBuildConf_Key(network_config);

                auto&& kernels =
                    handle.GetKernels("miopenConvolutionBwdWeightsAlgoDirect", network_config);
                if(kernels.empty())
                    MIOPEN_THROW("Error running direct backwards weights convolution. Was Find() "
                                 "executed previously?");
                auto kernel = kernels[0];

                handle.ResetKernelTime();

                if((kernel.GetName() == "gcnAsmConv3x3WrW") ||
                   (kernel.GetName() == "gcnAsmConv1x1WrW"))
                {
                    assert(kernels.size() == 1);
                    int unused       = 0;
                    int* return_addr = nullptr;
                    int N, C, H, W, K, n_groups;
                    construct_params.getCompiledInParameters(&N, &C, &H, &W, &K, &n_groups);
                    kernel(N, C, H, W, K, n_groups, unused, unused, x, dw, dy, return_addr);
                }
                else if(kernels.size() == 1)
                {
                    float padding_val = 0;
                    kernel(dy, x, dw, as_float(padding_val));
                }
                else
                {
                    assert(kernels.size() == 2);

                    // Relaxed the workspace size assertion as
                    // it is possible to receive the workspace size from perf db
                    // that matches with the best chosen solver but less than
                    // the worst workspace required among all solvers.
                    // ConvOclBwdWrW2<n> solvers are the examples.
                    // With the relaxed workspace size assertion, need to
                    // distinguish group vs non-group case vanishes.
                    assert(workSpace != nullptr && workSpaceSize > 0);

                    if(kernel.GetName() == "SubSample")
                    {
                        // subsampling kernel
                        kernel(x, workSpace);
                        float time0 = handle.GetKernelTime();

                        // wrw  kernel
                        if(kernels[1].GetName() == "gcnAsmConv1x1WrW")
                        {
                            int unused       = 0;
                            int* return_addr = nullptr;
                            int N, C, H, W, K, n_groups, out_H, out_W;
                            construct_params.getCompiledInParameters(
                                &N, &C, &H, &W, &K, &n_groups, &out_H, &out_W);
                            // out_H/W are used instead of H/W; see comment in
                            // AsmImgHeight(), conv_asm_dir_BwdWrW1x1.cpp.
                            kernels[1](N,
                                       C,
                                       out_H,
                                       out_W,
                                       K,
                                       n_groups,
                                       unused,
                                       unused,
                                       workSpace,
                                       dw,
                                       dy,
                                       return_addr);
                        }
                        else
                        {
                            float padding_val = 0;
                            kernels[1](dy, workSpace, dw, as_float(padding_val));
                        }

                        handle.AccumKernelTime(time0);
                    }
                    else
                    {
                        float padding_val = 0;
                        kernel(dy, x, workSpace, as_float(padding_val));

                        float time0 = handle.GetKernelTime();
                        // second kernel - reduction
                        kernels[1](workSpace, dw);

                        handle.AccumKernelTime(time0);
                    }
                }
            });
        }
        break;
        case miopenConvolutionBwdWeightsAlgoWinograd:
        {
            mlo_construct_winograd_wrw construct_params(xDesc, dwDesc, dyDesc, *this, 0);
            construct_params.setStream(&handle);

            std::string network_config;
            construct_params.mloBuildConf_Key(network_config);

            auto&& kernels =
                handle.GetKernels("miopenConvolutionBwdWeightsAlgoWinograd", network_config);
            if(kernels.size() != 1)
                MIOPEN_THROW("Error running Winograd WrW. Was Find() run previously?");
            auto kernel = kernels[0];

            static const int F_FLIP_K_C    = 1 << 2;
            static const int F_NKC_STRIDES = 1 << 9;
            int flags                      = F_FLIP_K_C + F_NKC_STRIDES;
            int reserved                   = 0;
            int* reserved_ptr              = nullptr;
            int pad_H                      = GetConvPads()[0];
            int pad_W                      = GetConvPads()[1];
            int N, C, H, W, K, n_groups, out_H, out_W, R, S, unused;
            // For bwd & wrw inputs and outputs reside in k_p in reversed order.
            construct_params.getCompiledInParameters(
                &N, &K, &out_H, &out_W, &C, &n_groups, &H, &W, &R, &S, &unused, &unused);
            using dataType = float;
            int d_N_stride = H * W * static_cast<int>(sizeof(dataType));
            int d_C_stride = C * d_N_stride;
            int f_K_stride = out_H * out_W * static_cast<int>(sizeof(dataType));
            int f_C_stride = K * f_K_stride;
            int o_N_stride = R * S * static_cast<int>(sizeof(dataType));
            int o_K_stride = C * o_N_stride;
            // clang-format off
            MIOPEN_LOG_I2(" N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                    << " n_groups=" << n_groups << " flags=" << flags << " R=" << R << " S=" << S
                    << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                    << " d_N_stride=" << d_N_stride << " d_C_stride=" << d_C_stride
                    << " f_K_stride=" << f_K_stride << " f_C_stride=" << f_C_stride
                    << " o_N_stride=" << o_N_stride << " o_K_stride=" << o_K_stride ); // clang-format on
            kernel(C,
                   N,
                   H,
                   W,
                   K,
                   n_groups,
                   flags,
                   reserved,
                   x,
                   dy,
                   dw,
                   reserved_ptr,
                   out_H,
                   out_W,
                   pad_H,
                   pad_W,
                   R,
                   S,
                   reserved_ptr,
                   reserved,
                   d_N_stride,
                   d_C_stride,
                   f_K_stride,
                   f_C_stride,
                   o_N_stride,
                   o_K_stride);
        }
        break;
        }
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, dwDesc, dw);
    }
}

void ConvolutionBackwardBias(Handle& handle,
                             const void* alpha,
                             const TensorDescriptor& dyDesc,
                             ConstData_t dy,
                             const void* beta,
                             const TensorDescriptor& dbDesc,
                             Data_t db)
{
    if(dy == nullptr || db == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(dyDesc.GetLengths()[1] != dbDesc.GetLengths()[1])
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, dyDesc, dy);
    }

    std::size_t out_n, out_k, stride_n, stride_k;
    std::tie(out_n, out_k)       = tie_pick<0, 1>()(dyDesc.GetLengths());
    std::tie(stride_n, stride_k) = tie_pick<0, 1>()(dyDesc.GetStrides());
    std::string program_name = "MIOpenConvBwdBias.cl";
    std::string kernel_name  = "MIOpenConvBwdB";

    std::string params;
    std::size_t lcl_grp_size0 = 256;
    std::size_t lcl_grp_size1 = 1;
    std::size_t local_mem_sz  = 256;

    std::size_t map_size = std::accumulate(dyDesc.GetLengths().begin() + 2,
                                           dyDesc.GetLengths().end(),
                                           std::size_t(1),
                                           std::multiplies<std::size_t>());
    std::size_t read_unit        = 4;
    std::size_t map_size_aligned = (map_size + (read_unit - 1)) / read_unit;
    std::size_t off_pix          = map_size - (map_size / read_unit) * read_unit;

    params = " -DMLO_CONVBWD_GROUP_SZ0=" + std::to_string(lcl_grp_size0);
    params += " -DMLO_CONVBWD_GROUP_SZ1=" + std::to_string(lcl_grp_size1);
    params += " -DMLO_CONVBWDB_LCL_MEMSZ=" + std::to_string(local_mem_sz);
    params += " -DMLO_CONVBWDB_UNITSIZE=" + std::to_string(read_unit);
    params += " -DMLO_OUT_BATCH_SZ=" + std::to_string(out_n);
    params += " -DMLO_OUT_CHANNEL_STRIDE=" + std::to_string(stride_k);
    params += " -DMLO_OUT_BATCH_STRIDE=" + std::to_string(stride_n);
    params += " -DMLO_WK_SIZE=" + std::to_string(map_size_aligned);
    params += " -DMLO_N_PIX_OFF=" + std::to_string(off_pix);
    if(dyDesc.GetType() == miopenFloat)
    {
        params += " -DMIOPEN_USE_FP16=0 ";
        params += " -DMIOPEN_USE_FP32=1 ";
    }
    else if(dyDesc.GetType() == miopenHalf)
    {
        params += " -DMIOPEN_USE_FP16=1 ";
        params += " -DMIOPEN_USE_FP32=0 ";
    }

    const std::vector<size_t> vld = {lcl_grp_size0, size_t{1}, size_t{1}};
    const std::vector<size_t> vgd = {lcl_grp_size0, static_cast<size_t>(out_k), size_t{1}};

    handle.AddKernel("miopenConvolutionBwdBias", "", program_name, kernel_name, vld, vgd, params)(
        dy, db);

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, dbDesc, db);
    }
}

} // namespace miopen
