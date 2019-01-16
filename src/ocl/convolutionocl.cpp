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
#include <miopen/db.hpp>
#include <miopen/env.hpp>
#include <miopen/util.hpp>
#include <miopen/solver.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/check_numerics.hpp>

#if MIOPEN_USE_GEMM
#include <miopen/gemm.hpp>
#include <miopen/gemm_v2.hpp>
#endif

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_ENABLE_FIND_DB)
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
    try
    {
        mlo_construct_winograd construct_params(xDesc, wDesc, yDesc, *this, direction);
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

        std::string algorithm = (direction == 1) ? "miopenConvolutionFwdAlgoWinograd"
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

    if(!IsDirectSupported(wDesc) || miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
        return {};

    mlo_construct_direct2D construct_params(xDesc, wDesc, yDesc, *this, isForward ? 1 : 0);
    construct_params.setDoSearch(exhaustiveSearch);
    construct_params.saveSearchRequest(true);
    construct_params.setGeneralCompOptions("");
    construct_params.setStream(&handle);
    construct_params.setupRocm();

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

    // GEMM based
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_c, wei_h, wei_w;

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    if(conv.mode == miopenTranspose)
    {
        std::tie(std::ignore, wei_n, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

#if MIOPEN_USE_MIOPENGEMM
        if(xDesc.GetType() == miopenFloat)
        {
            std::string network_config;

            size_t workspace_req = conv.BackwardDataGetWorkSpaceSizeGEMM(handle, wDesc, xDesc);
            float time_gemm      = 0;
            GemmGeometry gg =
                CreateGemmGeometryConvBwdData(xDesc, wDesc, yDesc, true, network_config);

            // 1x1 does not require im2col or workspace
            if(wei_h == 1 && wei_w == 1 && conv.v == 1 && conv.u == 1 && conv.pad_h == 0 &&
               conv.pad_w == 0)
            {
                MIOPEN_LOG_FUNCTION("transpose, 1x1");

                gg.FindSolution(.003, handle, w, x, tmp_y.get(), false);
                gg.RunGemm(handle, w, x, tmp_y.get(), 0, 0, 0);

                time_gemm = in_n * handle.GetKernelTime();

                record.SetValues(
                    "miopenConvolutionFwdAlgoGEMM",
                    FindDbData{"gemm", time_gemm, 0, network_config}); // Todo: gemm solver id?
            }

            // if not 1x1
            else if(workSpace != nullptr && workSpaceSize >= workspace_req)
            {
                MIOPEN_LOG_FUNCTION("transpose, non 1x1");

                float time_col2im = 0;
                int out_offset    = 0;

                gg.FindSolution(.003, handle, w, x, workSpace, false);
                gg.RunGemm(handle, w, x, workSpace, 0, 0, 0);

                time_gemm   = in_n * handle.GetKernelTime();
                time_col2im = Col2ImGPU(handle,
                                        workSpace,
                                        in_h,
                                        in_w,
                                        wei_h,
                                        wei_w,
                                        conv.pad_h,
                                        conv.pad_w,
                                        conv.u,
                                        conv.v,
                                        conv.dilation_h,
                                        conv.dilation_w,
                                        wei_n,
                                        out_h,
                                        out_w,
                                        tmp_y.get(),
                                        out_offset,
                                        xDesc.GetType());

                time_gemm += in_n * time_col2im;

                record.SetValues(
                    "miopenConvolutionFwdAlgoGEMM",
                    FindDbData{
                        "gemm", time_gemm, workspace_req, network_config}); // Todo: gemm solver id?
            }
        }
#else
        (void)workSpace;     // Suppress warning
        (void)workSpaceSize; // Suppress warning
#endif
    }
    else if(conv.mode == miopenGroupConv || conv.mode == miopenDepthwise)
    {
        std::tie(wei_n, wei_c, wei_h, wei_w) = tien<4>(wDesc.GetLengths());
        if(in_c % conv.group_count != 0 || wei_n % conv.group_count != 0 ||
           conv.group_count > in_c || conv.group_count > wei_n || conv.group_count < 1 ||
           (conv.mode == miopenDepthwise && conv.group_count != in_c))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
        if(in_c / conv.group_count != wei_c || (conv.mode == miopenDepthwise && wei_c != 1))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");

#if MIOPEN_USE_GEMM
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
        { // GEMM algo
            float time_gemm = 0;

            // Use transpose path if input ht and width <= 14 for 1x1_stride=1 convolutions OR
            // for 1x1_stride=2
            if((wei_h == 1 && wei_w == 1 && conv.pad_h == 0 && conv.pad_w == 0) &&
               ((in_h <= 14 && in_w <= 14 && conv.u == 1 && conv.v == 1) ||
                (conv.u == 2 && conv.v == 2)))
            {
                size_t workspace_req = conv.ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc);
                if(workSpace != nullptr && workSpaceSize >= workspace_req)
                {
                    transpose_NCHW2CNHW(handle,
                                        in_n,
                                        in_c,
                                        in_h,
                                        in_w,
                                        out_h,
                                        out_w,
                                        x,
                                        workSpace,
                                        0,
                                        0,
                                        conv.v,
                                        conv.u,
                                        xDesc.GetType());
                    time_gemm = handle.GetKernelTime();

                    size_t x_t_size = in_n * in_c * out_h * out_w;
                    GemmDescriptor gemm_desc =
                        CreateGemmDescriptorGroupConvCNHWFwd(wDesc, xDesc, yDesc, conv.group_count);
                    std::string kcache_key;
                    miopenStatus_t gemm_status = miopenStatusNotInitialized;
                    if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                    {
                        // rocBLAS need extra warm-up call for accurate timing
                        CallGemmStridedBatched(handle,
                                               gemm_desc,
                                               w,
                                               0,
                                               workSpace,
                                               0,
                                               workSpace,
                                               x_t_size,
                                               nullptr,
                                               false);

                        gemm_status = CallGemmStridedBatched(handle,
                                                             gemm_desc,
                                                             w,
                                                             0,
                                                             workSpace,
                                                             0,
                                                             workSpace,
                                                             x_t_size,
                                                             &kcache_key,
                                                             true);
                    }
                    else
                    {
                        gemm_status = CallGemmStridedBatched(handle,
                                                             gemm_desc,
                                                             w,
                                                             0,
                                                             workSpace,
                                                             0,
                                                             workSpace,
                                                             x_t_size,
                                                             &kcache_key,
                                                             false);
                    }

                    time_gemm += (handle.GetKernelTime());

                    transpose_CNHW2NCHW(handle,
                                        in_n,
                                        wei_n,
                                        out_h,
                                        out_w,
                                        out_h,
                                        out_w,
                                        workSpace,
                                        tmp_y.get(),
                                        x_t_size,
                                        0,
                                        1,
                                        1,
                                        xDesc.GetType());
                    time_gemm += handle.GetKernelTime();

                    if(gemm_status == miopenStatusSuccess)
                        record.SetValues("miopenConvolutionFwdAlgoGEMM",
                                         FindDbData{"gemm",
                                                    time_gemm,
                                                    workspace_req,
                                                    kcache_key}); // Todo: gemm solver id?
                }
            }
            // 1x1_stride=1 with GEMM and zero workspace
            else if(wei_h == 1 && wei_w == 1 && conv.pad_h == 0 && conv.pad_w == 0 && conv.u == 1 &&
                    conv.v == 1)
            {
                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, conv.group_count);
                std::string kcache_key;
                miopenStatus_t gemm_status = miopenStatusNotInitialized;
                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, x, 0, tmp_y.get(), 0, nullptr, false);

                    gemm_status = CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, x, 0, tmp_y.get(), 0, &kcache_key, true);
                }
                else
                {
                    gemm_status = CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, x, 0, tmp_y.get(), 0, &kcache_key, false);
                }

                time_gemm = in_n * (handle.GetKernelTime());

                if(gemm_status == miopenStatusSuccess)
                    record.SetValues(
                        "miopenConvolutionFwdAlgoGEMM",
                        FindDbData{"gemm", time_gemm, 0, kcache_key}); // Todo: gemm solver id?
            }
            // if not 1x1
            else if(workSpace != nullptr &&
                    workSpaceSize >= conv.ForwardGetWorkSpaceSizeGEMM(handle, wDesc, yDesc))
            {
                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, conv.group_count);

                float time_im2col = 0;
                size_t in_offset  = 0;
                time_im2col       = Im2ColGPU(handle,
                                        xDesc.GetElementSize(),
                                        x,
                                        in_offset,
                                        in_c,
                                        in_h,
                                        in_w,
                                        wei_h,
                                        wei_w,
                                        out_h,
                                        out_w,
                                        conv.pad_h,
                                        conv.pad_w,
                                        conv.u,
                                        conv.v,
                                        conv.dilation_h,
                                        conv.dilation_w,
                                        workSpace,
                                        xDesc.GetType());

                std::string kcache_key;
                miopenStatus_t gemm_status = miopenStatusNotInitialized;
                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need extra warm-up call for accurate timing
                    CallGemmStridedBatched(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           workSpace,
                                           0,
                                           tmp_y.get(),
                                           0,
                                           nullptr,
                                           false,
                                           GemmBackend_t::miopengemm);

                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         workSpace,
                                                         0,
                                                         tmp_y.get(),
                                                         0,
                                                         &kcache_key,
                                                         true,
                                                         GemmBackend_t::miopengemm);
                }
                else
                {
                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         workSpace,
                                                         0,
                                                         tmp_y.get(),
                                                         0,
                                                         &kcache_key,
                                                         false,
                                                         GemmBackend_t::miopengemm);
                }

                time_gemm = in_n * (time_im2col + handle.GetKernelTime());

                if(gemm_status == miopenStatusSuccess)
                    record.SetValues(
                        "miopenConvolutionFwdAlgoGEMM",
                        FindDbData{"gemm",
                                   time_gemm,
                                   conv.ForwardGetWorkSpaceSizeGEMM(handle, wDesc, yDesc),
                                   kcache_key}); // Todo: gemm solver id?
            }
        }
#else
        (void)workSpace;     // Suppress warning
        (void)workSpaceSize; // Suppress warning
#endif
        if(conv.dilation_h == 1 && conv.dilation_w == 1)
        {
            { // Direct algo
                std::string network_config;
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
                                             << best
                                             << ", workspce_sz = "
                                             << sol.workspce_sz);
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
        }
    }
    else if(conv.mode == miopenConvolution)
    {
        std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

#if MIOPEN_USE_GEMM
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
        {
            // Use transpose path if input ht and width <= 14 for 1x1_stride=1 convolutions OR
            // for 1x1_stride=2
            if((wei_h == 1 && wei_w == 1 && conv.pad_h == 0 && conv.pad_w == 0) &&
               ((in_h <= 14 && in_w <= 14 && conv.u == 1 && conv.v == 1) ||
                (conv.u == 2 && conv.v == 2)))
            {
                size_t workspace_req = conv.ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc);
                if(workSpace != nullptr && workSpaceSize >= workspace_req)
                {
                    MIOPEN_LOG_FUNCTION("convolution, 1x1, h14xw14 || u2xv2");

                    float time_gemm = 0;

                    transpose_NCHW2CNHW(handle,
                                        in_n,
                                        in_c,
                                        in_h,
                                        in_w,
                                        out_h,
                                        out_w,
                                        x,
                                        workSpace,
                                        0,
                                        0,
                                        conv.v,
                                        conv.u,
                                        xDesc.GetType());
                    time_gemm = handle.GetKernelTime();

                    size_t x_t_size = in_n * in_c * out_h * out_w;

                    // y = CNHW2NCHW(w * NCHW2CNHW(x))
                    GemmDescriptor gemm_desc = CreateGemmDescriptorConvCNHWFwd(wDesc, xDesc, yDesc);

                    std::string kcache_key;
                    miopenStatus_t gemm_status = miopenStatusNotInitialized;

                    if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                    {
                        // rocBLAS need a warm-up call for accurate timing
                        CallGemm(
                            handle, gemm_desc, w, 0, workSpace, 0, tmp_y.get(), 0, nullptr, false);

                        // y = CNHW2NCHW(w * NCHW2CNHW(x))
                        gemm_status = CallGemm(handle,
                                               gemm_desc,
                                               w,
                                               0,
                                               workSpace,
                                               0,
                                               tmp_y.get(),
                                               0,
                                               &kcache_key,
                                               true);
                    }
                    else
                    {
                        // y = CNHW2NCHW(w * NCHW2CNHW(x))
                        gemm_status = CallGemm(handle,
                                               gemm_desc,
                                               w,
                                               0,
                                               workSpace,
                                               0,
                                               tmp_y.get(),
                                               0,
                                               &kcache_key,
                                               false);
                    }

                    time_gemm += handle.GetKernelTime();

                    transpose_CNHW2NCHW(handle,
                                        in_n,
                                        wei_n,
                                        out_h,
                                        out_w,
                                        out_h,
                                        out_w,
                                        workSpace,
                                        tmp_y.get(),
                                        x_t_size,
                                        0,
                                        1,
                                        1,
                                        xDesc.GetType());
                    time_gemm += handle.GetKernelTime();

                    if(gemm_status == miopenStatusSuccess)
                        record.SetValues("miopenConvolutionFwdAlgoGEMM",
                                         FindDbData{"gemm",
                                                    time_gemm,
                                                    workspace_req,
                                                    kcache_key}); // Todo: gemm solver id?
                }
            }
            // 1x1_stride=1 with GEMM and zero workspace
            else if(wei_h == 1 && wei_w == 1 && conv.pad_h == 0 && conv.pad_w == 0 &&
                    (conv.u == 1 && conv.v == 1))
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1");

                // y = w * x
                GemmDescriptor gemm_desc =
                    CreateGemmStridedBatchedDescriptorConv1x1Fwd(wDesc, xDesc, yDesc);

                std::string kcache_key;
                miopenStatus_t gemm_status = miopenStatusNotInitialized;

                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need extra warm-up call for accurate timing
                    CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, x, 0, tmp_y.get(), 0, nullptr, false);

                    // y = w * x
                    gemm_status = CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, x, 0, tmp_y.get(), 0, &kcache_key, true);
                }
                else
                {
                    // y = w * x
                    gemm_status = CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, x, 0, tmp_y.get(), 0, &kcache_key, false);
                }

                float time_gemm = handle.GetKernelTime();

                if(gemm_status == miopenStatusSuccess)
                    record.SetValues(
                        "miopenConvolutionFwdAlgoGEMM",
                        FindDbData{"gemm", time_gemm, 0, kcache_key}); // Todo: gemm solver id?
            }
            // if not 1x1
            else if(workSpace != nullptr &&
                    workSpaceSize >= conv.ForwardGetWorkSpaceSizeGEMM(handle, wDesc, yDesc))
            {
                MIOPEN_LOG_FUNCTION("convolution, non 1x1");

                // y = w * Im2Col(x)
                GemmDescriptor gemm_desc = CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

                float time_im2col = 0;
                int in_offset     = 0;
                time_im2col       = Im2ColGPU(handle,
                                        xDesc.GetElementSize(),
                                        x,
                                        in_offset,
                                        in_c,
                                        in_h,
                                        in_w,
                                        wei_h,
                                        wei_w,
                                        out_h,
                                        out_w,
                                        conv.pad_h,
                                        conv.pad_w,
                                        conv.u,
                                        conv.v,
                                        conv.dilation_h,
                                        conv.dilation_w,
                                        workSpace,
                                        xDesc.GetType());

                std::string kcache_key;
                miopenStatus_t gemm_status = miopenStatusNotInitialized;

                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemm(handle,
                             gemm_desc,
                             w,
                             0,
                             workSpace,
                             0,
                             tmp_y.get(),
                             0,
                             nullptr,
                             false,
                             GemmBackend_t::miopengemm);

                    // y = w * Im2Col(x)
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           workSpace,
                                           0,
                                           tmp_y.get(),
                                           0,
                                           &kcache_key,
                                           true,
                                           GemmBackend_t::miopengemm);
                }
                else
                {
                    // y = w * Im2Col(x)
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           workSpace,
                                           0,
                                           tmp_y.get(),
                                           0,
                                           &kcache_key,
                                           false,
                                           GemmBackend_t::miopengemm);
                }

                float time_gemm = in_n * (time_im2col + handle.GetKernelTime());

                if(gemm_status == miopenStatusSuccess)
                    record.SetValues(
                        "miopenConvolutionFwdAlgoGEMM",
                        FindDbData{"gemm",
                                   time_gemm,
                                   conv.ForwardGetWorkSpaceSizeGEMM(handle, wDesc, yDesc),
                                   kcache_key}); // Todo: gemm solver id?
            }
        }
#else
        (void)workSpace;     // Suppress warning
        (void)workSpaceSize; // Suppress warning
#endif
        if(conv.dilation_h == 1 && conv.dilation_w == 1)
        {
            // Winograd algo
            WinogradKernelParams k_p;
            KernelInvoke kernel_wino;
            std::string network_config;
            std::string solver_id;
            if(conv.FindWinogradKernel(
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
                    << " conv.pad_h=" << conv.pad_h << " conv.pad_w=" << conv.pad_w << " out_H=" << out_H << " out_W=" << out_W); // clang-format on
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
                                conv.pad_h,
                                conv.pad_w,
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

            { // Direct algo
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
                                             << best
                                             << ", workspce_sz = "
                                             << sol.workspce_sz);
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

    const auto find_db_path = GetFindDbPath() + "/" + handle.GetDbPathFilename() + ".cd.fdb.txt";
    (void)find_db_path;
    auto record =
        boost::optional<DbRecord>{boost::none}; // Db{find_db_path, false}.FindRecord(problem);
    auto loaded = record.is_initialized();

    if(!loaded)
    {
        record = DbRecord(problem);
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
                        *record);
    }

    std::vector<PerfField> perf_db;
    std::string network_config;
    std::ignore = problem.mloBuildConf_Key(network_config);

    for(const auto& pair : record->As<FindDbData>())
    {
        perf_db.push_back({pair.first, pair.second.time, pair.second.workspace});

        if(loaded && (pair.second.kchache_key == FindDbData::GetUnusedKCacheKey() ||
                      !handle.HasKernel(pair.first, pair.second.kchache_key)))
        {
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
                            *record);
            loaded = false;
        }
    }

    if(IsEnabled(MIOPEN_DEBUG_ENABLE_FIND_DB{}) && !loaded)
    {
        if(!Db{find_db_path, false}.StoreRecord(record.get()))
            MIOPEN_LOG_W("Failed to store record to find-db at <" << find_db_path << ">");
    }

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
    if(xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != wDesc.GetType())
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

    if(mode == miopenConvolution)
    {
        if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1])
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
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
                << " pad_h=" << pad_h << " pad_w=" << pad_w << " out_H=" << out_H << " out_W=" << out_W); // clang-format on
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
                       pad_h,
                       pad_w,
                       out_H,
                       out_W);
            }
            else
            {
                kernel(N, C, H, W, K, n_groups, flags, reserved, x, w, y, return_addr);
            }
        }
        break;

        case miopenConvolutionFwdAlgoGEMM: {
#if MIOPEN_USE_GEMM
            if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
            {
                MIOPEN_THROW("GEMM convolution is disabled");
            }
            int in_n, in_c, in_h, in_w;
            std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

            int wei_n, wei_h, wei_w;
            std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

            int out_h, out_w;
            std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

            // Use transpose path if input ht and width <= 14 for 1x1_stride=1 convolutions OR for
            // 1x1_stride=2
            if((wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0) &&
               ((in_h <= 14 && in_w <= 14 && u == 1 && v == 1) || (u == 2 && v == 2)))
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1, h14xw14 || u2xv2");

                assert(workSpace != nullptr &&
                       workSpaceSize >= ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc));

                float t1 = 0;
                transpose_NCHW2CNHW(handle,
                                    in_n,
                                    in_c,
                                    in_h,
                                    in_w,
                                    out_h,
                                    out_w,
                                    x,
                                    workSpace,
                                    0,
                                    0,
                                    v,
                                    u,
                                    xDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 = handle.GetKernelTime();

                size_t x_t_size = in_n * in_c * out_h * out_w;

                // y = CNHW2NCHW(w * NCHW2CNHW(x))
                GemmDescriptor gemm_desc = CreateGemmDescriptorConvCNHWFwd(wDesc, xDesc, yDesc);

                // y = CNHW2NCHW(w * NCHW2CNHW(x))
                CallGemm(
                    handle, gemm_desc, w, 0, workSpace, 0, workSpace, x_t_size, nullptr, false);

                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                transpose_CNHW2NCHW(handle,
                                    in_n,
                                    wei_n,
                                    out_h,
                                    out_w,
                                    out_h,
                                    out_w,
                                    workSpace,
                                    y,
                                    x_t_size,
                                    0,
                                    1,
                                    1,
                                    xDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(t1);
                }
            }
            else if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 1 && v == 1))
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1");

                // y = w * x
                GemmDescriptor gemm_desc =
                    CreateGemmStridedBatchedDescriptorConv1x1Fwd(wDesc, xDesc, yDesc);

                // y = w * x
                CallGemmStridedBatched(handle, gemm_desc, w, 0, x, 0, y, 0, nullptr, false);
            }
            // if not 1x1
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, non 1x1");

                assert(workSpace != nullptr &&
                       workSpaceSize >= ForwardGetWorkSpaceSizeGEMM(handle, wDesc, yDesc));

                // y = w * Im2Col(x)
                GemmDescriptor gemm_desc = CreateGemmDescriptorConvFwd(wDesc, xDesc, yDesc);

                float time_0 = 0;
                float t1     = 0;
                for(int i = 0; i < in_n; i++)
                {
                    int out_offset = i * wei_n * out_h * out_w;
                    int in_offset  = i * in_c * in_h * in_w;
                    Im2ColGPU(handle,
                              xDesc.GetElementSize(),
                              x,
                              in_offset,
                              in_c,
                              in_h,
                              in_w,
                              wei_h,
                              wei_w,
                              out_h,
                              out_w,
                              pad_h,
                              pad_w,
                              u,
                              v,
                              dilation_h,
                              dilation_w,
                              workSpace,
                              xDesc.GetType());

                    if(handle.IsProfilingEnabled())
                        t1 = handle.GetKernelTime();

                    // y = w * Im2Col(x)
                    CallGemm(handle,
                             gemm_desc,
                             w,
                             0,
                             workSpace,
                             0,
                             y,
                             out_offset,
                             nullptr,
                             false,
                             GemmBackend_t::miopengemm);

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
#else
            MIOPEN_THROW("GEMM is not supported");
#endif
        }
        break;

        case miopenConvolutionFwdAlgoFFT:
        {
            size_t workspace_fft = ForwardGetWorkSpaceSizeFFT(wDesc, xDesc, yDesc);
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
    else if(mode == miopenTranspose)
    {
        if(xDesc.GetLengths()[1] != wDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

        // GEMM based
        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

        int wei_n, wei_h, wei_w;
        std::tie(std::ignore, wei_n, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

        int out_h, out_w;
        std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

        if(wei_h != 1 || wei_w != 1 || u != 1 || v != 1 || pad_h != 0 || pad_w != 0)
        {
            assert(workSpace != nullptr &&
                   workSpaceSize >= BackwardDataGetWorkSpaceSizeGEMM(handle, wDesc, xDesc));
        }

        std::string network_config;

#if MIOPEN_USE_MIOPENGEMM
        CreateGemmGeometryConvBwdData(xDesc, wDesc, yDesc, true, network_config);
        GemmGeometry gg =
            GetGemmGeometry(handle, "miopenConvolutionBwdDataAlgoGEMM", network_config);

        float time_0 = 0;
        float t1     = 0;
        for(int i = 0; i < in_n; i++)
        {
            int out_offset = i * wei_n * out_h * out_w;
            if(wei_h != 1 || wei_w != 1 || v != 1 || u != 1 || pad_h != 0 || pad_w != 0)
            {
                MIOPEN_LOG_FUNCTION("transppose, non 1x1");

                int in_offset = i * in_c * in_h * in_w;

                gg.RunGemm(handle, w, x, workSpace, 0, in_offset, 0);

                if(handle.IsProfilingEnabled())
                    t1 = handle.GetKernelTime();

                Col2ImGPU(handle,
                          workSpace,
                          in_h,
                          in_w,
                          wei_h,
                          wei_w,
                          pad_h,
                          pad_w,
                          u,
                          v,
                          dilation_h,
                          dilation_w,
                          wei_n,
                          out_h,
                          out_w,
                          y,
                          out_offset,
                          xDesc.GetType());

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
            else if(wei_h == 1 && wei_w == 1 && v == 1 && u == 1 && pad_h == 0 && pad_w == 0)
            {
                MIOPEN_LOG_FUNCTION("transppose, 1x1");

                int in_offset = i * in_c * in_h * in_w;
                gg.RunGemm(handle, w, x, y, 0, in_offset, out_offset);
                if(handle.IsProfilingEnabled())
                {
                    if(i == in_n - 1)
                        handle.AccumKernelTime(time_0);
                    time_0 += handle.GetKernelTime();
                }
            }
        }
#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(mode == miopenGroupConv || mode == miopenDepthwise)
    {

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
                float elapsed = 0;
                if(num_kernels == 1)
                {
                    kernel(x, w, y, as_float(0.0f));
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

        case miopenConvolutionFwdAlgoGEMM:
#if MIOPEN_USE_GEMM
        {
            int in_n, in_c, in_h, in_w;
            std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

            int wei_n, wei_c, wei_h, wei_w;
            std::tie(wei_n, wei_c, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

            int out_h, out_w;
            std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

            std::tie(wei_n, wei_c, wei_h, wei_w) = tien<4>(wDesc.GetLengths());
            if(in_c % group_count != 0 || wei_n % group_count != 0 || group_count > in_c ||
               group_count > wei_n || group_count < 1 ||
               (mode == miopenDepthwise && group_count != in_c))
                MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
            if(in_c / group_count != wei_c || (mode == miopenDepthwise && wei_c != 1))
                MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");

            // Use transpose path if input ht and width <= 14 for 1x1_stride=1 convolutions OR for
            // 1x1_stride=2
            if((wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0) &&
               ((in_h <= 14 && in_w <= 14 && u == 1 && v == 1) || (u == 2 && v == 2)))
            {
                assert(workSpace != nullptr &&
                       workSpaceSize >= ForwardGetWorkSpaceSizeGEMMTranspose(xDesc, yDesc));

                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvCNHWFwd(wDesc, xDesc, yDesc, group_count);

                float t1 = 0;
                transpose_NCHW2CNHW(handle,
                                    in_n,
                                    in_c,
                                    in_h,
                                    in_w,
                                    out_h,
                                    out_w,
                                    x,
                                    workSpace,
                                    0,
                                    0,
                                    v,
                                    u,
                                    xDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 = handle.GetKernelTime();

                size_t x_t_size = in_n * in_c * out_h * out_w;
                CallGemmStridedBatched(
                    handle, gemm_desc, w, 0, workSpace, 0, workSpace, x_t_size, nullptr, false);
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                transpose_CNHW2NCHW(handle,
                                    in_n,
                                    wei_n,
                                    out_h,
                                    out_w,
                                    out_h,
                                    out_w,
                                    workSpace,
                                    y,
                                    x_t_size,
                                    0,
                                    1,
                                    1,
                                    xDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(t1);
                }
            }
            else if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && u == 1 && v == 1)
            {
                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, group_count);
                float time_0 = 0;

                for(int i = 0; i < in_n; i++)
                {
                    size_t out_offset = i * wei_n * out_h * out_w;
                    size_t in_offset  = i * in_c * in_h * in_w;
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
                assert(workSpace != nullptr &&
                       workSpaceSize >= ForwardGetWorkSpaceSizeGEMM(handle, wDesc, yDesc));

                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvFwd(wDesc, xDesc, yDesc, group_count);

                float time_0 = 0;
                float t1     = 0;
                for(int i = 0; i < in_n; i++)
                {
                    size_t in_offset = i * in_c * in_h * in_w;
                    Im2ColGPU(handle,
                              xDesc.GetElementSize(),
                              x,
                              in_offset,
                              in_c,
                              in_h,
                              in_w,
                              wei_h,
                              wei_w,
                              out_h,
                              out_w,
                              pad_h,
                              pad_w,
                              u,
                              v,
                              dilation_h,
                              dilation_w,
                              workSpace,
                              xDesc.GetType());
                    if(handle.IsProfilingEnabled())
                        t1 += handle.GetKernelTime();

                    size_t out_offset = i * wei_n * out_h * out_w;
                    CallGemmStridedBatched(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           workSpace,
                                           0,
                                           y,
                                           out_offset,
                                           nullptr,
                                           false,
                                           GemmBackend_t::miopengemm);

                    // Update times for both the kernels
                    if(handle.IsProfilingEnabled())
                    {
                        if(i == in_n - 1)
                            handle.AccumKernelTime(t1 + time_0);
                        time_0 += handle.GetKernelTime();
                    }
                }
            }
        }
        break;
#else
            MIOPEN_THROW("GEMM is not supported");
#endif

        case miopenConvolutionFwdAlgoWinograd:
            MIOPEN_THROW("Winograd is not supported for group conv");
        case miopenConvolutionFwdAlgoFFT: MIOPEN_THROW("FFT is not supported for group conv");
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

    *returnedAlgoCount = 0;

    // create a dummy buffer for use as output for the kernel calls
    // because kernels are called purely for timing purposes
    auto tmp_dx = handle.Create(dxDesc.GetElementSize() * GetTypeSize(dxDesc.GetType()));

    AutoEnableProfiling enableProfiling{handle};

    // < algorith_name, <time, workspace_size> >
    std::vector<PerfField> perf_db;

    // GEMM based
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

    int wei_n, wei_c, wei_h, wei_w;

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    std::string network_config;

    if(mode == miopenTranspose)
    {
#if MIOPEN_USE_MIOPENGEMM
        // GEMM based
        std::tie(std::ignore, wei_n, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

        if(dyDesc.GetType() == miopenFloat)
        {
            size_t workspace_req = ForwardGetWorkSpaceSizeGEMM(handle, wDesc, dxDesc);
            float time_gemm      = 0;
            GemmGeometry gg =
                CreateGemmGeometryTranBwdData(dyDesc, wDesc, dxDesc, true, network_config);

            // 1x1 does not require im2col or workspace
            if(wei_h == 1 && wei_w == 1 && v == 1 && u == 1 && pad_h == 0 && pad_w == 0)
            {
                MIOPEN_LOG_FUNCTION("transppose, 1x1");

                gg.FindSolution(.003, handle, w, dy, tmp_dx.get(), false);
                gg.RunGemm(handle, w, dy, tmp_dx.get(), 0, 0, 0);

                time_gemm = in_n * handle.GetKernelTime();
                perf_db.push_back(PerfField{"miopenTransposeBwdDataAlgoGEMM", time_gemm, 0});
            }

            // if not 1x1
            else if(workSpace != nullptr && workSpaceSize >= workspace_req)
            {
                MIOPEN_LOG_FUNCTION("transppose, non 1x1");

                float time_im2col = 0;
                size_t out_offset = 0;
                time_im2col       = Im2ColGPU(handle,
                                        dyDesc.GetElementSize(),
                                        dy,
                                        out_offset,
                                        wei_n,
                                        out_h,
                                        out_w,
                                        wei_h,
                                        wei_w,
                                        in_h,
                                        in_w,
                                        pad_h,
                                        pad_w,
                                        u,
                                        v,
                                        dilation_h,
                                        dilation_w,
                                        workSpace,
                                        dyDesc.GetType());

                gg.FindSolution(.003, handle, w, workSpace, tmp_dx.get(), false);
                gg.RunGemm(handle, w, workSpace, tmp_dx.get(), 0, 0, 0);
                time_gemm = in_n * (time_im2col + handle.GetKernelTime());
                perf_db.push_back(
                    PerfField{"miopenTransposeBwdDataAlgoGEMM", time_gemm, workspace_req});
            }
        }
#else
        (void)workSpace;     // Suppress warning
        (void)workSpaceSize; // Suppress warning
#endif
    }
    else if(mode == miopenGroupConv || mode == miopenDepthwise)
    {

        std::tie(wei_n, wei_c, wei_h, wei_w) = tien<4>(wDesc.GetLengths());
        if(in_c % group_count != 0 || wei_n % group_count != 0 || group_count > in_c ||
           group_count > wei_n || group_count < 1 ||
           (mode == miopenDepthwise && group_count != in_c))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
        if(in_c / group_count != wei_c || (mode == miopenDepthwise && wei_c != 1))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");

#if MIOPEN_USE_GEMM
        { // GEMM algo
            float time_gemm = 0;

            // 1x1 does not require col2im or workspace
            if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 2 && v == 2) &&
               workSpace != nullptr &&
               workSpaceSize >= BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc))
            {

                // Initialization required for upsampling in bwd direction
                float zero = 0.f;
                SetTensor(handle, dxDesc, tmp_dx.get(), &zero);
                time_gemm = handle.GetKernelTime();

                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvCNHWBwdData(wDesc, dyDesc, dxDesc, group_count);

                transpose_NCHW2CNHW(handle,
                                    in_n,
                                    wei_n,
                                    out_h,
                                    out_w,
                                    out_h,
                                    out_w,
                                    dy,
                                    workSpace,
                                    0,
                                    0,
                                    1,
                                    1,
                                    dyDesc.GetType());
                time_gemm += handle.GetKernelTime();

                miopenStatus_t gemm_status = miopenStatusNotInitialized;
                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
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

                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         workSpace,
                                                         0,
                                                         workSpace,
                                                         dyDesc.GetElementSize(),
                                                         nullptr,
                                                         true);
                }
                else
                {
                    gemm_status = CallGemmStridedBatched(handle,
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

                time_gemm += (handle.GetKernelTime());

                transpose_CNHW2NCHW(handle,
                                    in_n,
                                    in_c,
                                    out_h,
                                    out_w,
                                    in_h,
                                    in_w,
                                    workSpace,
                                    tmp_dx.get(),
                                    dyDesc.GetElementSize(),
                                    0,
                                    u,
                                    v,
                                    dyDesc.GetType());
                time_gemm += handle.GetKernelTime();
                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdDataAlgoGEMM",
                                  time_gemm,
                                  BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc)});
            }
            // 1x1_stride=1 convolutions use GEMM and zero workspace
            else if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 1 && v == 1))
            {
                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count);

                miopenStatus_t gemm_status = miopenStatusNotInitialized;
                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, dy, 0, tmp_dx.get(), 0, nullptr, false);

                    gemm_status = CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, dy, 0, tmp_dx.get(), 0, nullptr, true);
                }
                else
                {
                    gemm_status = CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, dy, 0, tmp_dx.get(), 0, nullptr, false);
                }

                time_gemm = in_n * handle.GetKernelTime();

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(PerfField{"miopenConvolutionBwdDataAlgoGEMM", time_gemm, 0});
            }
            // if not 1x1
            else if(workSpace != nullptr &&
                    workSpaceSize >=
                        (group_count * BackwardDataGetWorkSpaceSizeGEMM(handle, wDesc, dyDesc)))
            {
                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count);

                float time_col2im = 0;
                size_t in_offset  = 0;

                miopenStatus_t gemm_status = miopenStatusNotInitialized;
                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemmStridedBatched(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           dy,
                                           0,
                                           workSpace,
                                           0,
                                           nullptr,
                                           false,
                                           GemmBackend_t::miopengemm);

                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         dy,
                                                         0,
                                                         workSpace,
                                                         0,
                                                         nullptr,
                                                         true,
                                                         GemmBackend_t::miopengemm);
                }
                else
                {
                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         dy,
                                                         0,
                                                         workSpace,
                                                         0,
                                                         nullptr,
                                                         false,
                                                         GemmBackend_t::miopengemm);
                }

                time_gemm   = (in_n * handle.GetKernelTime());
                time_col2im = Col2ImGPU(handle,
                                        workSpace,
                                        out_h,
                                        out_w,
                                        wei_h,
                                        wei_w,
                                        pad_h,
                                        pad_w,
                                        u,
                                        v,
                                        dilation_h,
                                        dilation_w,
                                        in_c,
                                        in_h,
                                        in_w,
                                        tmp_dx.get(),
                                        in_offset,
                                        dyDesc.GetType());

                time_gemm += (in_n * time_col2im);

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdDataAlgoGEMM",
                                  time_gemm,
                                  BackwardDataGetWorkSpaceSizeGEMM(handle, wDesc, dyDesc)});
            }
        }
#else
        (void)workSpace;     // Suppress warning
        (void)workSpaceSize; // Suppress warning
#endif

        if(dilation_h == 1 && dilation_w == 1)
        {
            { // Direct algo
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
                                             << best
                                             << ", workspce_sz = "
                                             << sol.workspce_sz);
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
                    perf_db.push_back(PerfField{algorithm_name, best, selected.workspce_sz});
                }
            }
        }
    }
    else if(mode == miopenConvolution)
    {
        if(dilation_h == 1 && dilation_w == 1)
        {

            // Winograd algo
            WinogradKernelParams k_p;
            KernelInvoke kernel_wino;
            std::string solver;
            if(FindWinogradKernel(handle, dxDesc, wDesc, dyDesc, k_p, kernel_wino, solver, 0) == 0)
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
                perf_db.push_back(PerfField{"miopenConvolutionBwdDataAlgoWinograd", time_wino, 0});
            }

            { // Direct algo
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
                                             << best
                                             << ", workspce_sz = "
                                             << sol.workspce_sz);
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
                    perf_db.push_back(PerfField{algorithm_name, best, selected.workspce_sz});
                }
            }

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
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdDataAlgoFFT", time_fft, workspace_fft});
                }
            }
        }

#if MIOPEN_USE_GEMM
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
        {
            std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

            // 1x1 does not require col2im
            if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 2 && v == 2) &&
               workSpace != nullptr &&
               workSpaceSize >= BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc))
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1 u2xv2");

                float time_gemm = 0;

                // Initialization required for upsampling in bwd direction
                float zero = 0.f;
                SetTensor(handle, dxDesc, tmp_dx.get(), &zero);
                time_gemm = handle.GetKernelTime();

                transpose_NCHW2CNHW(handle,
                                    in_n,
                                    wei_n,
                                    out_h,
                                    out_w,
                                    out_h,
                                    out_w,
                                    dy,
                                    workSpace,
                                    0,
                                    0,
                                    1,
                                    1,
                                    dyDesc.GetType());
                time_gemm += handle.GetKernelTime();

                // dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorConvCNHWBwdData(wDesc, dyDesc, dxDesc);

                miopenStatus_t gemm_status = miopenStatusNotInitialized;

                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemm(
                        handle, gemm_desc, w, 0, workSpace, 0, tmp_dx.get(), 0, nullptr, false);

                    // dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
                    gemm_status = CallGemm(
                        handle, gemm_desc, w, 0, workSpace, 0, tmp_dx.get(), 0, nullptr, true);
                }
                else
                {
                    // dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
                    gemm_status = CallGemm(
                        handle, gemm_desc, w, 0, workSpace, 0, tmp_dx.get(), 0, nullptr, false);
                }

                time_gemm += handle.GetKernelTime();

                transpose_CNHW2NCHW(handle,
                                    in_n,
                                    in_c,
                                    out_h,
                                    out_w,
                                    in_h,
                                    in_w,
                                    workSpace,
                                    tmp_dx.get(),
                                    dyDesc.GetElementSize(),
                                    0,
                                    u,
                                    v,
                                    dyDesc.GetType());
                time_gemm += handle.GetKernelTime();

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdDataAlgoGEMM",
                                  time_gemm,
                                  BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc)});
            }
            // 1x1_stride=1 convolutions use GEMM and zero workspace
            else if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 1 && v == 1))
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1");

                // dx = transpose(w) * dy
                GemmDescriptor gemm_desc =
                    CreateGemmStridedBatchedDescriptorConv1x1BwdData(wDesc, dyDesc, dxDesc);

                miopenStatus_t gemm_status = miopenStatusNotInitialized;

                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, dy, 0, tmp_dx.get(), 0, nullptr, false);

                    // dx = transpose(w) * dy
                    gemm_status = CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, dy, 0, tmp_dx.get(), 0, nullptr, true);
                }
                else
                {
                    // dx = transpose(w) * dy
                    gemm_status = CallGemmStridedBatched(
                        handle, gemm_desc, w, 0, dy, 0, tmp_dx.get(), 0, nullptr, false);
                }

                float time_gemm = handle.GetKernelTime();

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(PerfField{"miopenConvolutionBwdDataAlgoGEMM", time_gemm, 0});
            }
            // if not 1x1
            else if(workSpace != nullptr &&
                    workSpaceSize >= BackwardDataGetWorkSpaceSizeGEMM(handle, wDesc, dyDesc))
            {
                MIOPEN_LOG_FUNCTION("convolution, non 1x1");

                // dx = transpose(w) * dy
                GemmDescriptor gemm_desc = CreateGemmDescriptorConvBwdData(wDesc, dyDesc, dxDesc);

                float time_col2im = 0;
                size_t in_offset  = 0;

                miopenStatus_t gemm_status = miopenStatusNotInitialized;

                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemm(handle,
                             gemm_desc,
                             w,
                             0,
                             dy,
                             0,
                             workSpace,
                             0,
                             nullptr,
                             false,
                             GemmBackend_t::miopengemm);

                    // dx = transpose(w) * dy
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           dy,
                                           0,
                                           workSpace,
                                           0,
                                           nullptr,
                                           true,
                                           GemmBackend_t::miopengemm);
                }
                else
                {
                    // dx = transpose(w) * dy
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           dy,
                                           0,
                                           workSpace,
                                           0,
                                           nullptr,
                                           false,
                                           GemmBackend_t::miopengemm);
                }

                float time_gemm = in_n * handle.GetKernelTime();
                time_col2im     = Col2ImGPU(handle,
                                        workSpace,
                                        out_h,
                                        out_w,
                                        wei_h,
                                        wei_w,
                                        pad_h,
                                        pad_w,
                                        u,
                                        v,
                                        dilation_h,
                                        dilation_w,
                                        in_c,
                                        in_h,
                                        in_w,
                                        tmp_dx.get(),
                                        in_offset,
                                        dyDesc.GetType());

                time_gemm += in_n * time_col2im;

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdDataAlgoGEMM",
                                  time_gemm,
                                  BackwardDataGetWorkSpaceSizeGEMM(handle, wDesc, dyDesc)});
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

    if(mode == miopenConvolution)
    {
        if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
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

        case miopenConvolutionBwdDataAlgoGEMM: {
#if MIOPEN_USE_GEMM
            if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
            {
                MIOPEN_THROW("GEMM convolution is disabled");
            }
            int in_n, in_c, in_h, in_w;
            std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

            int wei_n, wei_h, wei_w;
            std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

            int out_h, out_w;
            std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

            if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 2 && v == 2))
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1, u2xv2");

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
                                    wei_n,
                                    out_h,
                                    out_w,
                                    out_h,
                                    out_w,
                                    dy,
                                    workSpace,
                                    0,
                                    0,
                                    1,
                                    1,
                                    dyDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

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

                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                transpose_CNHW2NCHW(handle,
                                    in_n,
                                    in_c,
                                    out_h,
                                    out_w,
                                    in_h,
                                    in_w,
                                    workSpace,
                                    dx,
                                    dyDesc.GetElementSize(),
                                    0,
                                    u,
                                    v,
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
            else if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 1 && v == 1))
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1");

                // dx = transpose(w) * dy
                GemmDescriptor gemm_desc =
                    CreateGemmStridedBatchedDescriptorConv1x1BwdData(wDesc, dyDesc, dxDesc);

                // dx = transpose(w) * dy
                CallGemmStridedBatched(handle, gemm_desc, w, 0, dy, 0, dx, 0, nullptr, false);
            }
            // if not 1x1
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, non 1x1");

                assert(workSpace != nullptr &&
                       workSpaceSize >= BackwardDataGetWorkSpaceSizeGEMM(handle, wDesc, dyDesc));

                // dx = transpose(w) * dy
                GemmDescriptor gemm_desc = CreateGemmDescriptorConvBwdData(wDesc, dyDesc, dxDesc);

                handle.ResetKernelTime();

                float time_0 = 0;
                float t1     = 0;
                for(int i = 0; i < in_n; i++)
                {
                    int out_offset   = i * wei_n * out_h * out_w;
                    size_t in_offset = i * in_c * in_h * in_w;

                    // dx = transpose(w) * dy
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
                              workSpace,
                              out_h,
                              out_w,
                              wei_h,
                              wei_w,
                              pad_h,
                              pad_w,
                              u,
                              v,
                              dilation_h,
                              dilation_w,
                              in_c,
                              in_h,
                              in_w,
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
#else
            MIOPEN_THROW("GEMM is not supported");
#endif
        }
        break;

        case miopenConvolutionBwdDataAlgoFFT:
        {
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
    else if(mode == miopenTranspose)
    {
        if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[1])
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

        int wei_n, wei_h, wei_w;
        std::tie(std::ignore, wei_n, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

        int out_h, out_w;
        std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

        if(wei_h != 1 || wei_w != 1 || u != 1 || v != 1 || pad_h != 0 || pad_w != 0)
        {
            assert(workSpace != nullptr &&
                   workSpaceSize >= ForwardGetWorkSpaceSizeGEMM(handle, wDesc, dxDesc));
        }

        std::string network_config;
#if MIOPEN_USE_MIOPENGEMM
        CreateGemmGeometryTranBwdData(dyDesc, wDesc, dxDesc, true, network_config);
        GemmGeometry gg = GetGemmGeometry(handle, "miopenTransposeBwdDataAlgoGEMM", network_config);

        float time_0 = 0;
        float t1     = 0;
        for(int i = 0; i < in_n; i++)
        {
            int in_offset = i * in_c * in_h * in_w;
            if(wei_h != 1 || wei_w != 1 || v != 1 || u != 1 || pad_h != 0 || pad_w != 0)
            {
                MIOPEN_LOG_FUNCTION("transpose, non 1x1");

                size_t out_offset = i * wei_n * out_h * out_w;
                Im2ColGPU(handle,
                          dyDesc.GetElementSize(),
                          dy,
                          out_offset,
                          wei_n,
                          out_h,
                          out_w,
                          wei_h,
                          wei_w,
                          in_h,
                          in_w,
                          pad_h,
                          pad_w,
                          u,
                          v,
                          dilation_h,
                          dilation_w,
                          workSpace,
                          dyDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 = handle.GetKernelTime();

                gg.RunGemm(handle, w, workSpace, dx, 0, 0, in_offset);

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
            else if(wei_h == 1 && wei_w == 1 && v == 1 && u == 1 && pad_h == 0 && pad_w == 0)
            {
                MIOPEN_LOG_FUNCTION("transpose, 1x1");

                int out_offset = i * wei_n * out_h * out_w;
                gg.RunGemm(handle, w, dy, dx, 0, out_offset, in_offset);
                if(handle.IsProfilingEnabled())
                {
                    if(i == in_n - 1)
                        handle.AccumKernelTime(time_0);
                    time_0 += handle.GetKernelTime();
                }
            }
        }
#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(mode == miopenGroupConv || mode == miopenDepthwise)
    {
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
                float t1          = 0;
                float padding_val = 0;
                kernels[0](dy, w, dx, as_float(padding_val));
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(t1);
                }
            });
            break;
        }

        case miopenConvolutionBwdDataAlgoGEMM:
#if MIOPEN_USE_GEMM
        {
            int in_n, in_c, in_h, in_w;
            std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

            int wei_n, wei_c, wei_h, wei_w;
            std::tie(wei_n, wei_c, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

            int out_h, out_w;
            std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

            std::tie(wei_n, wei_c, wei_h, wei_w) = tien<4>(wDesc.GetLengths());
            if(in_c % group_count != 0 || wei_n % group_count != 0 || group_count > in_c ||
               group_count > wei_n || group_count < 1 ||
               (mode == miopenDepthwise && group_count != in_c))
                MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
            if(in_c / group_count != wei_c || (mode == miopenDepthwise && wei_c != 1))
                MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");

            if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 2 && v == 2))
            {
                float t1 = 0;
                // Initialization required for upsampling in bwd direction
                float zero = 0.f;
                SetTensor(handle, dxDesc, dx, &zero);
                if(handle.IsProfilingEnabled())
                    t1 = handle.GetKernelTime();

                assert(workSpace != nullptr &&
                       workSpaceSize >= BackwardDataGetWorkSpaceSizeGEMMTranspose(dyDesc, dxDesc));

                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvCNHWBwdData(wDesc, dyDesc, dxDesc, group_count);

                transpose_NCHW2CNHW(handle,
                                    in_n,
                                    wei_n,
                                    out_h,
                                    out_w,
                                    out_h,
                                    out_w,
                                    dy,
                                    workSpace,
                                    0,
                                    0,
                                    1,
                                    1,
                                    dyDesc.GetType());
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

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
                if(handle.IsProfilingEnabled())
                    t1 += handle.GetKernelTime();

                transpose_CNHW2NCHW(handle,
                                    in_n,
                                    in_c,
                                    out_h,
                                    out_w,
                                    in_h,
                                    in_w,
                                    workSpace,
                                    dx,
                                    dyDesc.GetElementSize(),
                                    0,
                                    u,
                                    v,
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
            else if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 1 && v == 1))
            {
                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count);

                float time_0 = 0;
                for(int i = 0; i < in_n; i++)
                {
                    size_t out_offset = i * wei_n * out_h * out_w;
                    size_t in_offset  = i * in_c * in_h * in_w;
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
            // if not 1x1
            else
            {
                assert(workSpace != nullptr &&
                       workSpaceSize >=
                           (group_count * BackwardDataGetWorkSpaceSizeGEMM(handle, wDesc, dyDesc)));

                GemmDescriptor gemm_desc =
                    CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count);

                handle.ResetKernelTime();

                float time_0 = 0;
                float t1     = 0;
                for(int i = 0; i < in_n; i++)
                {
                    size_t in_offset  = i * in_c * in_h * in_w;
                    size_t out_offset = i * wei_n * out_h * out_w;

                    CallGemmStridedBatched(handle,
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
                        t1 += handle.GetKernelTime();

                    Col2ImGPU(handle,
                              workSpace,
                              out_h,
                              out_w,
                              wei_h,
                              wei_w,
                              pad_h,
                              pad_w,
                              u,
                              v,
                              dilation_h,
                              dilation_w,
                              in_c,
                              in_h,
                              in_w,
                              dx,
                              in_offset,
                              dyDesc.GetType());

                    // Update times for both the kernels
                    if(handle.IsProfilingEnabled())
                    {
                        if(i == in_n - 1)
                            handle.AccumKernelTime(time_0 + t1);
                        time_0 += handle.GetKernelTime();
                    }
                }
            }
        }
        break;
#else
            MIOPEN_THROW("GEMM is not supported");
#endif

        case miopenConvolutionBwdDataAlgoWinograd:
            MIOPEN_THROW("Winograd is not supported for group conv");
        case miopenConvolutionBwdDataAlgoFFT: MIOPEN_THROW("FFT is not supported for group conv");
        case miopenTransposeBwdDataAlgoGEMM:
            MIOPEN_THROW("Transpose conv is not supported for group conv");
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

    *returnedAlgoCount = 0;

    // create a dummy buffer for use as output for the kernel calls
    // because kernels are called purely for timing purposes
    auto tmp_dw = handle.Create(dwDesc.GetElementSize() * GetTypeSize(dwDesc.GetType()));

    AutoEnableProfiling enableProfiling{handle};

    // < algorith_name, <time, workspace_size> >
    std::vector<PerfField> perf_db;

    // GEMM based
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_c, wei_h, wei_w;

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    std::string network_config;

    if(mode == miopenTranspose)
    {
#if MIOPEN_USE_MIOPENGEMM
        std::tie(std::ignore, wei_n, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());

        if(dyDesc.GetType() == miopenFloat)
        {
            GemmGeometry gg =
                CreateGemmGeometryConvBwdWeights(xDesc, dyDesc, dwDesc, false, network_config);
            std::size_t workspace_req = BackwardWeightsGetWorkSpaceSizeGEMM(handle, xDesc, dwDesc);

            float time_gemm = 0;

            // 1x1 does not require im2col or workspace
            if(wei_h == 1 && wei_w == 1 && v == 1 && u == 1 && pad_h == 0 && pad_w == 0)
            {
                MIOPEN_LOG_FUNCTION("transpose, 1x1");

                gg.FindSolution(.003, handle, dy, x, tmp_dw.get(), false);
                gg.RunGemm(handle, dy, x, tmp_dw.get(), 0, 0, 0);

                time_gemm = in_n * handle.GetKernelTime();
                perf_db.push_back(PerfField{"miopenConvolutionBwdWeightsAlgoGEMM", time_gemm, 0});
            }
            // if not 1x1
            else if(workSpace != nullptr && workSpaceSize >= workspace_req)
            {
                MIOPEN_LOG_FUNCTION("transpose, non 1x1");

                float time_im2col = 0;
                int out_offset    = 0;
                time_im2col       = Im2ColGPU(handle,
                                        dyDesc.GetElementSize(),
                                        dy,
                                        out_offset,
                                        wei_n,
                                        out_h,
                                        out_w,
                                        wei_h,
                                        wei_w,
                                        in_h,
                                        in_w,
                                        pad_h,
                                        pad_w,
                                        u,
                                        v,
                                        dilation_h,
                                        dilation_w,
                                        workSpace,
                                        dyDesc.GetType());

                gg.FindSolution(.003, handle, workSpace, x, tmp_dw.get(), false);
                gg.RunGemm(handle, workSpace, x, tmp_dw.get(), 0, 0, 0);
                time_gemm = in_n * (time_im2col + handle.GetKernelTime());
                perf_db.push_back(
                    PerfField{"miopenConvolutionBwdWeightsAlgoGEMM", time_gemm, workspace_req});
            }
        }
#else
        (void)workSpace;     // Suppress warning
        (void)workSpaceSize; // Suppress warning
                             // (void)workspace_req; // Suppress warning
#endif
    }
    else if(mode == miopenGroupConv || mode == miopenDepthwise)
    {
        std::tie(wei_n, wei_c, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());
        if(in_c % group_count != 0 || wei_n % group_count != 0 || group_count > in_c ||
           group_count > wei_n || group_count < 1 ||
           (mode == miopenDepthwise && group_count != in_c))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
        if(in_c / group_count != wei_c || (mode == miopenDepthwise && wei_c != 1))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");

#if MIOPEN_USE_GEMM
        { // GEMM algo
            GemmDescriptor gemm_desc =
                CreateGemmDescriptorGroupConvBwdWeight(dyDesc, xDesc, dwDesc, group_count);
            std::size_t workspace_req =
                group_count * BackwardWeightsGetWorkSpaceSizeGEMM(handle, dyDesc, dwDesc);
            float time_gemm = 0;

            // 1x1 does not require im2col or workspace
            if(wei_h == 1 && wei_w == 1 && v == 1 && u == 1 && pad_h == 0 && pad_w == 0)
            {
                miopenStatus_t gemm_status = miopenStatusNotInitialized;
                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemmStridedBatched(handle,
                                           gemm_desc,
                                           dy,
                                           0,
                                           x,
                                           0,
                                           tmp_dw.get(),
                                           0,
                                           nullptr,
                                           false,
                                           GemmBackend_t::miopengemm);

                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         dy,
                                                         0,
                                                         x,
                                                         0,
                                                         tmp_dw.get(),
                                                         0,
                                                         nullptr,
                                                         true,
                                                         GemmBackend_t::miopengemm);
                }
                else
                {
                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         dy,
                                                         0,
                                                         x,
                                                         0,
                                                         tmp_dw.get(),
                                                         0,
                                                         nullptr,
                                                         false,
                                                         GemmBackend_t::miopengemm);
                }

                time_gemm = in_n * handle.GetKernelTime();

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdWeightsAlgoGEMM", time_gemm, 0});
            }
            // if not 1x1
            else if(workSpace != nullptr && workSpaceSize >= workspace_req)
            {
                float time_im2col = 0;
                size_t in_offset  = 0;
                time_im2col       = Im2ColGPU(handle,
                                        xDesc.GetElementSize(),
                                        x,
                                        in_offset,
                                        in_c,
                                        in_h,
                                        in_w,
                                        wei_h,
                                        wei_w,
                                        out_h,
                                        out_w,
                                        pad_h,
                                        pad_w,
                                        u,
                                        v,
                                        dilation_h,
                                        dilation_w,
                                        workSpace,
                                        dyDesc.GetType());

                miopenStatus_t gemm_status = miopenStatusNotInitialized;
                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemmStridedBatched(handle,
                                           gemm_desc,
                                           dy,
                                           0,
                                           workSpace,
                                           0,
                                           tmp_dw.get(),
                                           0,
                                           nullptr,
                                           false,
                                           GemmBackend_t::miopengemm);

                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         dy,
                                                         0,
                                                         workSpace,
                                                         0,
                                                         tmp_dw.get(),
                                                         0,
                                                         nullptr,
                                                         true,
                                                         GemmBackend_t::miopengemm);
                }
                else
                {
                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         dy,
                                                         0,
                                                         workSpace,
                                                         0,
                                                         tmp_dw.get(),
                                                         0,
                                                         nullptr,
                                                         false,
                                                         GemmBackend_t::miopengemm);
                }

                time_gemm = in_n * (time_im2col + handle.GetKernelTime());

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdWeightsAlgoGEMM", time_gemm, workspace_req});
            }
        }
#endif
        // direct convolution for groups
        {

            if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
            {
                mlo_construct_BwdWrW2D construct_params(
                    xDesc, dwDesc, dyDesc, *this, 0); // backward with regards to weights
                construct_params.setDoSearch(exhaustiveSearch);
                construct_params.setStream(&handle);

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
                                         << best
                                         << ", workspce_sz = "
                                         << sol.workspce_sz);
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
                    perf_db.push_back(PerfField{algorithm_name, best, selected.workspce_sz});
                }
            }
        }
    }
    else if(mode == miopenConvolution)
    {
#if MIOPEN_USE_GEMM
        if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
        {
            std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());

            // if not 1x1
            if((wei_h != 1 || wei_w != 1 || pad_h != 0 || pad_w != 0 || u != 1 || v != 1) &&
               (workSpace != nullptr &&
                workSpaceSize >= BackwardWeightsGetWorkSpaceSizeGEMM(handle, dyDesc, dwDesc)))
            {
                MIOPEN_LOG_FUNCTION("convolution, non 1x1");

                // dw = dy * transpose(Im2Col(x))
                GemmDescriptor gemm_desc = CreateGemmDescriptorConvBwdWeight(dyDesc, xDesc, dwDesc);

                float time_im2col = 0;
                int in_offset     = 0;
                time_im2col       = Im2ColGPU(handle,
                                        xDesc.GetElementSize(),
                                        x,
                                        in_offset,
                                        in_c,
                                        in_h,
                                        in_w,
                                        wei_h,
                                        wei_w,
                                        out_h,
                                        out_w,
                                        pad_h,
                                        pad_w,
                                        u,
                                        v,
                                        dilation_h,
                                        dilation_w,
                                        workSpace,
                                        dyDesc.GetType());

                miopenStatus_t gemm_status = miopenStatusNotInitialized;

                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemm(handle,
                             gemm_desc,
                             dy,
                             0,
                             workSpace,
                             0,
                             tmp_dw.get(),
                             0,
                             nullptr,
                             false,
                             GemmBackend_t::miopengemm);

                    // dw = dy * transpose(Im2Col(x))
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           dy,
                                           0,
                                           workSpace,
                                           0,
                                           tmp_dw.get(),
                                           0,
                                           nullptr,
                                           true,
                                           GemmBackend_t::miopengemm);
                }
                else
                {
                    // dw = dy * transpose(Im2Col(x))
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           dy,
                                           0,
                                           workSpace,
                                           0,
                                           tmp_dw.get(),
                                           0,
                                           nullptr,
                                           false,
                                           GemmBackend_t::miopengemm);
                }

                float time_gemm = in_n * (time_im2col + handle.GetKernelTime());

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdWeightsAlgoGEMM",
                                  time_gemm,
                                  BackwardWeightsGetWorkSpaceSizeGEMM(handle, dyDesc, dwDesc)});
            }
            // 1x1 does not require im2col or workspace
            else if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 1 && v == 1))
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1");

                // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
                GemmDescriptor gemm_desc =
                    CreateGemmStridedBatchedDescriptorConv1x1BwdWeight(dyDesc, xDesc, dwDesc);

                miopenStatus_t gemm_status = miopenStatusNotInitialized;

                if(!IsDisabled(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING{}))
                {
                    // rocBLAS need a warm-up call for accurate timing
                    CallGemmStridedBatchedSequential(handle,
                                                     gemm_desc,
                                                     dy,
                                                     0,
                                                     x,
                                                     0,
                                                     tmp_dw.get(),
                                                     0,
                                                     nullptr,
                                                     false,
                                                     GemmBackend_t::miopengemm);

                    // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
                    gemm_status = CallGemmStridedBatchedSequential(handle,
                                                                   gemm_desc,
                                                                   dy,
                                                                   0,
                                                                   x,
                                                                   0,
                                                                   tmp_dw.get(),
                                                                   0,
                                                                   nullptr,
                                                                   true,
                                                                   GemmBackend_t::miopengemm);
                }
                else
                {
                    // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
                    gemm_status = CallGemmStridedBatchedSequential(handle,
                                                                   gemm_desc,
                                                                   dy,
                                                                   0,
                                                                   x,
                                                                   0,
                                                                   tmp_dw.get(),
                                                                   0,
                                                                   nullptr,
                                                                   false,
                                                                   GemmBackend_t::miopengemm);
                }

                float time_gemm = handle.GetKernelTime();

                if(gemm_status == miopenStatusSuccess)
                    perf_db.push_back(
                        PerfField{"miopenConvolutionBwdWeightsAlgoGEMM", time_gemm, 0});
            }
        }
#endif

        // direct convolution
        {
            std::tie(std::ignore, std::ignore, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());

            if(!miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT{}))
            {
                mlo_construct_BwdWrW2D construct_params(
                    xDesc, dwDesc, dyDesc, *this, 0); // backward with regards to weights
                construct_params.setDoSearch(exhaustiveSearch);
                construct_params.setStream(&handle);

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
                                         << best
                                         << ", workspce_sz = "
                                         << sol.workspce_sz);
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
                    perf_db.push_back(PerfField{algorithm_name, best, selected.workspce_sz});
                }
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

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_c, wei_h, wei_w;

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    if(mode == miopenConvolution)
    {
        std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());

        switch(algo)
        {
        case miopenConvolutionBwdWeightsAlgoGEMM: {
#if MIOPEN_USE_GEMM
            if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_GEMM{}))
            {
                MIOPEN_THROW("GEMM convolution is disabled");
            }

            // Zeroing out the output buffer
            float zero = 0.0f;
            SetTensor(handle, dwDesc, dw, &zero);

            handle.ResetKernelTime();

            if(wei_h != 1 || wei_w != 1 || pad_h != 0 || pad_w != 0 || u != 1 || v != 1)
            {
                MIOPEN_LOG_FUNCTION("convolution, non 1x1");

                assert(workSpace != nullptr &&
                       workSpaceSize >=
                           BackwardWeightsGetWorkSpaceSizeGEMM(handle, dyDesc, dwDesc));

                // dw = dy * transpose(Im2Col(x))
                GemmDescriptor gemm_desc = CreateGemmDescriptorConvBwdWeight(dyDesc, xDesc, dwDesc);

                float time_0 = 0;
                float t1     = 0;

                for(int i = 0; i < in_n; i++)
                {
                    int out_offset = i * wei_n * out_h * out_w;
                    int in_offset  = i * in_c * in_h * in_w;
                    Im2ColGPU(handle,
                              xDesc.GetElementSize(),
                              x,
                              in_offset,
                              in_c,
                              in_h,
                              in_w,
                              wei_h,
                              wei_w,
                              out_h,
                              out_w,
                              pad_h,
                              pad_w,
                              u,
                              v,
                              dilation_h,
                              dilation_w,
                              workSpace,
                              dyDesc.GetType());

                    if(handle.IsProfilingEnabled())
                        t1 = handle.GetKernelTime();

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
            else if(wei_h == 1 && wei_w == 1 && pad_h == 0 && pad_w == 0 && (u == 1 && v == 1))
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
            else
            {
                MIOPEN_THROW("GEMM WrW convolution cannot be executed due to incorrect params");
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
                    /// We can't use BackwardWeightsGetWorkSpaceSizeDirect() to check if enough
                    /// workspace is provided by the user, because the function returns max of
                    /// all available Solutions, but we do not know how much workspace is
                    /// required for the specific Solution (which is reduced to a vector of
                    /// kernels here) we are going to invoke. So let's check against 0 for now.
                    /// \todo Implement full ws size check. See #1127.
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
        };
    }
    else if(mode == miopenTranspose)
    {
        std::tie(std::ignore, wei_n, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());

        std::string network_config;

        if(wei_h != 1 || wei_w != 1 || v != 1 || u != 1 || pad_h != 0 || pad_w != 0)
        {
            assert(workSpace != nullptr &&
                   workSpaceSize >= BackwardWeightsGetWorkSpaceSizeGEMM(handle, xDesc, dwDesc));
        }

#if MIOPEN_USE_MIOPENGEMM
        CreateGemmGeometryConvBwdWeights(xDesc, dyDesc, dwDesc, false, network_config);
        GemmGeometry gg =
            GetGemmGeometry(handle, "miopenConvolutionBwdWeightsAlgoGEMM", network_config);

        handle.ResetKernelTime();
        float time_0 = 0;
        float t1     = 0;
        for(int i = 0; i < in_n; i++)
        {
            int in_offset = i * in_c * in_h * in_w;
            if(wei_h != 1 || wei_w != 1 || v != 1 || u != 1 || pad_h != 0 || pad_w != 0)
            {
                MIOPEN_LOG_FUNCTION("transpose, non 1x1");

                int out_offset = i * wei_n * out_h * out_w;
                Im2ColGPU(handle,
                          dyDesc.GetElementSize(),
                          dy,
                          out_offset,
                          wei_n,
                          out_h,
                          out_w,
                          wei_h,
                          wei_w,
                          in_h,
                          in_w,
                          pad_h,
                          pad_w,
                          u,
                          v,
                          dilation_h,
                          dilation_w,
                          workSpace,
                          dyDesc.GetType());

                if(handle.IsProfilingEnabled())
                    t1 = handle.GetKernelTime();

                gg.RunGemm(handle, workSpace, x, dw, 0, in_offset, 0);

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
            else if(wei_h == 1 && wei_w == 1 && v == 1 && u == 1 && pad_h == 0 && pad_w == 0)
            {
                MIOPEN_LOG_FUNCTION("transpose, 1x1");

                int out_offset = i * wei_n * out_h * out_w;
                gg.RunGemm(handle, dy, x, dw, out_offset, in_offset, 0);

                if(handle.IsProfilingEnabled())
                {
                    if(i == in_n - 1)
                        handle.AccumKernelTime(time_0);
                    time_0 += handle.GetKernelTime();
                }
            }
        }
#else
        MIOPEN_THROW("GEMM is not supported");
#endif
    }
    else if(mode == miopenGroupConv || mode == miopenDepthwise)
    {
        std::tie(wei_n, wei_c, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());
        if(in_c % group_count != 0 || wei_n % group_count != 0 || group_count > in_c ||
           group_count > wei_n || group_count < 1 ||
           (mode == miopenDepthwise && group_count != in_c))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid group number");
        if(in_c / group_count != wei_c || (mode == miopenDepthwise && wei_c != 1))
            MIOPEN_THROW(miopenStatusBadParm, "Invalid filter channel number");

        switch(algo)
        {
        case miopenConvolutionBwdWeightsAlgoGEMM: {
#if MIOPEN_USE_GEMM
            // Zeroing out the output buffer
            float zero = 0.0f;
            SetTensor(handle, dwDesc, dw, &zero);

            if(wei_h != 1 || wei_w != 1 || v != 1 || u != 1 || pad_h != 0 || pad_w != 0)
            {
                assert(workSpace != nullptr &&
                       workSpaceSize >= (group_count * BackwardWeightsGetWorkSpaceSizeGEMM(
                                                           handle, dyDesc, dwDesc)));
            }

            GemmDescriptor gemm_desc =
                CreateGemmDescriptorGroupConvBwdWeight(dyDesc, xDesc, dwDesc, group_count);

            handle.ResetKernelTime();
            float time_0 = 0;
            float t1     = 0;
            for(int i = 0; i < in_n; i++)
            {
                if(wei_h != 1 || wei_w != 1 || v != 1 || u != 1 || pad_h != 0 || pad_w != 0)
                {
                    size_t in_offset = i * in_c * in_h * in_w;
                    Im2ColGPU(handle,
                              xDesc.GetElementSize(),
                              x,
                              in_offset,
                              in_c,
                              in_h,
                              in_w,
                              wei_h,
                              wei_w,
                              out_h,
                              out_w,
                              pad_h,
                              pad_w,
                              u,
                              v,
                              dilation_h,
                              dilation_w,
                              workSpace,
                              dyDesc.GetType());
                    if(handle.IsProfilingEnabled())
                        t1 += handle.GetKernelTime();
                }

                size_t out_offset = i * wei_n * out_h * out_w;
                if(wei_h != 1 || wei_w != 1 || v != 1 || u != 1 || pad_h != 0 || pad_w != 0)
                {
                    CallGemmStridedBatched(handle,
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

                    // Update times for both the kernels
                    if(handle.IsProfilingEnabled())
                    {
                        if(i == in_n - 1)
                            handle.AccumKernelTime(t1 + time_0);
                        time_0 += handle.GetKernelTime();
                    }
                }
                else
                {
                    size_t in_offset = i * in_c * in_h * in_w;
                    CallGemmStridedBatched(handle,
                                           gemm_desc,
                                           dy,
                                           out_offset,
                                           x,
                                           in_offset,
                                           dw,
                                           0,
                                           nullptr,
                                           false,
                                           GemmBackend_t::miopengemm);

                    if(handle.IsProfilingEnabled())
                    {
                        if(i == in_n - 1)
                            handle.AccumKernelTime(time_0);
                        time_0 += handle.GetKernelTime();
                    }
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

                if(kernels.size() == 1)
                {
                    float padding_val = 0;
                    kernel(dy, x, dw, as_float(padding_val));
                }
                else
                {
                    assert(kernels.size() == 2);
                    // this pointer needed here as a workaround in gcc 5
                    assert(workSpace != nullptr);
                    float padding_val = 0;
                    kernel(dy, x, workSpace, as_float(padding_val));

                    float time0 = handle.GetKernelTime();
                    // second kernel - reduction
                    kernels[1](workSpace, dw);

                    handle.AccumKernelTime(time0);
                }
            });
        } // case
        } // switch
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

    int out_n, out_c, out_h, out_w, stride_n, stride_c, stride_h, stride_w;
    std::tie(out_n, out_c, out_h, out_w)             = tien<4>(dyDesc.GetLengths());
    std::tie(stride_n, stride_c, stride_h, stride_w) = tien<4>(dyDesc.GetStrides());
    std::string program_name = "MIOpenConvBwdBias.cl";
    std::string kernel_name  = "MIOpenConvBwdB";

    std::string params;
    size_t lcl_grp_size0 = 256;
    size_t lcl_grp_size1 = 1;
    size_t local_mem_sz  = 256;

    size_t map_size         = out_w * out_h;
    size_t read_unit        = 4;
    size_t map_size_aligned = (map_size + (read_unit - 1)) / read_unit;
    size_t off_pix          = map_size - (map_size / read_unit) * read_unit;

    params = " -DMLO_CONVBWD_GROUP_SZ0=" + std::to_string(lcl_grp_size0);
    params += " -DMLO_CONVBWD_GROUP_SZ1=" + std::to_string(lcl_grp_size1);
    params += " -DMLO_CONVBWDB_LCL_MEMSZ=" + std::to_string(local_mem_sz);
    params += " -DMLO_CONVBWDB_UNITSIZE=" + std::to_string(read_unit);
    params += " -DMLO_OUT_WIDTH=" + std::to_string(out_w);
    params += " -DMLO_OUT_HEIGHT=" + std::to_string(out_h);
    params += " -DMLO_OUT_BATCH_SZ=" + std::to_string(out_n);
    params += " -DMLO_OUT_CHANNEL_STRIDE=" + std::to_string(stride_c);
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
    const std::vector<size_t> vgd = {lcl_grp_size0, static_cast<size_t>(out_c), size_t{1}};

    handle.AddKernel("miopenConvolutionBwdBias", "", program_name, kernel_name, vld, vgd, params)(
        dy, db);

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, dbDesc, db);
    }
}

} // namespace miopen
