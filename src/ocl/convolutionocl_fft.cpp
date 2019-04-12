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
#include <miopen/convolution.hpp>
#include <miopen/convolution_fft.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/tensor.hpp>
#include <miopen/util.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_FFT)

static void cgemm_grid(size_t* global_work_size,
                       size_t* local_work_size,
                       int cgemm_choice,
                       const int N,
                       const int out_c,
                       const int out_n)
{
    unsigned int threadTile[2];
    unsigned int groupSize[2];

    // grid for cgemm
    if(cgemm_choice == 1)
    {
        threadTile[0] = 4;
        threadTile[1] = 4;

        groupSize[0] = 16;
        groupSize[1] = 16;

        local_work_size[0] = 16;
        local_work_size[1] = 16;
    }
    else if(cgemm_choice == 2)
    {
        threadTile[0] = 4;
        threadTile[1] = 4;

        groupSize[0] = 4;
        groupSize[1] = 4;

        local_work_size[0] = 64;
        local_work_size[1] = 1;
    }
    else
    {
        threadTile[0] = 2;
        threadTile[1] = 2;

        groupSize[0] = 4;
        groupSize[1] = 4;

        local_work_size[0] = 64;
        local_work_size[1] = 1;
    }

    global_work_size[2] = 1;
    global_work_size[2] *= N;

    unsigned int sizeOfC0         = out_c;
    unsigned int sizeOfC1         = out_n;
    auto macroTile0               = static_cast<unsigned int>(groupSize[0] * threadTile[0]);
    auto macroTile1               = static_cast<unsigned int>(groupSize[1] * threadTile[1]);
    unsigned int totalWorkGroups0 = sizeOfC0 / macroTile0;
    unsigned int totalWorkGroups1 = sizeOfC1 / macroTile1;
    // b/c single kernel, add extra work-group here if edge needed
    if(totalWorkGroups0 * macroTile0 < sizeOfC0)
    {
        totalWorkGroups0++;
    }
    if(totalWorkGroups1 * macroTile1 < sizeOfC1)
    {
        totalWorkGroups1++;
    }
    global_work_size[0] = totalWorkGroups0 * local_work_size[0];
    global_work_size[1] = totalWorkGroups1 * local_work_size[1];
}

static std::string make_config_prefix(int in_h, int in_w, int in_n, int in_c, int out_c)
{
    std::string config_prefix = "FFT_x";
    config_prefix += "_in_h_";
    config_prefix += std::to_string(in_h);
    config_prefix += "_in_w_";
    config_prefix += std::to_string(in_w);
    config_prefix += "_in_n_";
    config_prefix += std::to_string(in_n);
    config_prefix += "_in_c_";
    config_prefix += std::to_string(in_c);
    config_prefix += "_out_c_";
    config_prefix += std::to_string(out_c);
    config_prefix += "_kernel_";

    return config_prefix;
}

static int FindFFTKernel(Handle& handle,
                         const TensorDescriptor& xDesc,
                         const TensorDescriptor& wDesc,
                         const TensorDescriptor& yDesc,
                         size_t workSpaceSize,
                         std::vector<KernelInvoke>& kernels,
                         bool fwd,
                         std::string* kcache_key = nullptr)
{

    if(workSpaceSize == 0)
        return -1;

    // disable running any FFT based convolutions by checking this env variable
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_FFT{}))
        return -1;

    if(xDesc.GetType() != miopenFloat || wDesc.GetType() != miopenFloat ||
       yDesc.GetType() != miopenFloat)
        return -1;

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(xDesc.GetLengths());

    (void)wDesc;

    int out_n, out_c;
    std::tie(out_n, out_c, std::ignore, std::ignore) = miopen::tien<4>(yDesc.GetLengths());

    const int N          = FFTConvParams::TileSize(in_h, in_w);
    const int NumKernels = FFTConvParams::NumKernels;

    size_t global_work_size[NumKernels][3];
    size_t local_work_size[NumKernels][3];

    for(int ik = 0; ik < NumKernels; ik++)
    {
        global_work_size[ik][0] = local_work_size[ik][0] = 1;
        global_work_size[ik][1] = local_work_size[ik][1] = 1;
        global_work_size[ik][2] = local_work_size[ik][2] = 1;
    }

    // grid for FFT kernels
    if((in_h == 7) && (in_w == 7))
    {
        local_work_size[0][0]  = 192;
        global_work_size[0][0] = ((in_c * out_n) / 16) * local_work_size[0][0];

        local_work_size[1][0]  = 192;
        global_work_size[1][0] = ((in_c * out_c) / 16) * local_work_size[1][0];

        local_work_size[6][0]  = 192;
        global_work_size[6][0] = ((out_n * out_c) / 16) * local_work_size[6][0];
    }
    else if((in_h == 14) && (in_w == 14))
    {
        local_work_size[0][0]  = 128;
        global_work_size[0][0] = ((in_c * out_n) / 4) * local_work_size[0][0];

        local_work_size[1][0]  = 128;
        global_work_size[1][0] = ((in_c * out_c) / 4) * local_work_size[1][0];

        local_work_size[6][0]  = 128;
        global_work_size[6][0] = ((out_n * out_c) / 4) * local_work_size[6][0];
    }
    else
    {
        local_work_size[0][0]  = 64;
        global_work_size[0][0] = in_c * out_n * local_work_size[0][0];

        local_work_size[1][0]  = 64;
        global_work_size[1][0] = in_c * out_c * local_work_size[1][0];

        local_work_size[6][0]  = 64;
        global_work_size[6][0] = out_n * out_c * local_work_size[6][0];
    }

    // decide tranpose kernel options based on params
    int in_tranpose_choice = 0;
    int wt_tranpose_choice = 0;
    int ot_tranpose_choice = 0;

    // grid for transpose kernels
    if((in_h == 7) && (in_w == 7))
    {
        local_work_size[5][0]  = 256;
        global_work_size[5][0] = (1 + N / 16) * (out_n * out_c / 16) * local_work_size[5][0];
    }
    else if((in_h == 14) && (in_w == 14))
    {
        local_work_size[2][0]  = 256;
        global_work_size[2][0] = (1 + N / 16) * (in_c * out_n / 16) * local_work_size[2][0];

        local_work_size[3][0]  = 256;
        global_work_size[3][0] = (1 + N / 16) * (in_c * out_c / 16) * local_work_size[3][0];

        local_work_size[5][0]  = 256;
        global_work_size[5][0] = (1 + N / 16) * (out_n * out_c / 16) * local_work_size[5][0];
    }
    else
    {
        if((in_n * in_c >= 64) && ((in_n * in_c) % 32 == 0))
            in_tranpose_choice = 1;
        if((out_c * in_c >= 64) && ((out_c * in_c) % 32 == 0))
            wt_tranpose_choice = 1;
        if((out_n * out_c >= 64) && ((out_n * out_c) % 32 == 0))
            ot_tranpose_choice = 1;

        int in_tranpose_bwidth = in_tranpose_choice != 0 ? 32 : 16;
        int wt_tranpose_bwidth = wt_tranpose_choice != 0 ? 32 : 16;
        int ot_tranpose_bwidth = ot_tranpose_choice != 0 ? 32 : 16;

        local_work_size[2][0] = 256;
        global_work_size[2][0] =
            (N / in_tranpose_bwidth) * (in_c * out_n / in_tranpose_bwidth) * local_work_size[2][0];

        local_work_size[3][0] = 256;
        global_work_size[3][0] =
            (N / wt_tranpose_bwidth) * (in_c * out_c / wt_tranpose_bwidth) * local_work_size[3][0];

        local_work_size[5][0] = 256;
        global_work_size[5][0] =
            (N / ot_tranpose_bwidth) * (out_n * out_c / ot_tranpose_bwidth) * local_work_size[5][0];
    }

    // cgemm kernel options
    int cgemm_choice = 0;

    if((in_h == 28) && (in_w == 28))
        cgemm_choice = 2;
    else if((in_h == 27) && (in_w == 27))
        cgemm_choice = 1;
    else if((in_h == 14) && (in_w == 14))
        cgemm_choice = 2;
    else if((in_h == 7) && (in_w == 7))
        cgemm_choice = 2;

    if((in_n < 16) || (in_c < 16) || (out_c < 16))
        cgemm_choice = 0;

    cgemm_grid(global_work_size[4], local_work_size[4], cgemm_choice, N, out_c, out_n);

    std::string parms;

    if(in_tranpose_choice == 0)
        parms += " -DCFF_TRANSP_IN_MOD16=1";
    if(wt_tranpose_choice == 0)
        parms += " -DCFF_TRANSP_WT_MOD16=1";
    if(ot_tranpose_choice == 0)
        parms += " -DCFF_TRANSP_OT_MOD16=1";

    switch(cgemm_choice)
    {
    case 1: parms += " -DCFF_CGEMM_CHOICE_1=1"; break;
    case 2: parms += " -DCFF_CGEMM_CHOICE_2=1"; break;
    default: parms += " -DCFF_CGEMM_CHOICE_0=1"; break;
    }

    if((in_h == 28) && (in_w == 28))
        parms += " -DCFF_IMG_SZ_28_28";
    else if((in_h == 27) && (in_w == 27))
        parms += " -DCFF_IMG_SZ_27_27";
    else if((in_h == 14) && (in_w == 14))
        parms += " -DCFF_IMG_SZ_14_14";
    else if((in_h == 7) && (in_w == 7))
        parms += " -DCFF_IMG_SZ_7_7";

    parms += " -DCFF_IMG_H=";
    parms += std::to_string(in_h);
    parms += " -DCFF_IMG_W=";
    parms += std::to_string(in_w);
    parms += " -DCFF_BATCH=";
    parms += std::to_string(in_n);
    parms += " -DCFF_NFILTER=";
    parms += std::to_string(out_c);
    parms += " -DCFF_CHANNELS=";
    parms += std::to_string(in_c);
    parms += " -DCFF_HALFW=";
    parms += std::to_string(workSpaceSize / (2 * 2 * sizeof(float)));

    if(!fwd)
    {
        parms += " -DCFF_BACKWARD";
    }

    const std::string algorithm    = "miopenConvolutionFwdAlgoFFT";
    const std::string program_name = "MIOpenConvFFT.cl";

    const std::string config_prefix = make_config_prefix(in_h, in_w, in_n, in_c, out_c);

    if(kcache_key != nullptr)
        *kcache_key = config_prefix;

    for(int ik = 0; ik < NumKernels; ik++)
    {
        std::string kernel_name;

        // skip front transposes for 7x7
        if((in_h == 7) && (in_w == 7))
        {
            if((ik == 2) || (ik == 3))
                continue;
        }

        switch(ik)
        {
        case 0: kernel_name += "MIOpenConvFFT_fwd_in"; break;
        case 1: kernel_name += "MIOpenConvFFT_fwd_we"; break;
        case 2: kernel_name += "MIOpenConvFFT_transpose_in"; break;
        case 3: kernel_name += "MIOpenConvFFT_transpose_we"; break;
        case 4: kernel_name += "MIOpenConvFFT_cgemm"; break;
        case 5: kernel_name += "MIOpenConvFFT_transpose_out"; break;
        case 6: kernel_name += "MIOpenConvFFT_inv_out"; break;
        default: assert(false);
        }

        std::string network_config = config_prefix + std::to_string(ik);

        std::vector<size_t> vld(3);
        std::vector<size_t> vgd(3);

        vld[0] = local_work_size[ik][0];
        vld[1] = local_work_size[ik][1];
        vld[2] = local_work_size[ik][2];

        vgd[0] = global_work_size[ik][0];
        vgd[1] = global_work_size[ik][1];
        vgd[2] = global_work_size[ik][2];

        auto k =
            handle.AddKernel(algorithm, network_config, program_name, kernel_name, vld, vgd, parms);

        kernels.push_back(k);
    }

    return 0;
}

int ConvolutionDescriptor::FindFwdFFTKernel(Handle& handle,
                                            const TensorDescriptor& xDesc,
                                            const TensorDescriptor& wDesc,
                                            const TensorDescriptor& yDesc,
                                            size_t workSpaceSize,
                                            std::vector<KernelInvoke>& kernels,
                                            std::string& kcache_key) const
{

    return FindFFTKernel(handle, xDesc, wDesc, yDesc, workSpaceSize, kernels, true, &kcache_key);
}

int ConvolutionDescriptor::FindBwdFFTKernel(Handle& handle,
                                            const TensorDescriptor& dyDesc,
                                            const TensorDescriptor& wDesc,
                                            const TensorDescriptor& dxDesc,
                                            size_t workSpaceSize,
                                            std::vector<KernelInvoke>& kernels) const
{

    return FindFFTKernel(handle, dyDesc, wDesc, dxDesc, workSpaceSize, kernels, false);
}

static float ExecuteFFTKernel(Handle& handle,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& wDesc,
                              ConstData_t w,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              Data_t workSpace,
                              size_t workSpaceSize,
                              bool timed,
                              bool fwd)
{

    (void)wDesc; // suppress warning
    (void)fwd;   // suppress warning

    int halfw = static_cast<int>(workSpaceSize) / (2 * 2 * static_cast<int>(sizeof(float)));
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(xDesc.GetLengths());

    int out_n, out_c;
    std::tie(out_n, out_c, std::ignore, std::ignore) = miopen::tien<4>(yDesc.GetLengths());

    const int N          = FFTConvParams::TileSize(in_h, in_w);
    const int Padding    = FFTConvParams::TransposePadding;
    const int NumKernels = FFTConvParams::NumKernels;

    float time_fft                  = 0;
    const std::string config_prefix = make_config_prefix(in_h, in_w, in_n, in_c, out_c);
    for(int ik = 0; ik < NumKernels; ik++)
    {
        // skip front transposes for 7x7
        if((in_h == 7) && (in_w == 7))
        {
            if((ik == 2) || (ik == 3))
                continue;
        }

        std::string network_config = config_prefix + std::to_string(ik);

        auto k = handle.GetKernel("miopenConvolutionFwdAlgoFFT", network_config);

        switch(ik)
        {
        case 0: k(x, workSpace); break;
        case 1: k(w, workSpace); break;
        case 2: k(workSpace); break;
        case 3: k(workSpace); break;
        case 4:
        {
            k(workSpace,
              0,
              halfw + N * (in_n * in_c + Padding),
              halfw + 0,
              out_c,
              out_n * out_c + Padding,
              in_c,
              in_c * out_c + Padding,
              in_c,
              in_n * in_c + Padding,
              out_c,
              in_n,
              N,
              in_c);
        }
        break;
        case 5: k(workSpace); break;
        case 6: k(workSpace, y); break;
        default: assert(false);
        }

        if(timed)
        {
            time_fft += handle.GetKernelTime();
        }
    }

    return time_fft;
}

float ConvolutionDescriptor::ExecuteFwdFFTKernel(Handle& handle,
                                                 const TensorDescriptor& xDesc,
                                                 ConstData_t x,
                                                 const TensorDescriptor& wDesc,
                                                 ConstData_t w,
                                                 const TensorDescriptor& yDesc,
                                                 Data_t y,
                                                 Data_t workSpace,
                                                 size_t workSpaceSize,
                                                 bool timed) const
{

    return ExecuteFFTKernel(
        handle, xDesc, x, wDesc, w, yDesc, y, workSpace, workSpaceSize, timed, true);
}

float ConvolutionDescriptor::ExecuteBwdFFTKernel(Handle& handle,
                                                 const TensorDescriptor& dyDesc,
                                                 ConstData_t dy,
                                                 const TensorDescriptor& wDesc,
                                                 ConstData_t w,
                                                 const TensorDescriptor& dxDesc,
                                                 Data_t dx,
                                                 Data_t workSpace,
                                                 size_t workSpaceSize,
                                                 bool timed) const
{

    return ExecuteFFTKernel(
        handle, dyDesc, dy, wDesc, w, dxDesc, dx, workSpace, workSpaceSize, timed, false);
}

} // namespace miopen
