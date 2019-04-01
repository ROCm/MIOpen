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
#ifndef GUARD_MIOPEN_CONV_DRIVER_HPP
#define GUARD_MIOPEN_CONV_DRIVER_HPP

/*#ifdef MIOPEN_NEURON_SOFTRELU /// \todo This needs to be explained or rewritten in clear manner.
#undef MIOPEN_NEURON_SOFTRELU
#endif

#ifdef MIOPEN_NEURON_POWER /// \todo This needs to be explained or rewritten in clear manner.
#undef MIOPEN_NEURON_POWER
#endif
*/
#include "InputFlags.hpp"
#include "conv_verify.hpp"
#include "driver.hpp"
#include "mloConvHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"
#include <miopen/convolution.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <fstream>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/env.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/logger.hpp>
#include "random.hpp"
#include <numeric>
#include <sstream>
#include <vector>
#include <type_traits>
#include <boost/range/adaptors.hpp>
#include <../test/verify.hpp>
#include <../test/serialize.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/cpu_conv.hpp>
#include <../test/cpu_bias.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DRIVER_PAD_BUFFERS_2M)

template <typename T>
void dumpBufferToFile(const char* fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile)
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
        printf("Wrote output to file %s\n", fileName);
    }
    else
    {
        printf("Could not open file %s for writing\n", fileName);
    }
}

template <typename T>
bool readBufferFromFile(T* data, size_t dataNumItems, const char* fileName)
{
    std::ifstream infile(fileName, std::ios::binary);
    if(infile)
    {
        infile.read(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        infile.close();
        printf("Read data from input file %s\n", fileName);
        return true;
    }
    else
    {
        printf("Could not open file %s for reading\n", fileName);
        return false;
    }
}

// Tgpu and Tref are the data-type in GPU memory and CPU memory respectively.
// They are not necessarily the same as the computation type on GPU or CPU
template <typename Tgpu, typename Tref>
class ConvDriver : public Driver
{
    public:
    ConvDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&biasTensor);
        miopenCreateTensorDescriptor(&inputTensor_vect4);
        miopenCreateTensorDescriptor(&weightTensor_vect4);

        miopenCreateConvolutionDescriptor(&convDesc);

        workspace_bwd_data_dev    = nullptr;
        workspace_bwd_weights_dev = nullptr;
        workspace_fwd_dev         = nullptr;
        // the variable name is implementation dependent, checking size instead
        data_type = std::is_same<Tgpu, int8_t>::value
                        ? miopenInt8
                        : std::is_same<Tgpu, float16>::value ? miopenHalf : miopenFloat;
    }

    int AddCmdLineArgs();
    int ParseCmdLineArgs(int argc, char* argv[]);
    InputFlags& GetInputFlags() { return inflags; }

    int GetandSetData();
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetWeightTensorLengthsFromCmdLine();
    std::vector<int> GetBiasTensorLengthsFromCmdLine();

    int SetConvDescriptorFromCmdLineArgs();

    std::vector<int> GetOutputTensorLengths();

    int AllocateBuffersAndCopy();

    int FindForward(int& ret_algo_count,
                    int request_algo_count,
                    std::vector<miopenConvAlgoPerf_t>& perf_results);
    int RunForwardGPU();
    int RunForwardCPU();

    int FindBackwardData(int& ret_algo_count,
                         int request_algo_count,
                         std::vector<miopenConvAlgoPerf_t>& perf_results);
    int FindBackwardWeights(int& ret_algo_count,
                            int request_algo_count,
                            std::vector<miopenConvAlgoPerf_t>& perf_results);
    int RunBackwardGPU();
    int RunBackwardDataCPU();
    int RunBackwardWeightsCPU();
    int RunBackwardBiasCPU();

    int VerifyBackward();
    int VerifyForward();
    ~ConvDriver()
    {

        miopenDestroyTensorDescriptor(biasTensor);
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(weightTensor);
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(inputTensor_vect4);
        miopenDestroyTensorDescriptor(weightTensor_vect4);

        miopenDestroyConvolutionDescriptor(convDesc);
    }

    private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t outputTensor;
    miopenTensorDescriptor_t biasTensor;
    miopenTensorDescriptor_t inputTensor_vect4;
    miopenTensorDescriptor_t weightTensor_vect4;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> in_vect4_dev;
    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> wei_dev;
    std::unique_ptr<GPUMem> wei_vect4_dev;
    std::unique_ptr<GPUMem> dwei_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> dout_dev;
    std::unique_ptr<GPUMem> workspace_bwd_data_dev;
    std::unique_ptr<GPUMem> workspace_bwd_weights_dev;
    std::unique_ptr<GPUMem> workspace_fwd_dev;
    std::unique_ptr<GPUMem> b_dev;
    std::unique_ptr<GPUMem> db_dev;

    tensor<Tgpu> in;
    tensor<Tgpu> wei;
    tensor<Tgpu> out;
    tensor<Tgpu> dout;
    tensor<Tgpu> b;
    tensor<Tref> outhost;
    tensor<Tref> dwei_host;
    tensor<Tref> din_host;
    tensor<Tref> db_host;

    std::vector<Tgpu> din;
    std::vector<Tgpu> dwei;
    std::vector<float> out_int8;
    std::vector<Tgpu> workspace_bwd_data;
    std::vector<Tgpu> workspace_bwd_weights;
    std::vector<Tgpu> workspace_fwd;
    std::vector<Tref> workspace_bwd_data_host;
    std::vector<Tref> workspace_bwd_weights_host;
    std::vector<Tref> workspace_fwd_host;
    std::vector<Tgpu> db;
    std::vector<float> b_int8;

    miopenConvolutionDescriptor_t convDesc;

    bool wrw_allowed = 1, bwd_allowed = 1, forward_allowed = 1;
    bool is_wrw_winograd = false;

    std::string GetVerificationCacheFileName() const;
    std::string GetVCacheFwdOutBasename() const;
    std::string GetVCacheBwdDataBasename() const;
    std::string GetVCacheBwdWeightBasename() const;
    std::string GetVCacheBiasBwdDataBasename() const;
    bool IsInputTensorTransform() const;

    bool TryReadVerificationCache(const std::string& file_name,
                                  miopenTensorDescriptor_t& tensorDesc,
                                  Tref* data) const;
    void TrySaveVerificationCache(const std::string& file_name, std::vector<Tref>& data) const;
};

// Check if int8 type tensor x and w need to be transformed to a pack of 4 elements along channel
// (NCHW_VECT_C format)
template <typename Tgpu, typename Tref>
bool ConvDriver<Tgpu, Tref>::IsInputTensorTransform() const
{
    return (data_type == miopenInt8 && inflags.GetValueInt("in_channels") % 4 != 0) ||
           data_type == miopenInt8x4;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forward_allowed = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 1);
    bwd_allowed     = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 2);
    wrw_allowed     = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 4);

    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> wei_len = GetWeightTensorLengthsFromCmdLine();

    SetTensorNd(inputTensor, in_len, data_type);
    SetTensorNd(weightTensor, wei_len, data_type);

    if(inflags.GetValueInt("tensor_vect") == 1 && data_type == miopenInt8)
    {
        data_type = miopenInt8x4;
    }

    if(IsInputTensorTransform())
    {
        std::vector<int> in_len_vect4(in_len.begin(), in_len.end()),
            wei_len_vect4(wei_len.begin(), wei_len.end());
        in_len_vect4[1] = ((in_len[1] + 3) / 4) * 4;
        SetTensorNd(inputTensor_vect4, in_len_vect4, data_type);
        wei_len_vect4[1] = ((wei_len[1] + 3) / 4) * 4;
        SetTensorNd(weightTensor_vect4, wei_len_vect4, data_type);
    }
    SetConvDescriptorFromCmdLineArgs();

    std::vector<int> out_len = GetOutputTensorLengths();

    miopenDataType_t y_type =
        (data_type == miopenInt8 || data_type == miopenInt8x4) ? miopenFloat : data_type;
    SetTensorNd(outputTensor, out_len, y_type);

    if(inflags.GetValueInt("bias") != 0)
    {
        std::vector<int> bias_len = GetBiasTensorLengthsFromCmdLine();
        SetTensorNd(biasTensor, bias_len, data_type);
    }
    return (0);
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "spatial_dim", '_', "2", "convolution spatial dimension (Default-2)", "int");
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Flag enables fwd, bwd, wrw convolutions"
                         "\n0 fwd+bwd+wrw (default)"
                         "\n1 fwd only"
                         "\n2 bwd only"
                         "\n4 wrw only"
                         "\n3 fwd+bwd"
                         "\n5 fwd+wrw"
                         "\n6 bwd+wrw",
                         "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", '!', "32", "Input Depth (Default=32)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag(
        "out_channels", 'k', "32", "Number of Output Channels (Default=32)", "int");
    inflags.AddInputFlag("fil_d", '@', "3", "Filter Depth (Default=3)", "int");
    inflags.AddInputFlag("fil_h", 'y', "3", "Filter Height (Default=3)", "int");
    inflags.AddInputFlag("fil_w", 'x', "3", "Filter Width (Default=3)", "int");
    inflags.AddInputFlag(
        "conv_stride_d", '#', "1", "Convolution Stride for Depth (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_h", 'u', "1", "Convolution Stride for Height (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_w", 'v', "1", "Convolution Stride for Width (Default=1)", "int");
    inflags.AddInputFlag("pad_d", '$', "0", "Zero Padding for Depth (Default=0)", "int");
    inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding for Height (Default=0)", "int");
    inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding for Width (Default=0)", "int");
    inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_d", '%', "0", "Zero Padding Output for Depth (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_h", 'Y', "0", "Zero Padding Output for Height (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_w", 'X', "0", "Zero Padding Output for Width (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("verification_cache",
                         'C',
                         "",
                         "Use specified directory to cache verification data. Off by default.",
                         "string");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    inflags.AddInputFlag("search", 's', "0", "Search Kernel Config (Default=0)", "int");
    inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
    inflags.AddInputFlag("in_data", 'd', "", "Input data filename (Default=)", "string");
    inflags.AddInputFlag("weights", 'e', "", "Input weights filename (Default=)", "string");
    inflags.AddInputFlag("bias", 'b', "", "Use Bias (Default=0)", "int");
    inflags.AddInputFlag(
        "mode", 'm', "conv", "Convolution Mode (conv, trans) (Default=conv)", "str");

    inflags.AddInputFlag(
        "pad_mode", 'z', "default", "Padding Mode (same, valid, default) (Default=default)", "str");
    inflags.AddInputFlag("tensor_vect",
                         'Z',
                         "0",
                         "tensor vectorization type (none, vect_c, vect_n) (Default=0)",
                         "int");
    inflags.AddInputFlag("dilation_d", '^', "1", "Dilation of Filter Depth (Default=1)", "int");
    inflags.AddInputFlag("dilation_h", 'l', "1", "Dilation of Filter Height (Default=1)", "int");
    inflags.AddInputFlag("dilation_w", 'j', "1", "Dilation of Filter Width (Default=1)", "int");
    inflags.AddInputFlag("in_bias", 'a', "", "Input bias filename (Default=)", "string");
    inflags.AddInputFlag("group_count", 'g', "1", "Number of Groups (Default=1)", "int");
    inflags.AddInputFlag("dout_data",
                         'D',
                         "",
                         "dy data filename for backward weight computation (Default=)",
                         "string");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    std::vector<int> in_lens;

    int spatial_dim = inflags.GetValueInt("spatial_dim");
    in_lens.resize(2 + spatial_dim);

    in_lens[0] = inflags.GetValueInt("batchsize");
    in_lens[1] = inflags.GetValueInt("in_channels");

    auto in_spatial_lens = boost::adaptors::slice(in_lens, 2, 2 + spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0] = inflags.GetValueInt("in_h");
        in_spatial_lens[1] = inflags.GetValueInt("in_w");
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0] = inflags.GetValueInt("in_d");
        in_spatial_lens[1] = inflags.GetValueInt("in_h");
        in_spatial_lens[2] = inflags.GetValueInt("in_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    return in_lens;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvDriver<Tgpu, Tref>::GetWeightTensorLengthsFromCmdLine()
{
    std::vector<int> wei_lens;

    int spatial_dim = inflags.GetValueInt("spatial_dim");
    wei_lens.resize(2 + spatial_dim);

    auto wei_spatial_lens = boost::adaptors::slice(wei_lens, 2, 2 + spatial_dim);

    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    int wei_k_len = inflags.GetValueInt("out_channels");
    int wei_c_len = inflags.GetValueInt("in_channels");

    if(spatial_dim == 2)
    {
        wei_spatial_lens[0] = inflags.GetValueInt("fil_h");
        wei_spatial_lens[1] = inflags.GetValueInt("fil_w");
    }
    else if(spatial_dim == 3)
    {
        wei_spatial_lens[0] = inflags.GetValueInt("fil_d");
        wei_spatial_lens[1] = inflags.GetValueInt("fil_h");
        wei_spatial_lens[2] = inflags.GetValueInt("fil_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    if(group_count > 1)
    {
        if(wei_c_len % group_count != 0 || wei_k_len % group_count != 0 ||
           group_count > wei_c_len || group_count > wei_k_len)
        {
            MIOPEN_THROW("Invalid group number\n");
        }
    }

    miopenConvolutionMode_t mode;
    if((inflags.GetValueStr("mode")) == "conv")
    {
        mode = miopenConvolution;
    }
    else if((inflags.GetValueStr("mode")) == "trans")
    {
        mode = miopenTranspose;
    }
    else
    {
        MIOPEN_THROW("Incorrect Convolution Mode\n");
    }

    if(mode == miopenTranspose)
    {
        wei_lens[0] = wei_c_len;
        wei_lens[1] = wei_k_len / group_count;
    }
    else
    {
        wei_lens[0] = wei_k_len;
        wei_lens[1] = wei_c_len / group_count;
    }

    return wei_lens;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvDriver<Tgpu, Tref>::GetBiasTensorLengthsFromCmdLine()
{
    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = inflags.GetValueInt("out_channels");

    return bias_lens;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::SetConvDescriptorFromCmdLineArgs()
{
    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> in_spatial_lens(spatial_dim);
    std::vector<int> wei_spatial_lens(spatial_dim);
    std::vector<int> pads(spatial_dim);
    std::vector<int> conv_strides(spatial_dim);
    std::vector<int> conv_dilations(spatial_dim);
    std::vector<int> trans_output_pads(spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0]   = inflags.GetValueInt("in_h");
        in_spatial_lens[1]   = inflags.GetValueInt("in_w");
        wei_spatial_lens[0]  = inflags.GetValueInt("fil_h");
        wei_spatial_lens[1]  = inflags.GetValueInt("fil_w");
        pads[0]              = inflags.GetValueInt("pad_h");
        pads[1]              = inflags.GetValueInt("pad_w");
        conv_strides[0]      = inflags.GetValueInt("conv_stride_h");
        conv_strides[1]      = inflags.GetValueInt("conv_stride_w");
        conv_dilations[0]    = inflags.GetValueInt("dilation_h");
        conv_dilations[1]    = inflags.GetValueInt("dilation_w");
        trans_output_pads[0] = inflags.GetValueInt("trans_output_pad_h");
        trans_output_pads[1] = inflags.GetValueInt("trans_output_pad_w");
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0]   = inflags.GetValueInt("in_d");
        in_spatial_lens[1]   = inflags.GetValueInt("in_h");
        in_spatial_lens[2]   = inflags.GetValueInt("in_w");
        wei_spatial_lens[0]  = inflags.GetValueInt("fil_d");
        wei_spatial_lens[1]  = inflags.GetValueInt("fil_h");
        wei_spatial_lens[2]  = inflags.GetValueInt("fil_w");
        pads[0]              = inflags.GetValueInt("pad_d");
        pads[1]              = inflags.GetValueInt("pad_h");
        pads[2]              = inflags.GetValueInt("pad_w");
        conv_strides[0]      = inflags.GetValueInt("conv_stride_d");
        conv_strides[1]      = inflags.GetValueInt("conv_stride_h");
        conv_strides[2]      = inflags.GetValueInt("conv_stride_w");
        conv_dilations[0]    = inflags.GetValueInt("dilation_d");
        conv_dilations[1]    = inflags.GetValueInt("dilation_h");
        conv_dilations[2]    = inflags.GetValueInt("dilation_w");
        trans_output_pads[0] = inflags.GetValueInt("trans_output_pad_d");
        trans_output_pads[1] = inflags.GetValueInt("trans_output_pad_h");
        trans_output_pads[2] = inflags.GetValueInt("trans_output_pad_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    int out_c       = inflags.GetValueInt("out_channels");
    int in_c        = inflags.GetValueInt("in_channels");
    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    if(group_count > 1)
    {
        if(in_c % group_count != 0 || out_c % group_count != 0 || group_count > in_c ||
           group_count > out_c)
        {
            printf("Invalid group number\n");
            exit(0);
        }
    }

    miopenConvolutionMode_t mode;
    if((inflags.GetValueStr("mode")) == "conv")
    {
        mode = miopenConvolution;
    }
    else if((inflags.GetValueStr("mode")) == "trans")
    {
        mode = miopenTranspose;
    }
    else
    {
        printf("Incorrect Convolution Mode\n");
        exit(0);
    }

    // adjust padding based on user-defined padding mode
    if(mode == miopenConvolution &&
       (miopen::all_of(conv_dilations, [](auto v) { return v == 1; }) ||
        miopen::all_of(wei_spatial_lens, [](auto v) { return v == 1; })))
    {
        if((inflags.GetValueStr("pad_mode")) == "same")
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] =
                    (in_spatial_lens[i] % conv_strides[i] == 0)
                        ? (std::max((wei_spatial_lens[i] - conv_strides[i]), 0))
                        : (std::max((wei_spatial_lens[i] - (in_spatial_lens[i] % conv_strides[i])),
                                    0));
                pads[i] /= 2;
            }
        }
        else if((inflags.GetValueStr("pad_mode")) == "valid")
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] = 0;
            }
        }
    }

    miopenInitConvolutionNdDescriptor(
        convDesc, spatial_dim, pads.data(), conv_strides.data(), conv_dilations.data(), mode);

    miopenSetConvolutionGroupCount(convDesc, group_count);

    if(mode == miopenTranspose)
    {
        miopenSetTransposeConvNdOutputPadding(convDesc, spatial_dim, trans_output_pads.data());
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvDriver<Tgpu, Tref>::GetOutputTensorLengths()
{
    int ndim = miopen::deref(inputTensor).GetSize();

    std::vector<int> out_lens(ndim);

    miopenGetConvolutionNdForwardOutputDim(
        convDesc, inputTensor, weightTensor, &ndim, out_lens.data());

    return out_lens;
}

namespace detail {

template <typename T>
T RanGenWeights()
{
    return RAN_GEN<T>(static_cast<T>(-0.5), static_cast<T>(0.5));
}

// Shift FP16 distribution towards positive numbers,
// otherwise Winograd FP16 validation fails.
template <>
float16 RanGenWeights()
{
    return RAN_GEN<float16>(static_cast<float16>(-1.0 / 3.0), static_cast<float16>(0.5));
}

} // namespace detail

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    bool is_transform           = IsInputTensorTransform();
    bool is_int8                = data_type == miopenInt8 || data_type == miopenInt8x4;
    size_t in_sz                = GetTensorSize(inputTensor);
    size_t wei_sz               = GetTensorSize(weightTensor);
    size_t out_sz               = GetTensorSize(outputTensor);
    size_t workSpaceSize_fwd    = 0;
    size_t workSpaceSize_bwd_wt = 0;
    size_t workSpaceSize_bwd_dt = 0;

    if(wrw_allowed)
        miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            GetHandle(), outputTensor, inputTensor, convDesc, weightTensor, &workSpaceSize_bwd_wt);
    if(bwd_allowed)
        miopenConvolutionBackwardDataGetWorkSpaceSize(
            GetHandle(), outputTensor, weightTensor, convDesc, inputTensor, &workSpaceSize_bwd_dt);
    if(forward_allowed)
        miopenConvolutionForwardGetWorkSpaceSize(GetHandle(),
                                                 (is_transform ? weightTensor_vect4 : weightTensor),
                                                 (is_transform ? inputTensor_vect4 : inputTensor),
                                                 convDesc,
                                                 outputTensor,
                                                 &workSpaceSize_fwd);

    // Workaround: Pad buffers allocations to be a multiple of 2M
    if(miopen::IsEnabled(MIOPEN_DRIVER_PAD_BUFFERS_2M{}))
    {
        // PadBufferSize(in_sz, sizeof(Tgpu));
        PadBufferSize(wei_sz, sizeof(Tgpu));
        PadBufferSize(out_sz, sizeof(Tgpu));
    }

    size_t workSpaceNbVal_bwd_dt = workSpaceSize_bwd_dt / sizeof(Tgpu);
    size_t workSpaceNbVal_bwd_wt = workSpaceSize_bwd_wt / sizeof(Tgpu);
    size_t workSpaceNbVal_fwd    = workSpaceSize_fwd / sizeof(Tgpu);

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    in_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    wei_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
    dwei_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
    dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    out_dev =
        std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, is_int8 ? sizeof(float) : sizeof(Tgpu)));
    if(workSpaceSize_bwd_dt != 0)
    {
        workspace_bwd_data_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceNbVal_bwd_dt, sizeof(Tgpu)));
        workspace_bwd_data      = std::vector<Tgpu>(workSpaceNbVal_bwd_dt, static_cast<Tgpu>(0));
        workspace_bwd_data_host = std::vector<Tref>(workSpaceNbVal_bwd_dt, static_cast<Tref>(0));
    }
    if(workSpaceSize_bwd_wt != 0)
    {
        workspace_bwd_weights_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceNbVal_bwd_wt, sizeof(Tgpu)));
        workspace_bwd_weights      = std::vector<Tgpu>(workSpaceNbVal_bwd_wt, static_cast<Tgpu>(0));
        workspace_bwd_weights_host = std::vector<Tref>(workSpaceNbVal_bwd_wt, static_cast<Tref>(0));
    }
    if(workSpaceSize_fwd != 0)
    {
        workspace_fwd_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceNbVal_fwd, sizeof(Tgpu)));
        workspace_fwd      = std::vector<Tgpu>(workSpaceNbVal_fwd, static_cast<Tgpu>(0));
        workspace_fwd_host = std::vector<Tref>(workSpaceNbVal_fwd, static_cast<Tref>(0));
    }

    in   = tensor<Tgpu>(miopen::deref(inputTensor).GetLengths());
    wei  = tensor<Tgpu>(miopen::deref(weightTensor).GetLengths());
    out  = tensor<Tgpu>(miopen::deref(outputTensor).GetLengths());
    dout = tensor<Tgpu>(miopen::deref(outputTensor).GetLengths());

    din  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    dwei = std::vector<Tgpu>(wei_sz, static_cast<Tgpu>(0));
    if(is_int8)
        out_int8 = std::vector<float>(out_sz, static_cast<float>(0));
    if(is_transform)
    {
        in_vect4_dev = std::unique_ptr<GPUMem>(
            new GPUMem(ctx, GetTensorSize(inputTensor_vect4), sizeof(Tgpu)));
        wei_vect4_dev = std::unique_ptr<GPUMem>(
            new GPUMem(ctx, GetTensorSize(weightTensor_vect4), sizeof(Tgpu)));
    }

    outhost   = tensor<Tref>(miopen::deref(outputTensor).GetLengths());
    din_host  = tensor<Tref>(miopen::deref(inputTensor).GetLengths());
    dwei_host = tensor<Tref>(miopen::deref(weightTensor).GetLengths());

    std::string inFileName   = inflags.GetValueStr("in_data");
    std::string weiFileName  = inflags.GetValueStr("weights");
    std::string biasFileName = inflags.GetValueStr("in_bias");
    std::string doutFileName = inflags.GetValueStr("dout_data");

    /* Unless seed is persistent between runs validation using cache stored in file is impossible.
     */
    srand(0);

    bool dataRead = false;
    if(!inFileName.empty())
    {
        dataRead = readBufferFromFile<Tgpu>(in.data.data(), in_sz, inFileName.c_str());
    }

    bool weiRead = false;
    if(!weiFileName.empty())
    {
        weiRead = readBufferFromFile<Tgpu>(wei.data.data(), wei_sz, weiFileName.c_str());
    }

    if(is_int8)
    {
        float Data_scale = 127.0;

        if(!dataRead)
        {
            for(int i = 0; i < in_sz; i++)
            {
                in.data[i] = static_cast<Tgpu>(
                    Data_scale * RAN_GEN<float>(static_cast<float>(0.0), static_cast<float>(1.0)));
                // printf("in  %d  %d \n",i,in.data[i]);
            }
        }

        if(inflags.GetValueInt("bias") != 0)
        {
            size_t b_sz = GetTensorSize(biasTensor);
            b_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(float)));
            b_int8      = std::vector<float>(b_sz, static_cast<float>(0));
            for(int i = 0; i < b_sz; i++)
            {
                b_int8[i] = static_cast<float>(i % 8) +
                            RAN_GEN<float>(static_cast<float>(0.0), static_cast<float>(1.0));
            }

            if(!biasFileName.empty())
            {
                readBufferFromFile<float>(b_int8.data(), b_sz, biasFileName.c_str());
            }

            b_dev->ToGPU(q, b_int8.data());
        }

        if(!weiRead)
        {
            for(int i = 0; i < wei_sz; i++)
            {
                wei.data[i] = static_cast<Tgpu>(Data_scale * 2 * detail::RanGenWeights<float>());
                // printf("wei  %d  %d \n",i,wei.data[i]);
            }
        }
    }
    else
    {
        Tgpu Data_scale = static_cast<Tgpu>(0.01);

        bool doutRead = false;
        if(!doutFileName.empty())
        {
            doutRead = readBufferFromFile<Tgpu>(dout.data.data(), out_sz, doutFileName.c_str());
        }

        if(!dataRead)
        {
            for(int i = 0; i < in_sz; i++)
            {
                in.data[i] =
                    Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            }
        }

        if(!doutRead)
        {
            for(int i = 0; i < out_sz; i++)
            {
                dout.data[i] =
                    Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            }
        }

        if(inflags.GetValueInt("bias") != 0)
        {
            size_t b_sz = GetTensorSize(biasTensor);
            b_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(Tgpu)));
            db_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(Tgpu)));
            b           = tensor<Tgpu>(miopen::deref(biasTensor).GetLengths());
            db          = std::vector<Tgpu>(b_sz, static_cast<Tgpu>(0));
            db_host     = tensor<Tref>(miopen::deref(biasTensor).GetLengths());
            for(int i = 0; i < b_sz; i++)
            {
                b.data[i] = static_cast<Tgpu>(i % 8) +
                            RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
                db[i] = static_cast<Tgpu>(i % 8) +
                        RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            }

            if(!biasFileName.empty())
            {
                readBufferFromFile<Tgpu>(b.data.data(), b_sz, biasFileName.c_str());
            }

            b_dev->ToGPU(q, b.data.data());
            db_dev->ToGPU(q, db.data());
        }

        if(!weiRead)
        {
            for(int i = 0; i < wei_sz; i++)
            {
                wei.data[i] = Data_scale * detail::RanGenWeights<Tgpu>();
            }
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tgpu>("dump_in.bin", in.data.data(), in_sz);
        dumpBufferToFile<Tgpu>("dump_wei.bin", wei.data.data(), wei_sz);
        if(inflags.GetValueInt("bias") != 0)
            dumpBufferToFile<Tgpu>("dump_bias.bin", b.data.data(), b.data.size());

        dumpBufferToFile<Tgpu>("dump_dout.bin", dout.data.data(), out_sz);
    }

#if MIOPEN_BACKEND_OPENCL
    cl_int status;
#elif MIOPEN_BACKEND_HIP
#define CL_SUCCESS 0
    int status;
#endif
    status = in_dev->ToGPU(q, in.data.data());
    status |= din_dev->ToGPU(q, din.data());
    status |= wei_dev->ToGPU(q, wei.data.data());
    status |= dwei_dev->ToGPU(q, dwei.data());
    status |= dout_dev->ToGPU(q, dout.data.data());
    status |= (is_int8 ? out_dev->ToGPU(q, out_int8.data()) : out_dev->ToGPU(q, out.data.data()));
    if(workSpaceSize_bwd_dt != 0)
        status |= workspace_bwd_data_dev->ToGPU(q, workspace_bwd_data.data());
    if(workSpaceSize_bwd_wt != 0)
        status |= workspace_bwd_weights_dev->ToGPU(q, workspace_bwd_weights.data());
    if(workSpaceSize_fwd != 0)
        status |= workspace_fwd_dev->ToGPU(q, workspace_fwd.data());

    if(status != CL_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::FindForward(int& ret_algo_count,
                                        int request_algo_count,
                                        std::vector<miopenConvAlgoPerf_t>& perf_results)
{
    bool is_transform = IsInputTensorTransform();

    return miopenFindConvolutionForwardAlgorithm(
        GetHandle(),
        (is_transform ? inputTensor_vect4 : inputTensor),
        (is_transform ? in_vect4_dev->GetMem() : in_dev->GetMem()),
        (is_transform ? weightTensor_vect4 : weightTensor),
        (is_transform ? wei_vect4_dev->GetMem() : wei_dev->GetMem()),
        convDesc,
        outputTensor,
        out_dev->GetMem(),
        request_algo_count,
        &ret_algo_count,
        perf_results.data(),
        (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetMem() : nullptr,
        (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetSize() : 0,
        (inflags.GetValueInt("search") == 1) ? true : false);
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunForwardGPU()
{
    if(!forward_allowed)
        return 0;

    int ret_algo_count;
    int request_algo_count = 2;
    std::vector<miopenConvAlgoPerf_t> perf_results(request_algo_count);

    bool is_transform = IsInputTensorTransform();
    if(is_transform)
    {
        float aph = 1.0;
        float bta = 0.0;
        miopenTransformTensor(GetHandle(),
                              &aph,
                              inputTensor,
                              in_dev->GetMem(),
                              &bta,
                              inputTensor_vect4,
                              in_vect4_dev->GetMem());

        miopenTransformTensor(GetHandle(),
                              &aph,
                              weightTensor,
                              wei_dev->GetMem(),
                              &bta,
                              weightTensor_vect4,
                              wei_vect4_dev->GetMem());
    }

    FindForward(ret_algo_count, request_algo_count, perf_results);

    if(ret_algo_count == 0)
        throw std::runtime_error("Find Forward Conv. ret_algo_count == 0");

    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenConvolutionForward(GetHandle(),
                                 &alpha,
                                 (is_transform ? inputTensor_vect4 : inputTensor),
                                 (is_transform ? in_vect4_dev->GetMem() : in_dev->GetMem()),
                                 (is_transform ? weightTensor_vect4 : weightTensor),
                                 (is_transform ? wei_vect4_dev->GetMem() : wei_dev->GetMem()),
                                 convDesc,
                                 perf_results[0].fwd_algo, // use the fastest algo
                                 &beta,
                                 outputTensor,
                                 out_dev->GetMem(),
                                 (workspace_fwd_dev != nullptr) ? workspace_fwd_dev->GetMem()
                                                                : nullptr,
                                 perf_results[0].memory);

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(miopen::deref(inputTensor).GetLengths());
        int wei_c, wei_n, wei_h, wei_w;
        std::tie(wei_c, wei_n, wei_h, wei_w) =
            miopen::tien<4>(miopen::deref(weightTensor).GetLengths());
        int out_n, out_c, out_h, out_w;
        std::tie(out_n, out_c, out_h, out_w) =
            miopen::tien<4>(miopen::deref(outputTensor).GetLengths());
        size_t flopCnt = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w;
        size_t inputBytes =
            in_n * in_c * in_h * in_w * miopen::GetTypeSize(miopen::deref(inputTensor).GetType());
        size_t weightBytes = wei_n * wei_c * wei_h * wei_w *
                             miopen::GetTypeSize(miopen::deref(weightTensor).GetType());
        size_t readBytes = inputBytes + weightBytes;

        size_t outputBytes = 1.0 * out_n * out_c * out_h * out_w *
                             miopen::GetTypeSize(miopen::deref(outputTensor).GetType());

        STOP_TIME
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward Conv. Elapsed: %f ms\n",
                   t.gettime_ms() / inflags.GetValueInt("iter"));

        int iter = inflags.GetValueInt("iter");
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;

        printf("MIOpen Forward Conv. Algorithm: %d\n", perf_results[0].fwd_algo);
        printf("GPU Kernel Time Forward Conv. Elapsed: %f ms (average)\n", kernel_average_time);
        printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
               "GB/s, timeMs\n");
        printf("stats: %s%dx%du%d, %u, %u, %u, %u, %u, %u, %u,  %zu, %zu, %zu, %.0f, %.0f, %f\n",
               "fwd-conv",
               wei_h,
               wei_w,
               miopen::deref(convDesc).GetConvStrides()[0],
               in_n,
               in_c,
               out_h,
               out_w,
               wei_h,
               wei_w,
               out_c,
               flopCnt,
               readBytes,
               outputBytes,
               flopCnt / kernel_average_time / 1e6,
               (readBytes + outputBytes) / kernel_average_time / 1e6,
               kernel_average_time);
    }

    if(inflags.GetValueInt("bias") != 0)
    {
        miopenConvolutionForwardBias(GetHandle(),
                                     &alpha,
                                     biasTensor,
                                     b_dev->GetMem(),
                                     &beta,
                                     outputTensor,
                                     out_dev->GetMem());

        if(inflags.GetValueInt("time") == 1)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);

            printf("GPU Kernel Time Forward Conv. Bias Elapsed: %f ms\n", time);
        }
    }

    bool is_int8 = data_type == miopenInt8 || data_type == miopenInt8x4;
    if(is_int8)
        out_dev->FromGPU(GetStream(), out_int8.data());
    else
        out_dev->FromGPU(GetStream(), out.data.data());

    if(inflags.GetValueInt("dump_output"))
    {
        if(is_int8)
            dumpBufferToFile<float>("dump_fwd_out_gpu.bin", out_int8.data(), out_int8.size());
        else
            dumpBufferToFile<Tgpu>("dump_fwd_out_gpu.bin", out.data.data(), out.data.size());
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(miopen::deref(convDesc).mode == miopenTranspose)
    {
        cpu_convolution_backward_data(miopen::deref(convDesc).GetSpatialDimension(),
                                      outhost,
                                      wei,
                                      in,
                                      miopen::deref(convDesc).GetConvPads(),
                                      miopen::deref(convDesc).GetConvStrides(),
                                      miopen::deref(convDesc).GetConvDilations(),
                                      miopen::deref(convDesc).GetGroupCount());

        if(inflags.GetValueInt("bias") != 0)
        {
            cpu_bias_forward(outhost, b);
        }
    }
    else
    {
        cpu_convolution_forward(miopen::deref(convDesc).GetSpatialDimension(),
                                in,
                                wei,
                                outhost,
                                miopen::deref(convDesc).GetConvPads(),
                                miopen::deref(convDesc).GetConvStrides(),
                                miopen::deref(convDesc).GetConvDilations(),
                                miopen::deref(convDesc).GetGroupCount());

        if(inflags.GetValueInt("bias") != 0)
        {
            outhost.par_for_each([&](auto out_n_id, auto out_k_id, auto... out_spatial_id_pack) {
                outhost(out_n_id, out_k_id, out_spatial_id_pack...) =
                    double(outhost(out_n_id, out_k_id, out_spatial_id_pack...)) +
                    double(b.data[out_k_id]);
            });
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>("dump_fwd_out_cpu.bin", outhost.data.data(), outhost.data.size());
    }

    TrySaveVerificationCache(GetVCacheFwdOutBasename(), outhost.data);
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::FindBackwardData(int& ret_algo_count,
                                             int request_algo_count,
                                             std::vector<miopenConvAlgoPerf_t>& perf_results)
{
    return miopenFindConvolutionBackwardDataAlgorithm(
        GetHandle(),
        outputTensor,
        dout_dev->GetMem(),
        weightTensor,
        wei_dev->GetMem(),
        convDesc,
        inputTensor,
        din_dev->GetMem(),
        request_algo_count,
        &ret_algo_count,
        perf_results.data(),
        (workspace_bwd_data_dev != nullptr) ? workspace_bwd_data_dev->GetMem() : nullptr,
        (workspace_bwd_data_dev != nullptr) ? workspace_bwd_data_dev->GetSize() : 0,
        (inflags.GetValueInt("search") == 1) ? true : false);
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::FindBackwardWeights(int& ret_algo_count,
                                                int request_algo_count,
                                                std::vector<miopenConvAlgoPerf_t>& perf_results)
{

    miopenFindConvolutionBackwardWeightsAlgorithm(
        GetHandle(),
        outputTensor,
        dout_dev->GetMem(),
        inputTensor,
        in_dev->GetMem(),
        convDesc,
        weightTensor,
        wei_dev->GetMem(),
        request_algo_count,
        &ret_algo_count,
        perf_results.data(),
        (workspace_bwd_weights_dev != nullptr) ? workspace_bwd_weights_dev->GetMem() : nullptr,
        (workspace_bwd_weights_dev != nullptr) ? workspace_bwd_weights_dev->GetSize() : 0,
        (inflags.GetValueInt("search") == 1) ? true : false);

    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardGPU()
{
    if(!(bwd_allowed || wrw_allowed))
        return 0;

    int ret_algo_count;
    int request_algo_count = 2;
    std::vector<miopenConvAlgoPerf_t> perf_results_data(request_algo_count);

    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    float alpha = static_cast<float>(1), beta = static_cast<float>(0);
    int ret = 0;

    Timer t;
    START_TIME

    if(bwd_allowed)
    {
        FindBackwardData(ret_algo_count, request_algo_count, perf_results_data);

        if(ret_algo_count == 0)
            throw std::runtime_error("Find Backward Data Conv. ret_algo_count == 0");

        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            ret = miopenConvolutionBackwardData(
                GetHandle(),
                &alpha,
                outputTensor,
                dout_dev->GetMem(),
                weightTensor,
                wei_dev->GetMem(),
                convDesc,
                perf_results_data[0].bwd_data_algo,
                &beta,
                inputTensor,
                din_dev->GetMem(),
                (workspace_bwd_data_dev != nullptr) ? workspace_bwd_data_dev->GetMem() : nullptr,
                perf_results_data[0].memory);

            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            kernel_total_time += time;
            if(i == 0)
                kernel_first_time = time;
        }

        if(inflags.GetValueInt("time") == 1)
        {
            STOP_TIME
            if(WALL_CLOCK)
                printf("Wall-clock Time Backward Data Conv. Elapsed: %f ms\n",
                       t.gettime_ms() / inflags.GetValueInt("iter"));

            int iter = inflags.GetValueInt("iter");
            float kernel_average_time =
                iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;

            printf("MIOpen Backward Data Conv. Algorithm: %d\n",
                   perf_results_data[0].bwd_data_algo);
            printf("GPU Kernel Time Backward Data Conv. Elapsed: %f ms (average)\n",
                   kernel_average_time);

            int in_n, in_c, in_h, in_w;
            std::tie(in_n, in_c, in_h, in_w) =
                miopen::tien<4>(miopen::deref(inputTensor).GetLengths());
            int wei_c, wei_n, wei_h, wei_w;
            std::tie(wei_c, wei_n, wei_h, wei_w) =
                miopen::tien<4>(miopen::deref(weightTensor).GetLengths());
            int out_n, out_c, out_h, out_w;
            std::tie(out_n, out_c, out_h, out_w) =
                miopen::tien<4>(miopen::deref(outputTensor).GetLengths());

            size_t flopCnt     = 2L * in_n * in_c * in_h * in_w * wei_h * wei_w * out_c;
            size_t weightBytes = wei_n * wei_c * wei_h * wei_w *
                                 miopen::GetTypeSize(miopen::deref(weightTensor).GetType());
            size_t inputBytes =
                in_n * in_c * out_c * miopen::GetTypeSize(miopen::deref(inputTensor).GetType());
            size_t readBytes = inputBytes + weightBytes;

            size_t outputBytes = 1.0 * out_n * out_c * out_h * out_w *
                                 miopen::GetTypeSize(miopen::deref(outputTensor).GetType());

            printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
                   "GB/s, timeMs\n");
            printf(
                "stats: %s%dx%du%d, %u, %u, %u, %u, %u, %u, %u,  %zu, %zu, %zu, %.0f, %.0f, %f\n",
                "bwdd-conv",
                wei_h,
                wei_w,
                miopen::deref(convDesc).GetConvStrides()[0],
                in_n,
                in_c,
                wei_h,
                wei_w,
                out_c,
                out_h,
                out_w,
                flopCnt,
                readBytes,
                outputBytes,
                flopCnt / kernel_average_time / 1e6,
                (readBytes + outputBytes) / kernel_average_time / 1e6,
                kernel_average_time);
        }

        din_dev->FromGPU(GetStream(), din.data());
    }

    if(wrw_allowed)
    {
        std::vector<miopenConvAlgoPerf_t> perf_results_weights(request_algo_count);

        FindBackwardWeights(ret_algo_count, request_algo_count, perf_results_weights);

        if(ret_algo_count == 0)
            throw std::runtime_error("Find Backward Weights Conv. ret_algo_count == 0");

        kernel_total_time = 0.0;
        kernel_first_time = 0.0;

        const auto wrw_algo      = perf_results_weights[0].bwd_weights_algo;
        const auto wrw_workspace = perf_results_weights[0].memory;
        is_wrw_winograd          = (wrw_algo == miopenConvolutionBwdWeightsAlgoWinograd);

        START_TIME
        for(int i = 0; i < inflags.GetValueInt("iter"); i++)
        {
            ret = miopenConvolutionBackwardWeights(GetHandle(),
                                                   &alpha,
                                                   outputTensor,
                                                   dout_dev->GetMem(),
                                                   inputTensor,
                                                   in_dev->GetMem(),
                                                   convDesc,
                                                   wrw_algo,
                                                   &beta,
                                                   weightTensor,
                                                   dwei_dev->GetMem(),
                                                   (workspace_bwd_weights_dev != nullptr)
                                                       ? workspace_bwd_weights_dev->GetMem()
                                                       : nullptr,
                                                   wrw_workspace);

            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            kernel_total_time += time;
            if(i == 0)
                kernel_first_time = time;
        }

        if(inflags.GetValueInt("time") == 1)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);

            STOP_TIME
            if(WALL_CLOCK)
                printf("Wall-clock Time Backward Weights Conv. Elapsed: %f ms\n",
                       t.gettime_ms() / inflags.GetValueInt("iter"));

            int iter = inflags.GetValueInt("iter");
            float kernel_average_time =
                iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;

            printf("MIOpen Backward Weights Conv. Algorithm: %d\n", wrw_algo);
            printf("GPU Kernel Time Backward Weights Conv. Elapsed: %f ms (average)\n",
                   kernel_average_time);

            int in_n, in_c, in_h, in_w;
            std::tie(in_n, in_c, in_h, in_w) =
                miopen::tien<4>(miopen::deref(inputTensor).GetLengths());
            int wei_c, wei_n, wei_h, wei_w;
            std::tie(wei_c, wei_n, wei_h, wei_w) =
                miopen::tien<4>(miopen::deref(weightTensor).GetLengths());
            int out_n, out_c, out_h, out_w;
            std::tie(out_n, out_c, out_h, out_w) =
                miopen::tien<4>(miopen::deref(outputTensor).GetLengths());

            size_t flopCnt     = 2L * in_n * in_c * in_h * in_w * wei_h * wei_w * out_c;
            size_t readBytes   = 0;
            size_t outputBytes = 0;

            printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
                   "GB/s, timeMs\n");
            printf(
                "stats: %s%dx%du%d, %u, %u, %u, %u, %u, %u, %u,  %zu, %zu, %zu, %.0f, %.0f, %f\n",
                "bwdw-conv",
                wei_h,
                wei_w,
                miopen::deref(convDesc).GetConvStrides()[0],
                in_n,
                in_c,
                out_h,
                out_w,
                wei_h,
                wei_w,
                out_c,
                flopCnt,
                readBytes,
                outputBytes,
                flopCnt / kernel_average_time / 1e6,
                (readBytes + outputBytes) / kernel_average_time / 1e6,
                kernel_average_time);
        }
        dwei_dev->FromGPU(GetStream(), dwei.data());

        if(workspace_bwd_weights_dev != nullptr)
        {
            if(wrw_algo == miopenConvolutionBwdWeightsAlgoGEMM)
            {
                workspace_bwd_weights_dev->FromGPU(GetStream(), workspace_bwd_weights.data());
            }
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        if(bwd_allowed)
            dumpBufferToFile<Tgpu>("dump_bwd_din_gpu.bin", din.data(), din.size());
        if(wrw_allowed)
            dumpBufferToFile<Tgpu>("dump_bwd_dwei_gpu.bin", dwei.data(), dwei.size());
    }

    if(inflags.GetValueInt("bias") != 0)
    {
        ret = miopenConvolutionBackwardBias(GetHandle(),
                                            &alpha,
                                            outputTensor,
                                            dout_dev->GetMem(),
                                            &beta,
                                            biasTensor,
                                            db_dev->GetMem());

        if(inflags.GetValueInt("time") == 1)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            printf("GPU Kernel Time Backward Bias Conv. Elapsed: %f ms\n", time);
        }

        db_dev->FromGPU(GetStream(), db.data());
        if(inflags.GetValueInt("dump_output"))
        {
            dumpBufferToFile<Tgpu>("dump_bwd_db_gpu.bin", db.data(), db.size());
        }
    }
    return ret;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardWeightsCPU()
{
    if(miopen::deref(convDesc).mode == miopenTranspose)
    {
        cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                        dout,
                                        dwei_host,
                                        in,
                                        miopen::deref(convDesc).GetConvPads(),
                                        miopen::deref(convDesc).GetConvStrides(),
                                        miopen::deref(convDesc).GetConvDilations(),
                                        miopen::deref(convDesc).GetGroupCount());
    }
    else
    {
        cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                        in,
                                        dwei_host,
                                        dout,
                                        miopen::deref(convDesc).GetConvPads(),
                                        miopen::deref(convDesc).GetConvStrides(),
                                        miopen::deref(convDesc).GetConvDilations(),
                                        miopen::deref(convDesc).GetGroupCount());
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>(
            "dump_bwd_dwei_cpu.bin", dwei_host.data.data(), dwei_host.data.size());
    }

    TrySaveVerificationCache(GetVCacheBwdWeightBasename(), dwei_host.data);
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardDataCPU()
{
    if(miopen::deref(convDesc).mode == miopenTranspose)
    {
        cpu_convolution_forward(miopen::deref(convDesc).GetSpatialDimension(),
                                dout,
                                wei,
                                din_host,
                                miopen::deref(convDesc).GetConvPads(),
                                miopen::deref(convDesc).GetConvStrides(),
                                miopen::deref(convDesc).GetConvDilations(),
                                miopen::deref(convDesc).GetGroupCount());
    }
    else
    {
        cpu_convolution_backward_data(miopen::deref(convDesc).GetSpatialDimension(),
                                      din_host,
                                      wei,
                                      dout,
                                      miopen::deref(convDesc).GetConvPads(),
                                      miopen::deref(convDesc).GetConvStrides(),
                                      miopen::deref(convDesc).GetConvDilations(),
                                      miopen::deref(convDesc).GetGroupCount());
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>("dump_bwd_din_cpu.bin", din_host.data.data(), din_host.data.size());
    }

    TrySaveVerificationCache(GetVCacheBwdDataBasename(), din_host.data);
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardBiasCPU()
{
    cpu_bias_backward_data(dout, db_host);

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>("dump_bwd_db_cpu.bin", db_host.data.data(), db_host.data.size());
    }

    TrySaveVerificationCache(GetVCacheBiasBwdDataBasename(), db_host.data);
    return 0;
}

template <typename Tgpu, typename Tref>
std::string ConvDriver<Tgpu, Tref>::GetVerificationCacheFileName() const
{
    std::ostringstream ss;

    miopenConvolutionMode_t mode;

    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> pads(spatial_dim);
    std::vector<int> conv_strides(spatial_dim);
    std::vector<int> conv_dilations(spatial_dim);
    std::vector<int> trans_output_pads(spatial_dim);

    miopenGetConvolutionNdDescriptor(convDesc,
                                     spatial_dim,
                                     &spatial_dim,
                                     pads.data(),
                                     conv_strides.data(),
                                     conv_dilations.data(),
                                     &mode);

    auto get_datatype_string = [](auto type) {
        if(std::is_same<decltype(type), int8_t>::value)
        {
            return "int8";
        }
        else if(std::is_same<decltype(type), float16>::value)
        {
            return "float16";
        }
        else if(std::is_same<decltype(type), float>::value)
        {
            return "float";
        }
        else if(std::is_same<decltype(type), double>::value)
        {
            return "double";
        }
        else
        {
            MIOPEN_THROW("unknown data type");
        }
    };

    ss << mode;
    ss << "_" << spatial_dim;
    ss << "_" << miopen::deref(convDesc).paddingMode;
    ss << "_" << miopen::deref(convDesc).GetGroupCount();
    miopen::LogRange(ss << "_", miopen::deref(inputTensor).GetLengths(), "x");
    miopen::LogRange(ss << "_", miopen::deref(weightTensor).GetLengths(), "x");
    miopen::LogRange(ss << "_", pads, "x");
    miopen::LogRange(ss << "_", conv_strides, "x");
    miopen::LogRange(ss << "_", conv_dilations, "x");
    miopen::LogRange(ss << "_", trans_output_pads, "x");
    ss << "_" << inflags.GetValueInt("pad_val");
    ss << "_" << inflags.GetValueInt("bias");
    ss << "_"
       << "GPU" << get_datatype_string(Tgpu{});
    ss << "_"
       << "REF" << get_datatype_string(Tref{});

    return ss.str();
}

template <typename Tgpu, typename Tref>
bool ConvDriver<Tgpu, Tref>::TryReadVerificationCache(const std::string& file_name,
                                                      miopenTensorDescriptor_t& tensorDesc,
                                                      Tref* data) const
{
    const auto verification_cache_path = inflags.GetValueStr("verification_cache");

    if(!verification_cache_path.empty())
    {
        const auto file_path =
            verification_cache_path + "/" + file_name + "_" + GetVerificationCacheFileName();

        if(std::ifstream(file_path).good())
        {
            if(readBufferFromFile<Tref>(data, GetTensorSize(tensorDesc), file_path.c_str()))
            {
                return true;
            }
        }
    }

    return false;
}

template <typename Tgpu, typename Tref>
void ConvDriver<Tgpu, Tref>::TrySaveVerificationCache(const std::string& file_name,
                                                      std::vector<Tref>& data) const
{
    const auto verification_cache_path = inflags.GetValueStr("verification_cache");
    if(!verification_cache_path.empty())
    {
        const auto file_path =
            verification_cache_path + "/" + file_name + "_" + GetVerificationCacheFileName();
        dumpBufferToFile<Tref>(file_path.c_str(), data.data(), data.size());
    }
}

template <typename Tgpu, typename Tref>
std::string ConvDriver<Tgpu, Tref>::GetVCacheFwdOutBasename() const
{
    return "conv_fwd_out";
}

template <typename Tgpu, typename Tref>
std::string ConvDriver<Tgpu, Tref>::GetVCacheBwdDataBasename() const
{
    return "conv_bwd_dat";
}

template <typename Tgpu, typename Tref>
std::string ConvDriver<Tgpu, Tref>::GetVCacheBwdWeightBasename() const
{
    return "conv_bwd_wei";
}

template <typename Tgpu, typename Tref>
std::string ConvDriver<Tgpu, Tref>::GetVCacheBiasBwdDataBasename() const
{
    return "bias_bwd_dat";
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::VerifyForward()
{
    if(!forward_allowed)
        return 0;

    if(!TryReadVerificationCache(GetVCacheFwdOutBasename(), outputTensor, outhost.data.data()))
    {
        RunForwardCPU();
    }

    auto error = miopen::rms_range(outhost.data, out.data);
    if(data_type == miopenInt8 || data_type == miopenInt8x4)
        error = miopen::rms_range(outhost.data, out_int8);

    const Tref tolerance = ((sizeof(Tgpu) == 4 || sizeof(Tgpu) == 1) ? static_cast<Tref>(1e-6)
                                                                     : static_cast<Tref>(7e-2));
    if(!(error < tolerance))
    {
        std::cout << "Forward Convolution Failed: " << error << std::endl;
    }
    else
    {
        std::cout << "Forward Convolution Verifies on CPU and GPU (" << error << ')' << std::endl;
    }

    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::VerifyBackward()
{

    if(!(bwd_allowed || wrw_allowed))
        return 0;

    const Tref tolerance =
        ((sizeof(Tgpu) == 4) ? static_cast<Tref>(1e-6) : static_cast<Tref>(7e-2));

    if(bwd_allowed)
    {
        if(!TryReadVerificationCache(GetVCacheBwdDataBasename(), inputTensor, din_host.data.data()))
        {
            RunBackwardDataCPU();
        }

        auto error_data = miopen::rms_range(din_host.data, din);

        if(!(error_data < tolerance))
        {
            std::cout << "Backward Convolution Data Failed: " << error_data << std::endl;
        }
        else
        {
            std::cout << "Backward Convolution Data Verifies on CPU and GPU (" << error_data << ')'
                      << std::endl;
        }
    }

    if(wrw_allowed)
    {
        if(!TryReadVerificationCache(
               GetVCacheBwdWeightBasename(), weightTensor, dwei_host.data.data()))
        {
            RunBackwardWeightsCPU();
        }

        // Winograd algorithm has worse precision than Direct and Gemm.
        // Winograd-specific precision loss is roughly 2+2 bits.
        // Affects only WrW FP32 for now.
        auto tolerance_wrw = tolerance;
        if(is_wrw_winograd && std::is_same<Tgpu, float>::value)
            tolerance_wrw *= 16;

        auto error_weights = miopen::rms_range(dwei_host.data, dwei);
        if(!(error_weights < tolerance_wrw))
        {
            std::cout << "Backward Convolution Weights Failed: " << error_weights << std::endl;
        }
        else
        {
            std::cout << "Backward Convolution Weights Verifies on CPU and GPU (" << error_weights
                      << ')' << std::endl;
        }
    }

    if(inflags.GetValueInt("bias") != 0)
    {
        if(!TryReadVerificationCache(
               GetVCacheBiasBwdDataBasename(), biasTensor, db_host.data.data()))
        {
            RunBackwardBiasCPU();
        }

        auto error_bias = miopen::rms_range(db_host.data, db);
        if(!(error_bias < tolerance))
        {
            std::cout << "Backward Convolution Bias Failed: " << error_bias << std::endl;
        }
        else
        {
            std::cout << "Backward Convolution Bias Verifies on CPU and GPU (" << error_bias << ')'
                      << std::endl;
        }
    }

    return 0;
}

#endif // GUARD_MIOPEN_CONV_DRIVER_HPP
