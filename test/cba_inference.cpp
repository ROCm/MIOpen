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
#include "test.hpp"
#include <array>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/convolution.hpp>
#include <miopen/activ.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>

// #include "network_data.hpp"
#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <miopen/stringutils.hpp>
#include <miopen/conv_batch_norm_activ.hpp>
#include <miopen/batch_norm_activ.hpp>
#include <miopen/direct_conv_ocl.hpp>

template <class T>
void convHostForward(const tensor<T>& input,
                     tensor<T>& output,
                     const tensor<T>& weights,
                     const int bias_mode,
                     const tensor<T>& bias,
                     const miopenConvolutionDescriptor_t convDesc)
{

    int in_n, in_c, in_h, in_w;
    int in_nstride, in_cstride, in_hstride, in_wstride;
    std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(input.desc.GetLengths());
    std::tie(in_nstride, in_cstride, in_hstride, in_wstride) =
        miopen::tien<4>(input.desc.GetStrides());

    int wei_n, wei_c, wei_h, wei_w;
    int wei_nstride, wei_cstride, wei_hstride, wei_wstride;
    std::tie(wei_n, wei_c, wei_h, wei_w) = miopen::tien<4>(weights.desc.GetLengths());
    std::tie(wei_nstride, wei_cstride, wei_hstride, wei_wstride) =
        miopen::tien<4>(weights.desc.GetStrides());

    int out_n, out_c, out_h, out_w;
    int out_nstride, out_cstride, out_hstride, out_wstride;
    std::tie(out_n, out_c, out_h, out_w) = miopen::tien<4>(output.desc.GetLengths());
    std::tie(out_nstride, out_cstride, out_hstride, out_wstride) =
        miopen::tien<4>(output.desc.GetStrides());

    int u, v, pad_h, pad_w, dilation_h, dilation_w;
    miopenConvolutionMode_t mode;
    miopenPaddingMode_t pmode = miopen::deref(convDesc).paddingMode;
    miopenGetConvolutionDescriptor(
        convDesc, &mode, &pad_h, &pad_w, &u, &v, &dilation_h, &dilation_w);

    if(pmode == miopenPaddingSame)
    {
        pad_h = (in_h % u == 0) ? (std::max((wei_h - u), 0)) : (std::max((wei_h - (in_h % u)), 0));
        pad_w = (in_w % v == 0) ? (std::max((wei_w - v), 0)) : (std::max((wei_w - (in_w % v)), 0));
        pad_h /= 2;
        pad_w /= 2;
    }
    else if(pmode == miopenPaddingValid)
    {
        pad_h = 0;
        pad_w = 0;
    }

    if(out_h <= 0 || out_w <= 0)
        MIOPEN_THROW("Invalid Test Case: Check Output Dimension.");

    /*    printf("CPU Conv input NCHW: %d, %d, %d, %d\n", in_n, in_c, in_h, in_w);
        printf("CPU Conv output NCHW: %d, %d, %d, %d\n", out_n, out_c, out_h, out_w);
        printf("CPU weights NCHW: %d, %d, %d, %d\n", wei_n, wei_c, wei_h, out_w);*/

    for(int o = 0; o < out_n; o++)
    { // mini-batch size
        for(int w = 0; w < out_c; w++)
        { // out_channels (num filters)
            for(int i = 0; i < out_h; i++)
            { // output_height (from getforwardoutputdim())
                int in_off_h = i * u;
                for(int j = 0; j < out_w; j++)
                { // output_width (from getforwardoutputdim())
                    T acc        = static_cast<T>(0);
                    int in_off_w = j * v;
                    for(int k = 0; k < in_c; k++)
                    { // in_channels (RGB)
                        for(int x = 0; x < wei_h; x++)
                        {
                            int in_x = in_off_h - pad_h + x * dilation_h;
                            if(in_x >= 0 && in_x < in_h)
                            {
                                for(int y = 0; y < wei_w; y++)
                                {
                                    int in_y = in_off_w - pad_w + y * dilation_w;
                                    if(in_y >= 0 && in_y < in_w)
                                    {
                                        acc +=
                                            static_cast<T>(input[o * in_nstride + k * in_cstride +
                                                                 in_x * in_w + in_y]) *
                                            static_cast<T>(weights(w, k, x, y));
                                    }
                                }
                            }
                        }
                    }
                    acc = bias_mode != 0 ? acc + static_cast<T>(bias[w]) : acc;
                    output[o * out_nstride + w * out_cstride + i * out_hstride + j] = acc;
                }
            }
        }
    }

    return;
}

template <class T>
void batchNormSpatialHostInference(const tensor<T>& input,
                                   tensor<T>& output,
                                   const tensor<T>& scale,
                                   const tensor<T>& bias,
                                   double epsilon,
                                   const tensor<T>& estimatedMean,
                                   const tensor<T>& estimatedVariance)
{

    int n_batches, channels, height, width;
    std::tie(n_batches, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
    par_for(channels, 1, [&](int cidx) { // via channel
        double mean      = estimatedMean(1, cidx, 1, 1);
        double variance  = estimatedVariance(1, cidx, 1, 1);
        double invertVar = 1.0 / sqrt(variance + epsilon);
        // process the batch per channel
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                for(int bidx = 0; bidx < n_batches; bidx++)
                { // via mini_batch
                    double elemStd = input(bidx, cidx, row, column) - mean;
                    double inhat   = elemStd * invertVar;
                    output(bidx, cidx, row, column) =
                        scale(1, cidx, 1, 1) * inhat + bias(1, cidx, 1, 1);
                }
            }
        }
    });
    return;
}

template <class T>
void batchNormPerActivHostInference(const tensor<T>& input,
                                    tensor<T>& output,
                                    const tensor<T>& scale,
                                    const tensor<T>& bias,
                                    double epsilon,
                                    const tensor<T>& estimatedMean,
                                    const tensor<T>& estimatedVariance)
{
    int n_batches, channels, height, width;
    std::tie(n_batches, channels, height, width) = miopen::tien<4>(input.desc.GetLengths());
    par_for(channels, 1, [&](int cidx) { // via channel
        for(int row = 0; row < height; row++)
        { // via rows
            for(int column = 0; column < width; column++)
            { // via columns
                // apply down the n_batch dimension
                double mean       = estimatedMean(1, cidx, row, column);
                double variance   = estimatedVariance(1, cidx, row, column);
                double elemInvVar = 1.0 / double(sqrt(variance + epsilon));
                for(int bidx = 0; bidx < n_batches; bidx++)
                { // via mini_batch
                    // per (x-dims) channel load a block of data into LDS
                    double elemStd = input(bidx, cidx, row, column) - mean;
                    double inhat   = elemStd * elemInvVar;
                    output(bidx, cidx, row, column) =
                        scale(1, cidx, row, column) * inhat + bias(1, cidx, row, column);
                }
            }
        }
    });
    return;
}

template <class T>
void activationHostInfererence(miopenActivationMode_t activMode,
                               T gamma,
                               T beta,
                               T alpha,
                               const tensor<T>& input,
                               tensor<T>& output)
{

    std::function<T(T)> f;

    switch(activMode)
    {
    case miopenActivationPASTHRU: //  x
        f = [=](T x) { return x; };
        break;
    case miopenActivationLOGISTIC: // 1 / (1 + e^-x)  //Sigmoid
        f = [=](T x) { return 1 / (1 + std::exp(-x)); };
        break;
    case miopenActivationTANH: // beta * tanh(alpha * x)
        f = [=](T x) { return beta * std::tanh(alpha * x); };
        break;
    case miopenActivationRELU: // max(0, x)
        f = [=](T x) { return (x > 0) ? x : 0; };
        break;
    case miopenActivationSOFTRELU: //  log(1 + e^x)   // bonomial normal log likelihood
        f = [=](T x) { return std::log1p(std::exp(x)); };
        break;
    case miopenActivationABS: //  abs(x)
        f = [=](T x) { return std::abs(x); };
        break;
    case miopenActivationPOWER: // (alpha + beta * x) ^ gamma
        f = [=](T x) {
            T v = alpha + beta * x;
            return v <= std::numeric_limits<T>::epsilon() ? 0 : pow(v, gamma);
        };
        break;
    case miopenActivationCLIPPEDRELU: // min(alpha, max(0, x))
        f = [=](T x) { return std::min(alpha, std::max(T(0), x)); };
        break;
    case miopenActivationLEAKYRELU: // alpha * x | x<=0; x | x>0
        f = [=](T x) { return (x > 0) ? x : x * alpha; };
        break;
    case miopenActivationELU: // alpah * (exp(x)-1) | x<=0; x | x>0
        f = [=](T x) { return (x > 0) ? x : alpha * std::expm1(x); };
        break;
        // default: printf("ERROR: unknown neuron type: %d\n", activMode); break;
    }

    par_for(input.desc.GetElementSize(), 1, [&](int index) { output[index] = f(input[index]); });
    return;
}

template <class T>
tensor<T> get_output_tensor(const miopen::ConvolutionDescriptor& filter,
                            const tensor<T>& input,
                            const tensor<T>& weights)
{
    return tensor<T>{filter.GetForwardOutputTensor(input.desc, weights.desc)};
}

// DLOWELL I'll resuse this for all ordered combinations
// of convolution + bias + batchnorm + activations
template <class T>
struct verify_forward_conv_bias
{
    tensor<T> input;
    tensor<T> weights;
    tensor<T> bias;
    miopenConvolutionDescriptor_t filter;
    miopenTensorDescriptor_t inputDesc{};
    miopenTensorDescriptor_t weightsDesc{};
    miopenTensorDescriptor_t outputDesc{};
    miopenTensorDescriptor_t biasDesc{};
    // tensor<T> output;
    // miopen::ConvolutionDescriptor filter;
    int bias_mode;

    // using conv_base<T>::search; //DLOWELL not needed right now
    verify_forward_conv_bias(/*miopenHandle_t phandle,*/ tensor<T>& pinput,
                             tensor<T>& pweights,
                             miopen::ConvolutionDescriptor& pfilter,
                             tensor<T>& pbias /*, int psearch = 0 */)
    {
        input       = pinput;
        inputDesc   = &pinput.desc;
        weights     = pweights;
        weightsDesc = &pweights.desc;
        bias        = pbias;
        biasDesc    = &pbias.desc;
        filter      = &pfilter;
        // search  = psearch;
    }

    tensor<T> cpu() const
    {
        // If we are using convolutions as the base, we can calculate the
        auto rout = get_output_tensor(miopen::deref(filter), input, weights);
        convHostForward(input, rout, weights, 1, bias, filter);
        return rout;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rout     = get_output_tensor(miopen::deref(filter), input, weights);
        auto in_dev   = handle.Write(input.data);
        auto wei_dev  = handle.Write(weights.data);
        auto b_dev    = handle.Write(bias.data);
        auto out_dev  = handle.Write(rout.data);

        // DLOWELL: All of this search logic is shelved until we have more fused algos to implement
        /*        size_t workspace_size =
                    filter.ForwardGetWorkSpaceSize(handle, weights.desc, input.desc, rout.desc);

                std::vector<char> workspace(workspace_size);
                auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

                int ret_algo_count;
                miopenConvAlgoPerf_t perf;
        */
        miopenStatus_t miopenError = miopenStatusSuccess;
        miopenFusionPlanDescriptor_t fusePlanDesc;
        // miopenFusionOpDescriptor_t bNormOp;
        miopenFusionOpDescriptor_t convoOp;
        miopenFusionOpDescriptor_t biasOp;
        // miopenFusionOpDescriptor_t activOp;
        miopenOperatorArgs_t fusionArgs;

        float alpha = static_cast<float>(1), beta = static_cast<float>(0);
        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
        miopenCreateOperatorArgs(&fusionArgs);
        miopenCreateOpConvForwardAlgo(
            fusePlanDesc,
            &convoOp,
            filter,
            // DLOWELL Hardcoded. This assumes immediate mode. Needs GetAlgo.
            miopenConvolutionFwdAlgoDirect,
            weightsDesc);

        miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasDesc);
        miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev.get());
        miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, b_dev.get());

        miopenError = miopenIsFusionPlanValid(fusePlanDesc);
        if(miopenError != miopenStatusSuccess)
        {
            std::cerr << "ConvBiasInference plan not supported." << std::endl;
        }
        else
        {
            miopenExecuteFusionPlan(&handle,
                                    fusePlanDesc,
                                    inputDesc,
                                    in_dev.get(),
                                    &rout.desc,
                                    out_dev.get(),
                                    fusionArgs);
            rout.data = handle.Read<T>(out_dev, rout.data.size());
        }
        return rout;
    }

    void fail(float = 0) const { std::cout << "Forward convolution+bias: " << std::endl; }
};

// DLOWELL I'll resuse this for all ordered combinations
// of convolution + bias + batchnorm + activations
template <class T>
struct verify_forward_conv_bias_activ
{
    tensor<T> input;
    tensor<T> weights;
    tensor<T> bias;
    miopenConvolutionDescriptor_t filter;
    miopenTensorDescriptor_t inputDesc{};
    miopenTensorDescriptor_t weightsDesc{};
    miopenTensorDescriptor_t outputDesc{};
    miopenTensorDescriptor_t biasDesc{};
    miopenActivationDescriptor_t activDesc{};
    // miopenHandle_t handle;
    // tensor<T> output;
    // miopen::ConvolutionDescriptor filter;
    int bias_mode;

    // using conv_base<T>::search; //DLOWELL not needed right now
    verify_forward_conv_bias_activ(/*miopenHandle_t phandle,*/ tensor<T>& pinput,
                                   tensor<T>& pweights,
                                   miopen::ConvolutionDescriptor& pfilter,
                                   tensor<T>& pbias,
                                   miopenActivationDescriptor_t& pactivDesc /*, int psearch = 0 */)
    {
        input       = pinput;
        inputDesc   = &pinput.desc;
        weights     = pweights;
        weightsDesc = &pweights.desc;
        bias        = pbias;
        biasDesc    = &pbias.desc;
        filter      = &pfilter;
        activDesc   = pactivDesc;
        // search  = psearch;
    }

    tensor<T> cpu() const
    {
        // If we are using convolutions as the base, we can calculate the
        auto rout = get_output_tensor(miopen::deref(filter), input, weights);
        convHostForward(input, rout, weights, 1, bias, filter);
        return rout;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto rout     = get_output_tensor(miopen::deref(filter), input, weights);
        auto in_dev   = handle.Write(input.data);
        auto wei_dev  = handle.Write(weights.data);
        auto b_dev    = handle.Write(bias.data);
        auto out_dev  = handle.Write(rout.data);

        // DLOWELL: All of this search logic is shelved until we have more fused algos to implement
        /*        size_t workspace_size =
                    filter.ForwardGetWorkSpaceSize(handle, weights.desc, input.desc, rout.desc);

                std::vector<char> workspace(workspace_size);
                auto workspace_dev = workspace_size != 0 ? handle.Write(workspace) : nullptr;

                int ret_algo_count;
                miopenConvAlgoPerf_t perf;
        */
        miopenStatus_t miopenError = miopenStatusSuccess;
        miopenFusionPlanDescriptor_t fusePlanDesc;
        // miopenFusionOpDescriptor_t bNormOp;
        miopenFusionOpDescriptor_t convoOp;
        miopenFusionOpDescriptor_t biasOp;
        miopenFusionOpDescriptor_t activOp;
        miopenOperatorArgs_t fusionArgs;

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        float alpha = static_cast<float>(1), beta = static_cast<float>(0);
        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
        miopenCreateOperatorArgs(&fusionArgs);
        miopenCreateOpConvForwardAlgo(
            fusePlanDesc,
            &convoOp,
            filter,
            // DLOWELL Hardcoded. This assumes immediate mode. Needs GetAlgo.
            miopenConvolutionFwdAlgoDirect,
            weightsDesc);
        miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasDesc);
        miopenCreateOpActivationForward(fusePlanDesc, &activOp, activ_mode);
        miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev.get());
        miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, b_dev.get());

        miopenSetOpArgsActivForward(
            fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);

        miopenError = miopenIsFusionPlanValid(fusePlanDesc);
        if(miopenError != miopenStatusSuccess)
        {
            std::cerr << "Conv+Bias+Activation Inference plan not supported." << std::endl;
        }
        else
        {
            miopenExecuteFusionPlan(&handle,
                                    fusePlanDesc,
                                    inputDesc,
                                    in_dev.get(),
                                    &rout.desc,
                                    out_dev.get(),
                                    fusionArgs);
            rout.data = handle.Read<T>(out_dev, rout.data.size());
        }

        return rout;
    }

    void fail(float = 0) const
    {
        std::cout << "Forward convolution+bias+activation: " << std::endl;
    }
};

template <class T>
struct cbna_fusion_driver : test_driver
{
    tensor<T> input;
    tensor<T> weights;
    miopen::ConvolutionDescriptor filter;
    miopenActivationDescriptor_t activDesc;
    miopenActivationMode_t activ_mode = miopenActivationRELU;
    int amode                         = 0;
    bool tactiv{};
    bool bias_mode = true;
    std::string conv_mode;
    std::string pad_mode;
    bool enable_backward_weights = false;
    bool do_backward_data        = true;
    int search                   = 0;
    unsigned long max_value      = miopen_type<T>{} == miopenHalf ? 5 : 17;
    double alpha, beta, gamma;

    std::unordered_map<std::string, miopenConvolutionMode_t> cmode_lookup = {
        {"CONV", miopenConvolution}}; //, {"TRANS", miopenTranspose}};

    std::unordered_map<std::string, miopenPaddingMode_t> pmode_lookup = {
        {"SAME", miopenPaddingSame},
        {"VALID", miopenPaddingValid},
        {"DEFAULT", miopenPaddingDefault}};

    cbna_fusion_driver()
    {
        add(input, "input", get_input_tensor());
        add(weights, "weights", get_weights_tensor());
        add(filter, "filter", generate_data(get_filters()));
        add(alpha, "alpha", generate_data({1. /*, 0.1*/}));
        add(beta, "beta", generate_data({0. /*, 0.5*/}));
        add(gamma, "gamma", generate_data({1. /*, 0.1*/}));
        /*        add(enable_backward_weights, "enable-backward-weights", flag());
                add(do_backward_data, "disable-backward-data", set_value(false));
                add(search, "search", set_value(1));*/
        add(bias_mode, "bmode", generate_data({true, false}));
        //       add(conv_mode, "cmode", generate_data({"conv"}/*, "trans"}*/)); //fusion can't
        //       handle trans right now
        add(pad_mode, "pmode", generate_data({"default" /*, "same", "valid"*/}));
        add(tactiv, "test_activ", generate_data({false, true}));
        add(amode, "amode", generate_data({0 /*3,4,5,6,7,8,9,0,1,2*/}));
    }

    std::vector<miopen::ConvolutionDescriptor> get_filters()
    {
        return {miopen::ConvolutionDescriptor{0, 0, 1, 1} /*,
                miopen::ConvolutionDescriptor{0, 0, 2, 2},
                miopen::ConvolutionDescriptor{1, 1, 1, 1},
                miopen::ConvolutionDescriptor{1, 1, 2, 2},
                miopen::ConvolutionDescriptor{2, 2, 1, 1},
                miopen::ConvolutionDescriptor{3, 3, 2, 2}*/};
    };

    void run()
    {

        switch(amode)
        {
        case 0: activ_mode = miopenActivationPASTHRU; break;
        case 1: activ_mode = miopenActivationLOGISTIC; break;
        case 2: activ_mode = miopenActivationTANH; break;
        case 3: activ_mode = miopenActivationRELU; break;
        case 4: activ_mode = miopenActivationSOFTRELU; break;
        case 5: activ_mode = miopenActivationABS; break;
        case 6: activ_mode = miopenActivationPOWER; break;
        case 7: activ_mode = miopenActivationCLIPPEDRELU; break;
        case 8: activ_mode = miopenActivationLEAKYRELU; break;
        case 9: activ_mode = miopenActivationELU;
        }

        int input_c, input_h, input_w, wei_c, wei_k, wei_h, wei_w;
        std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tien<4>(weights.desc.GetLengths());
        std::tie(std::ignore, input_c, input_h, input_w) = miopen::tien<4>(input.desc.GetLengths());

        filter.mode        = cmode_lookup[miopen::ToUpper(conv_mode)];
        filter.paddingMode = pmode_lookup[miopen::ToUpper(pad_mode)];

        auto u            = filter.u;
        auto v            = filter.v;
        auto fpad_h       = filter.pad_h;
        auto fpad_w       = filter.pad_w;
        auto fpaddingMode = filter.paddingMode;

        if(tactiv)
        {
            miopenCreateActivationDescriptor(&activDesc);
            miopenSetActivationDescriptor(activDesc, activ_mode, alpha, beta, gamma);
        }

        if(((filter.mode == miopenTranspose) && (input_c == wei_k)) ||
           ((filter.mode == miopenConvolution) && (input_c == wei_c)))
        {
            if(fpaddingMode == miopenPaddingSame)
            {

                if(u == 0 || v == 0)
                    return;
                auto _pad_h = (input_h % u == 0)
                                  ? (std::max(static_cast<int>(wei_h - u), 0))
                                  : (std::max(static_cast<int>(wei_h - (input_h % u)), 0));
                auto _pad_w = (input_w % v == 0)
                                  ? (std::max(static_cast<int>(wei_w - v), 0))
                                  : (std::max(static_cast<int>(wei_w - (input_w % v)), 0));

                filter.pad_h = _pad_h / 2;
                filter.pad_w = _pad_w / 2;

                int out_h = std::ceil(static_cast<double>(input_h) / u);
                int out_w = std::ceil(static_cast<double>(input_w) / v);

                if(out_h <= 0 || out_w <= 0)
                    return;
            }
            else if(fpaddingMode == miopenPaddingValid)
            {
                if(u == 0 || v == 0)
                    return;
                filter.pad_h = 0;
                filter.pad_w = 0;

                int out_h = std::ceil(static_cast<double>(input_h - wei_h + 1) / u);
                int out_w = std::ceil(static_cast<double>(input_w - wei_w + 1) / v);

                if(out_h <= 0 || out_w <= 0)
                    return;
            }

            if(input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(1) &&
               wei_h > 2 * fpad_h && wei_w > 2 * fpad_w && input_h >= (2 * fpad_h + wei_h) &&
               input_w >= (2 * fpad_w + wei_w))
            {
                // printf("HERE!\n" );
                auto output = get_output_tensor(filter, input, weights);
                // std::cout << "bias_mode: " << bias_mode << std::endl;
                if(bias_mode)
                { // here we handle cb, cba, cbbna
                    auto bias =
                        tensor<T>{1, output.desc.GetLengths()[1], 1, 1}.generate(rand_gen{});
                    // auto out_p =
                    verify(verify_forward_conv_bias<T>{input, weights, filter, bias});

                    // create activation descriptor here
                    if(tactiv)
                    {
                        //  printf("Running CBA.\n");
                        verify(verify_forward_conv_bias_activ<T>{
                            input, weights, filter, bias, activDesc});
                    }
                }
                else
                {
                    if(tactiv)
                    {
                        //  printf("Running CBA.\n");
                        auto bias = tensor<T>{1, 1, 1, 1};
                        verify(verify_forward_conv_bias_activ<T>{
                            input, weights, filter, bias, activDesc});
                    }
                }
                /*for(auto& x : out_p.first)
                    x = (long(x + 19) * 2) % max_value; // Clamp big numbers
                if(do_backward_data)
                    verify(verify_backward_conv<T>{input, weights, out_p.first, filter, 0, search});
                if(enable_backward_weights or (MIOPEN_USE_MIOPENGEMM and sizeof(T) > 2))
                {
                    verify(verify_backward_weights_conv<T>{
                        input, weights, out_p.first, filter, 0, search});
                }*/
            }
        }
        if(tactiv)
            miopenDestroyActivationDescriptor(activDesc);
    }
};

int main(int argc, const char* argv[]) { test_drive<cbna_fusion_driver>(argc, argv); }
