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

#include "fusionHost.hpp"
#include <miopen/stringutils.hpp>
#include <miopen/conv_batch_norm_activ.hpp>
#include <miopen/batch_norm_activ.hpp>
#include <miopen/direct_conv_ocl.hpp>

#define MIO_CONV_ALGO_COUNT 4

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
        // Using this code section to filter out unsupported fusion plans
        auto&& handle = get_handle();
        auto rout     = get_output_tensor(miopen::deref(filter), input, weights);

        miopenFusionPlanDescriptor_t fusePlanDesc;
        miopenFusionOpDescriptor_t convoOp;
        miopenFusionOpDescriptor_t biasOp;
        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
        miopenCreateOpConvForward(fusePlanDesc, &convoOp, filter, weightsDesc);
#if 0
        miopenConvFwdAlgorithm_t sup_algos[MIO_CONV_ALGO_COUNT];
        int retAlgCount = 0;
        // Query the supported algorithms
        miopenFusionPlanConvolutionGetAlgo(
            fusePlanDesc, MIO_CONV_ALGO_COUNT, &retAlgCount, sup_algos);

        // TODO: Replace this with WinoGrad to check for wino grad supported kernels
        miopenConvFwdAlgorithm_t req_algo = miopenConvolutionFwdAlgoDirect;
        if((std::begin(sup_algos) + retAlgCount) != std::find(std::begin(sup_algos),
                                                              std::begin(sup_algos) + retAlgCount,
                                                              miopenConvolutionFwdAlgoDirect))
        {
            // should not throw
            miopenFusionPlanConvolutionSetAlgo(fusePlanDesc, req_algo);
        }
#endif

        miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasDesc);

        // Compile
        miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, fusePlanDesc);
        if(miopenError != miopenStatusSuccess)
        {
            std::cerr << "ConvBiasInference plan not supported." << std::endl;
        }
        else
        {
            // If we are using convolutions as the base, we can calculate the
            convHostForward(input, rout, weights, 1, bias, filter);
        }
        miopenDestroyFusionPlanDescriptor(fusePlanDesc);
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

        miopenFusionPlanDescriptor_t fusePlanDesc;
        miopenFusionOpDescriptor_t convoOp;
        miopenFusionOpDescriptor_t biasOp;
        miopenOperatorArgs_t fusionArgs;

        double alpha = 1., beta = 0.;
        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
        miopenCreateOperatorArgs(&fusionArgs);
        miopenCreateOpConvForward(
            fusePlanDesc,
            &convoOp,
            filter,
            // \todo : DLOWELL Hardcoded. This assumes immediate mode. Needs GetAlgo.
            // miopenConvolutionFwdAlgoDirect,
            weightsDesc);
#if 0
        miopenConvFwdAlgorithm_t sup_algos[MIO_CONV_ALGO_COUNT];
        int retAlgCount = 0;
        // Query the supported algorithms
        miopenFusionPlanConvolutionGetAlgo(
            fusePlanDesc, MIO_CONV_ALGO_COUNT, &retAlgCount, sup_algos);
        // TODO: Replace this with WinoGrad to check for wino grad supported kernels
        miopenConvFwdAlgorithm_t req_algo = miopenConvolutionFwdAlgoDirect;
        if(std::begin(sup_algos) + retAlgCount != std::find(std::begin(sup_algos),
                                                            std::begin(sup_algos) + retAlgCount,
                                                            miopenConvolutionFwdAlgoDirect))
        {
            // should not throw
            miopenFusionPlanConvolutionSetAlgo(fusePlanDesc, req_algo);
        }
#endif
        miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasDesc);

        // Compile
        miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, fusePlanDesc);
        if(miopenError != miopenStatusSuccess)
        {
            std::cerr << "ConvBiasInference plan not supported." << std::endl;
        }
        else
        {
            miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev.get());
            miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, b_dev.get());
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
    miopenConvolutionDescriptor_t filter;
    tensor<T> bias{};
    miopenTensorDescriptor_t inputDesc{};
    miopenTensorDescriptor_t weightsDesc{};
    miopenTensorDescriptor_t outputDesc{};
    miopenTensorDescriptor_t biasDesc{};
    miopenActivationDescriptor_t activDesc{};
    int bias_mode = 0;

    // using conv_base<T>::search; //DLOWELL not needed right now
    verify_forward_conv_bias_activ(tensor<T>& pinput,
                                   tensor<T>& pweights,
                                   miopen::ConvolutionDescriptor& pfilter,
                                   int pbias_mode,
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
        bias_mode   = pbias_mode;
        // search  = psearch;
    }

    tensor<T> cpu() const
    {

        auto&& handle = get_handle();
        auto rout     = get_output_tensor(miopen::deref(filter), input, weights);
        auto aout     = rout;
        std::fill(aout.begin(), aout.end(), 0.);
        miopenFusionPlanDescriptor_t fusePlanDesc;
        // miopenFusionOpDescriptor_t bNormOp;
        miopenFusionOpDescriptor_t convoOp = nullptr;
        miopenFusionOpDescriptor_t biasOp  = nullptr;
        miopenFusionOpDescriptor_t activOp = nullptr;

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
        miopenCreateOpConvForward(
            fusePlanDesc,
            &convoOp,
            filter,
            // \todo dlowell: Hardcoded right now. This assumes immediate mode. Needs GetAlgo.
            // miopenConvolutionFwdAlgoDirect,
            weightsDesc);
#if 0
        miopenConvFwdAlgorithm_t sup_algos[MIO_CONV_ALGO_COUNT];
        int retAlgCount = 0;
        // Query the supported algorithms
        miopenFusionPlanConvolutionGetAlgo(
            fusePlanDesc, MIO_CONV_ALGO_COUNT, &retAlgCount, sup_algos);
        // TODO: Replace this with WinoGrad to check for wino grad supported kernels
        miopenConvFwdAlgorithm_t req_algo = miopenConvolutionFwdAlgoDirect;
        if((std::begin(sup_algos) + retAlgCount) != std::find(std::begin(sup_algos),
                                                              std::begin(sup_algos) + retAlgCount,
                                                              miopenConvolutionFwdAlgoDirect))
        {
            // should not throw
            miopenFusionPlanConvolutionSetAlgo(fusePlanDesc, req_algo);
        }
#endif
        if(bias_mode)
            miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasDesc);

        miopenCreateOpActivationForward(fusePlanDesc, &activOp, activ_mode);

        // Compile
        miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, fusePlanDesc);
        if(miopenError != miopenStatusSuccess)
        {
            if(bias_mode)
                std::cerr << "Conv+Bias+Activation Inference plan not supported." << std::endl;
            else
                std::cerr << "Conv+Activation Inference plan not supported." << std::endl;
        }
        else
        {
            // If we are using convolutions as the base, we can calculate the
            convHostForward(input, rout, weights, 1, bias, filter);
            activationHostInfer(activ_mode,
                                activ_gamma,
                                activ_beta,
                                activ_alpha,
                                rout.data,
                                aout.data);
        }
        return aout;
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
        miopenFusionPlanDescriptor_t fusePlanDesc;
        // miopenFusionOpDescriptor_t bNormOp;
        miopenFusionOpDescriptor_t convoOp = nullptr;
        miopenFusionOpDescriptor_t biasOp  = nullptr;
        miopenFusionOpDescriptor_t activOp = nullptr;
        miopenOperatorArgs_t fusionArgs;

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        double alpha = 1., beta = 0.;
        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
        miopenCreateOperatorArgs(&fusionArgs);
        miopenCreateOpConvForward(
            fusePlanDesc,
            &convoOp,
            filter,
            // \todo dlowell: Hardcoded right now. This assumes immediate mode. Needs GetAlgo.
            // miopenConvolutionFwdAlgoDirect,
            weightsDesc);
#if 0
        miopenConvFwdAlgorithm_t sup_algos[MIO_CONV_ALGO_COUNT];
        int retAlgCount = 0;
        // Query the supported algorithms
        miopenFusionPlanConvolutionGetAlgo(
            fusePlanDesc, MIO_CONV_ALGO_COUNT, &retAlgCount, sup_algos);
        // TODO: Replace this with WinoGrad to check for wino grad supported kernels
        miopenConvFwdAlgorithm_t req_algo = miopenConvolutionFwdAlgoDirect;
        if((std::begin(sup_algos) + retAlgCount) != std::find(std::begin(sup_algos),
                                                              std::begin(sup_algos) + retAlgCount,
                                                              miopenConvolutionFwdAlgoDirect))
        {
            // should not throw
            miopenFusionPlanConvolutionSetAlgo(fusePlanDesc, req_algo);
        }
#endif
        if(bias_mode)
            miopenCreateOpBiasForward(fusePlanDesc, &biasOp, biasDesc);

        miopenCreateOpActivationForward(fusePlanDesc, &activOp, activ_mode);

        // Compile
        miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, fusePlanDesc);
        if(miopenError != miopenStatusSuccess)
        {
            if(bias_mode)
                std::cerr << "Conv+Bias+Activation Inference plan not supported." << std::endl;
            else
                std::cerr << "Conv+Activation Inference plan not supported." << std::endl;
        }
        else
        {

            miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev.get());

            if(bias_mode)
                miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, b_dev.get());

            miopenSetOpArgsActivForward(
                fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);

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
struct cba_fusion_driver : test_driver
{
    tensor<T> input;
    tensor<T> weights;
    miopen::ConvolutionDescriptor filter;
    miopenActivationDescriptor_t activDesc{};
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
    double alpha = 0., beta = 0., gamma = 0.;

    std::unordered_map<std::string, miopenConvolutionMode_t> cmode_lookup = {
        {"CONV", miopenConvolution}}; //, {"TRANS", miopenTranspose}};

    std::unordered_map<std::string, miopenPaddingMode_t> pmode_lookup = {
        {"SAME", miopenPaddingSame},
        {"VALID", miopenPaddingValid},
        {"DEFAULT", miopenPaddingDefault}};

    cba_fusion_driver()
    {
        add(input, "input", get_input_tensor());
        add(weights, "weights", get_weights_tensor());
        add(filter, "filter", generate_data(get_filters()));
        add(alpha, "alpha", generate_data({/*1. , */ 0.5}));
        add(beta, "beta", generate_data({/*0. , */ 0.5}));
        add(gamma, "gamma", generate_data({/*1. ,*/ 0.5}));
        add(bias_mode, "bmode", generate_data({true, false}));
        // \todo dlowell: fusion can't handle trans right now.
        //       add(conv_mode, "cmode", generate_data({"conv"}/*, "trans"}*/));
        add(pad_mode, "pmode", generate_data({"default" /*, "same", "valid"*/}));
        add(tactiv, "test_activ", generate_data({false, true}));
        add(amode, "amode", generate_data({3, 6}));
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
                auto output = get_output_tensor(filter, input, weights);

                if(bias_mode)
                {
                    auto bias =
                        tensor<T>{1, output.desc.GetLengths()[1], 1, 1}.generate(rand_gen{});

                    // create activation descriptor here
                    if(tactiv)
                    {
                        verify(verify_forward_conv_bias_activ<T>{
                            input, weights, filter, bias_mode, bias, activDesc});
                    }
                    else
                    {
                        verify(verify_forward_conv_bias<T>{input, weights, filter, bias});
                    }
                }
                else
                {
                    if(tactiv)
                    {
                        auto bias = tensor<T>{1, 1, 1, 1};
                        verify(verify_forward_conv_bias_activ<T>{
                            input, weights, filter, bias_mode, bias, activDesc});
                    }
                }
            }
        }
        if(tactiv)
            miopenDestroyActivationDescriptor(activDesc);
    }
};

int main(int argc, const char* argv[]) { test_drive<cba_fusion_driver>(argc, argv); }
