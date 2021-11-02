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
#include "random.hpp"
#include <miopen/stringutils.hpp>

using ptr_FusionPlanDesc = MIOPEN_MANAGE_PTR(miopenFusionPlanDescriptor_t, miopenDestroyFusionPlan);
using ptr_FusionPlanArgs = MIOPEN_MANAGE_PTR(miopenOperatorArgs_t, miopenDestroyOperatorArgs);
using ptr_ActivationDesc = MIOPEN_MANAGE_PTR(miopenActivationDescriptor_t,
                                             miopenDestroyActivationDescriptor);
ptr_FusionPlanDesc GetManagedFusionPlanDesc(miopenTensorDescriptor_t inputDesc)
{
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
    return ptr_FusionPlanDesc{fusePlanDesc};
}

ptr_FusionPlanArgs GetManageFusionPlanArgs()
{
    miopenOperatorArgs_t fusionArgs;
    miopenCreateOperatorArgs(&fusionArgs);
    return ptr_FusionPlanArgs{fusionArgs};
}

ptr_ActivationDesc GetManagedActivDesc()
{
    miopenActivationDescriptor_t activdesc;
    miopenCreateActivationDescriptor(&activdesc);
    return ptr_ActivationDesc{activdesc};
}

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
    miopenFusionPlanDescriptor_t fusionplan;

    verify_forward_conv_bias(miopenFusionPlanDescriptor_t pfusionplan,
                             tensor<T>& pinput,
                             tensor<T>& pweights,
                             miopen::ConvolutionDescriptor& pfilter,
                             tensor<T>& pbias)
    {
        input       = pinput;
        inputDesc   = &pinput.desc;
        weights     = pweights;
        weightsDesc = &pweights.desc;
        bias        = pbias;
        biasDesc    = &pbias.desc;
        filter      = &pfilter;
        fusionplan  = pfusionplan;
    }

    tensor<T> cpu() const
    {
        auto rout = get_output_tensor(miopen::deref(filter), input, weights);
        convHostForward(input, rout, weights, 1, bias, filter);
        return rout;
    }

    tensor<T> gpu() const
    {
        double alpha = 1., beta = 0.;
        miopenFusionOpDescriptor_t convoOp{};
        miopenFusionOpDescriptor_t biasOp{};

        auto&& handle              = get_handle();
        auto rout                  = get_output_tensor(miopen::deref(filter), input, weights);
        auto in_dev                = handle.Write(input.data);
        auto wei_dev               = handle.Write(weights.data);
        auto b_dev                 = handle.Write(bias.data);
        auto out_dev               = handle.Write(rout.data);
        auto ptr_fusionargs        = GetManageFusionPlanArgs();
        miopenStatus_t miopenError = miopenFusionPlanGetOp(fusionplan, 0, &convoOp);
        EXPECT(miopenError == miopenStatusSuccess);
        miopenError = miopenFusionPlanGetOp(fusionplan, 1, &biasOp);
        EXPECT(miopenError == miopenStatusSuccess);
        miopenSetOpArgsConvForward(ptr_fusionargs.get(), convoOp, &alpha, &beta, wei_dev.get());
        miopenSetOpArgsBiasForward(ptr_fusionargs.get(), biasOp, &alpha, &beta, b_dev.get());
        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                inputDesc,
                                in_dev.get(),
                                &rout.desc,
                                out_dev.get(),
                                ptr_fusionargs.get());
        rout.data = handle.Read<T>(out_dev, rout.data.size());
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
    miopenFusionPlanDescriptor_t fusionplan;
    int bias_mode = 0;

    verify_forward_conv_bias_activ(miopenFusionPlanDescriptor_t pfusionplan,
                                   tensor<T>& pinput,
                                   tensor<T>& pweights,
                                   miopen::ConvolutionDescriptor& pfilter,
                                   int pbias_mode,
                                   tensor<T>& pbias,
                                   miopenActivationDescriptor_t pactivDesc)
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
        fusionplan  = pfusionplan;
    }

    tensor<T> cpu() const
    {
        auto rout = get_output_tensor(miopen::deref(filter), input, weights);
        auto aout = rout;
        std::fill(aout.begin(), aout.end(), 0.);
        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
        convHostForward(input, rout, weights, bias_mode, bias, filter);
        activationHostInfer(activ_mode, activ_gamma, activ_beta, activ_alpha, rout.data, aout.data);
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

        miopenFusionOpDescriptor_t convoOp{};
        miopenFusionOpDescriptor_t biasOp{};
        miopenFusionOpDescriptor_t activOp{};

        double alpha = 1., beta = 0.;
        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        auto ptr_fusionargs = GetManageFusionPlanArgs();

        auto opcounter             = 0;
        miopenStatus_t miopenError = miopenFusionPlanGetOp(fusionplan, opcounter++, &convoOp);
        EXPECT(miopenError == miopenStatusSuccess);
        miopenSetOpArgsConvForward(ptr_fusionargs.get(), convoOp, &alpha, &beta, wei_dev.get());

        if(bias_mode != 0)
        {
            miopenError = miopenFusionPlanGetOp(fusionplan, opcounter++, &biasOp);
            EXPECT(miopenError == miopenStatusSuccess);
            miopenSetOpArgsBiasForward(ptr_fusionargs.get(), biasOp, &alpha, &beta, b_dev.get());
        }

        miopenError = miopenFusionPlanGetOp(fusionplan, opcounter, &activOp);
        EXPECT(miopenError == miopenStatusSuccess);
        miopenSetOpArgsActivForward(
            ptr_fusionargs.get(), activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);

        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                inputDesc,
                                in_dev.get(),
                                &rout.desc,
                                out_dev.get(),
                                ptr_fusionargs.get());
        rout.data = handle.Read<T>(out_dev, rout.data.size());
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
    tensor<T> bias;
    miopen::ConvolutionDescriptor filter;
    std::vector<int> pads_strides_dilations;
    ptr_ActivationDesc ptr_activdesc  = nullptr;
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
        add(input, "input", get_input_tensor(tensor_elem_gen_integer{max_value}));
        add(weights, "weights", get_weights_tensor(tensor_elem_gen_integer{max_value}));
        add(pads_strides_dilations,
            "pads_strides_dilations",
            generate_data(get_pads_strides_dilations()));
        add(alpha, "alpha", generate_data({/*1. , */ 0.5}));
        add(beta, "beta", generate_data({/*0. , */ 0.5}));
        add(gamma, "gamma", generate_data({/*1. ,*/ 0.5}));
        add(bias_mode, "bmode", generate_data({true /*, false*/}));
        // \todo dlowell: fusion can't handle transpose right now.
        //       add(conv_mode, "cmode", generate_data({"conv"}/*, "trans"}*/));
        add(pad_mode, "pmode", generate_data({"default" /*, "same", "valid"*/}));
        add(tactiv, "test_activ", generate_data({/*false, */ true}));
        add(amode, "amode", generate_data({3 /*, 6*/}));
    }

    std::vector<std::vector<int>> get_pads_strides_dilations()
    {
        return {
            {0, 0, 1, 1, 1, 1},
            //       {1, 1, 1, 1, 1, 1},
            //       {0, 0, 2, 2, 1, 1},
            {1, 1, 2, 2, 1, 1},
            //       {2, 2, 1, 1, 1, 1},
            //       {2, 2, 2, 2, 1, 1},
            //       {3, 3, 2, 2, 1, 1}
        };
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

        filter.mode         = cmode_lookup[miopen::ToUpper(conv_mode)];
        filter.paddingMode  = pmode_lookup[miopen::ToUpper(pad_mode)];
        filter.pads[0]      = pads_strides_dilations[0];
        filter.pads[1]      = pads_strides_dilations[1];
        filter.strides[0]   = pads_strides_dilations[2];
        filter.strides[1]   = pads_strides_dilations[3];
        filter.dilations[0] = pads_strides_dilations[4];
        filter.dilations[1] = pads_strides_dilations[5];

        auto stride_h     = filter.strides[0];
        auto stride_w     = filter.strides[1];
        auto fpad_h       = filter.pads[0];
        auto fpad_w       = filter.pads[1];
        auto fpaddingMode = filter.paddingMode;

        auto&& handle = get_handle();

        miopenFusionOpDescriptor_t convoOp = nullptr;
        miopenFusionOpDescriptor_t biasOp  = nullptr;
        miopenFusionOpDescriptor_t activOp = nullptr;

        if(((filter.mode == miopenTranspose) && (input_c == wei_k)) ||
           ((filter.mode == miopenConvolution) && (input_c == wei_c)))
        {
            if(fpaddingMode == miopenPaddingSame)
            {

                if(stride_h == 0 || stride_w == 0)
                    return;
                auto _pad_h = (input_h % stride_h == 0)
                                  ? (std::max(static_cast<int>(wei_h - stride_h), 0))
                                  : (std::max(static_cast<int>(wei_h - (input_h % stride_h)), 0));
                auto _pad_w = (input_w % stride_w == 0)
                                  ? (std::max(static_cast<int>(wei_w - stride_w), 0))
                                  : (std::max(static_cast<int>(wei_w - (input_w % stride_w)), 0));

                filter.pads[0] = _pad_h / 2;
                filter.pads[1] = _pad_w / 2;

                int out_h = std::ceil(static_cast<double>(input_h) / stride_h);
                int out_w = std::ceil(static_cast<double>(input_w) / stride_w);

                if(out_h <= 0 || out_w <= 0)
                    return;
            }
            else if(fpaddingMode == miopenPaddingValid)
            {
                if(stride_h == 0 || stride_w == 0)
                    return;
                filter.pads[0] = 0;
                filter.pads[1] = 0;

                int out_h = std::ceil(static_cast<double>(input_h - wei_h + 1) / stride_h);
                int out_w = std::ceil(static_cast<double>(input_w - wei_w + 1) / stride_w);

                if(out_h <= 0 || out_w <= 0)
                    return;
            }

            auto ptr_fusionplan = GetManagedFusionPlanDesc(&input.desc);
            miopenCreateOpConvForward(ptr_fusionplan.get(), &convoOp, &filter, &weights.desc);

            auto output = get_output_tensor(filter, input, weights);
            if(bias_mode)
            {
                if(input.desc.GetType() == miopenFloat)
                {
                    bias = tensor<T>{1, output.desc.GetLengths()[1], 1, 1}.generate(
                        tensor_elem_gen_integer{17});
                }
                else
                {
                    bias = tensor<T>{1, output.desc.GetLengths()[1], 1, 1};
                    srand(0);
                    for(std::size_t i = 0; i < bias.desc.GetElementSize(); i++)
                    {
                        bias[i] = (((GET_RAND() % 2) == 1) ? -1 : 1) * (0.1 * T(GET_RAND() % 100));
                    }
                }

                miopenCreateOpBiasForward(ptr_fusionplan.get(), &biasOp, &bias.desc);
            }
            else
            {
                bias = tensor<T>{1, 1, 1, 1};
            }

            if(tactiv)
            {
                ptr_activdesc = GetManagedActivDesc();
                miopenSetActivationDescriptor(ptr_activdesc.get(), activ_mode, alpha, beta, gamma);
                miopenCreateOpActivationForward(ptr_fusionplan.get(), &activOp, activ_mode);
            }
            miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, ptr_fusionplan.get());
            if(miopenError != miopenStatusSuccess)
            {
                std::cerr << "CBA Inference plan not supported." << std::endl;
            }
            else if(input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(1) &&
                    wei_h > 2 * fpad_h && wei_w > 2 * fpad_w && input_h >= (2 * fpad_h + wei_h) &&
                    input_w >= (2 * fpad_w + wei_w))
            {

                if(bias_mode)
                {
                    // create activation descriptor here
                    if(tactiv)
                    {
                        verify(verify_forward_conv_bias_activ<T>{ptr_fusionplan.get(),
                                                                 input,
                                                                 weights,
                                                                 filter,
                                                                 bias_mode,
                                                                 bias,
                                                                 ptr_activdesc.get()});
                    }
                    else
                    {
                        verify(verify_forward_conv_bias<T>{
                            ptr_fusionplan.get(), input, weights, filter, bias});
                    }
                }
                else
                {
                    if(tactiv)
                    {
                        verify(verify_forward_conv_bias_activ<T>{ptr_fusionplan.get(),
                                                                 input,
                                                                 weights,
                                                                 filter,
                                                                 bias_mode,
                                                                 bias,
                                                                 ptr_activdesc.get()});
                    }
                }
            }
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<cba_fusion_driver>(argc, argv); }
