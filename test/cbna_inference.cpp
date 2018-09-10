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

#define MIO_CONV_ALGO_COUNT 4

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
struct verify_forward_conv_bias_batchnorm_activ
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
    miopenTensorDescriptor_t biasScaleTensor{};
    tensor<T> bnscale{};
    tensor<T> bnbias{};
    tensor<T> estMean{};
    tensor<T> estVariance{};
    miopenFusionPlanDescriptor_t fusionplan;

    miopenBatchNormMode_t bnmode;
    bool bias_mode = false;
    bool doactive  = false;
    double epsilon;

    // using conv_base<T>::search; //DLOWELL not needed right now
    verify_forward_conv_bias_batchnorm_activ(miopenFusionPlanDescriptor_t pfusionplan,
                                             tensor<T>& pinput,
                                             tensor<T>& pweights,
                                             miopen::ConvolutionDescriptor& pfilter,
                                             bool pbias_mode,
                                             tensor<T>& pbias,
                                             miopenActivationDescriptor_t pactivDesc,
                                             bool pdoactiv,
                                             tensor<T>& pbnscale,
                                             tensor<T>& pbnbias,
                                             tensor<T>& pestMean,
                                             tensor<T>& pestVariance,
                                             miopenBatchNormMode_t pbnmode)
    {
        input           = pinput;
        inputDesc       = &pinput.desc;
        weights         = pweights;
        weightsDesc     = &pweights.desc;
        bias            = pbias;
        biasDesc        = &pbias.desc;
        filter          = &pfilter;
        activDesc       = pactivDesc;
        doactive        = pdoactiv;
        bias_mode       = pbias_mode;
        biasScaleTensor = &pbnscale.desc;
        bnscale         = pbnscale;
        bnbias          = pbnbias;
        estMean         = pestMean;
        estVariance     = pestVariance;
        bnmode          = pbnmode;
        fusionplan      = pfusionplan;
        epsilon         = 1.0e-5;
    }

    tensor<T> cpu() const
    {

        auto rout = get_output_tensor(miopen::deref(filter), input, weights);
        auto aout = rout;
        std::fill(aout.begin(), aout.end(), 0.);
        auto bout = rout;
        std::fill(bout.begin(), bout.end(), 0.);

        // If we are using convolutions as the base, we can calculate the
        convHostForward(input, rout, weights, bias_mode, bias, filter);
        if(bnmode == miopenBNPerActivation)
        {
            batchNormPerActivHostInference(
                rout, bout, bnscale, bnbias, epsilon, estMean, estVariance);
        }
        else
        {
            batchNormSpatialHostInference(
                rout, bout, bnscale, bnbias, epsilon, estMean, estVariance);
        }
        if(doactive)
        {
            double activ_alpha, activ_beta, activ_gamma;
            miopenActivationMode_t activ_mode;
            miopenGetActivationDescriptor(
                activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
            activationHostInfer(
                activ_mode, activ_gamma, activ_beta, activ_alpha, bout.data, aout.data);
        }
        else
        {
            return bout;
        }
        return aout;
    }

    tensor<T> gpu() const
    {
        auto&& handle        = get_handle();
        auto rout            = get_output_tensor(miopen::deref(filter), input, weights);
        auto in_dev          = handle.Write(input.data);
        auto wei_dev         = handle.Write(weights.data);
        auto b_dev           = handle.Write(bias.data);
        auto out_dev         = handle.Write(rout.data);
        auto bnscale_dev     = handle.Write(bnscale.data);
        auto bnbias_dev      = handle.Write(bnbias.data);
        auto estMean_dev     = handle.Write(estMean.data);
        auto estVariance_dev = handle.Write(estVariance.data);

        miopenFusionOpDescriptor_t convoOp = nullptr;
        miopenFusionOpDescriptor_t biasOp  = nullptr;
        miopenFusionOpDescriptor_t bNormOp = nullptr;
        miopenFusionOpDescriptor_t activOp = nullptr;
        auto ptr_fusionargs                = GetManageFusionPlanArgs();

        double alpha = 1., beta = 0.;
        auto opcounter             = 0;
        miopenStatus_t miopenError = miopenFusionPlanGetOp(fusionplan, opcounter++, &convoOp);
        EXPECT(miopenError == miopenStatusSuccess);
        miopenSetOpArgsConvForward(ptr_fusionargs.get(), convoOp, &alpha, &beta, wei_dev.get());

        if(bias_mode)
        {
            miopenError = miopenFusionPlanGetOp(fusionplan, opcounter++, &biasOp);
            EXPECT(miopenError == miopenStatusSuccess);
            miopenSetOpArgsBiasForward(ptr_fusionargs.get(), biasOp, &alpha, &beta, b_dev.get());
        }

        miopenError = miopenFusionPlanGetOp(fusionplan, opcounter++, &bNormOp);
        EXPECT(miopenError == miopenStatusSuccess);
        miopenSetOpArgsBatchNormInference(ptr_fusionargs.get(),
                                          bNormOp,
                                          &alpha,
                                          &beta,
                                          bnscale_dev.get(),
                                          bnbias_dev.get(),
                                          estMean_dev.get(),
                                          estVariance_dev.get(),
                                          epsilon);
        if(doactive)
        {
            miopenError = miopenFusionPlanGetOp(fusionplan, opcounter, &activOp);
            EXPECT(miopenError == miopenStatusSuccess);
            double activ_alpha, activ_beta, activ_gamma;
            miopenActivationMode_t activ_mode;
            miopenGetActivationDescriptor(
                activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);
            miopenSetOpArgsActivForward(
                ptr_fusionargs.get(), activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
        }
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
        if(bias_mode)
            if(doactive)
            {
                std::cerr << "Conv+Bias+BatchNorm+Activation Inference:" << std::endl;
            }
            else
            {
                std::cerr << "Conv+Bias+BatchNorm Inference:" << std::endl;
            }
        else
        {
            if(doactive)
            {
                std::cerr << "Conv+BatchNorm+Activation Inference:" << std::endl;
            }
            else
            {
                std::cerr << "Conv+BatchNorm Inference:" << std::endl;
            }
        }
    }
};

template <class T>
struct cbna_fusion_driver : test_driver
{
    tensor<T> input;
    tensor<T> output;
    tensor<T> weights;
    tensor<T> scale;
    tensor<T> shift;
    tensor<T> estMean;
    tensor<T> estVariance;
    tensor<T> bias;
    miopen::ConvolutionDescriptor filter;
    ptr_ActivationDesc ptr_activdesc  = nullptr;
    miopenActivationMode_t activ_mode = miopenActivationRELU;
    int amode                         = 0;
    bool tactiv{};
    bool bias_mode = true;
    miopenBatchNormMode_t bnmode{};
    int batchnormMode = 0;
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

    cbna_fusion_driver()
    {
        add(input, "input", get_input_tensor(tensor_elem_gen_integer{max_value}));
        add(weights, "weights", get_weights_tensor(tensor_elem_gen_integer{max_value}));
        add(filter, "filter", generate_data(get_filters()));
        add(alpha, "alpha", generate_data({/*1. , */ 0.5}));
        add(beta, "beta", generate_data({/*0. , */ 0.5}));
        add(gamma, "gamma", generate_data({/*1. ,*/ 0.5}));
        add(bias_mode, "bmode", generate_data({true, false}));
        // \todo dlowell: fusion can't handle trans right now.
        //       add(conv_mode, "cmode", generate_data({"conv"}/*, "trans"}*/));
        add(pad_mode, "pmode", generate_data({"default" /*, "same", "valid"*/}));
        add(tactiv, "test_activ", generate_data({false, true}));
        add(amode, "amode", generate_data({3}));
        add(batchnormMode, "batch-norm-mode", generate_data({0, 1}));
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

        miopenFusionOpDescriptor_t convoOp = nullptr;
        miopenFusionOpDescriptor_t biasOp  = nullptr;
        miopenFusionOpDescriptor_t bNormOp = nullptr;
        miopenFusionOpDescriptor_t activOp = nullptr;

        auto&& handle       = get_handle();
        auto ptr_fusionplan = GetManagedFusionPlanDesc(&input.desc);

        filter.mode        = cmode_lookup[miopen::ToUpper(conv_mode)];
        filter.paddingMode = pmode_lookup[miopen::ToUpper(pad_mode)];

        auto u            = filter.u;
        auto v            = filter.v;
        auto fpad_h       = filter.pad_h;
        auto fpad_w       = filter.pad_w;
        auto fpaddingMode = filter.paddingMode;

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

            if(batchnormMode == 1)
            {
                bnmode = miopenBNSpatial;
            }
            else if(batchnormMode == 0)
            {
                bnmode = miopenBNPerActivation;
            }

            std::size_t ssn, ssc, ssh, ssw;
            auto derivedBnDesc = miopen::TensorDescriptor{};
            output             = get_output_tensor(filter, input, weights);
            miopen::DeriveBNTensorDescriptor(derivedBnDesc, output.desc, bnmode);
            std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());

            if(input.desc.GetType() == miopenFloat)
            {
                scale       = tensor<T>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{17});
                shift       = tensor<T>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{17});
                estMean     = tensor<T>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{17});
                estVariance = tensor<T>{ssn, ssc, ssh, ssw}.generate(tensor_elem_gen_integer{17});
            }
            else
            {
                scale       = tensor<T>{ssn, ssc, ssh, ssw};
                shift       = tensor<T>{ssn, ssc, ssh, ssw};
                estMean     = tensor<T>{ssn, ssc, ssh, ssw};
                estVariance = tensor<T>{ssn, ssc, ssh, ssw};

                srand(0);
                for(int i = 0; i < scale.desc.GetElementSize(); i++)
                {
                    scale[i]   = (((rand() % 2) == 1) ? -1 : 1) * 1e-4 * T(rand() % 100);
                    shift[i]   = (((rand() % 2) == 1) ? -1 : 1) * 1e-4 * T(rand() % 100);
                    estMean[i] = (((rand() % 2) == 1) ? -1 : 1) * 1e-4 * T(rand() % 100);
                    estVariance[i] =
                        std::fabs((((rand() % 2) == 1) ? -1 : 1) * 1e-1 * T(rand() % 100));
                }
                for(int i = 0; i < input.desc.GetElementSize(); i++)
                {
                    input[i] = (((rand() % 2) == 1) ? -1 : 1) * (0.1 * T(rand() % 100));
                }
            }
        }

        miopenCreateOpConvForward(ptr_fusionplan.get(), &convoOp, &filter, &weights.desc);

        miopenConvFwdAlgorithm_t sup_algos[MIO_CONV_ALGO_COUNT];
        int retAlgCount = 0;
        // Query the supported algorithms
        miopenFusionPlanConvolutionGetAlgo(
            ptr_fusionplan.get(), MIO_CONV_ALGO_COUNT, &retAlgCount, sup_algos);
        // TODO: Replace this with WinoGrad to check for wino grad supported kernels
        miopenConvFwdAlgorithm_t req_algo = miopenConvolutionFwdAlgoDirect;
        if((std::begin(sup_algos) + retAlgCount) != std::find(std::begin(sup_algos),
                                                              std::begin(sup_algos) + retAlgCount,
                                                              miopenConvolutionFwdAlgoDirect))
        {
            // should not throw
            miopenFusionPlanConvolutionSetAlgo(ptr_fusionplan.get(), req_algo);
        }

        if(bias_mode)
        {
            bias = tensor<T>{1, output.desc.GetLengths()[1], 1, 1}.generate(
                tensor_elem_gen_integer{17});
            miopenCreateOpBiasForward(ptr_fusionplan.get(), &biasOp, &bias.desc);
        }
        else
        {
            bias = tensor<T>{1, 1, 1, 1};
        }

        miopenCreateOpBatchNormInference(ptr_fusionplan.get(), &bNormOp, bnmode, &scale.desc);

        ptr_activdesc = GetManagedActivDesc();
        if(tactiv)
        {
            miopenSetActivationDescriptor(ptr_activdesc.get(), activ_mode, alpha, beta, gamma);
            miopenCreateOpActivationForward(ptr_fusionplan.get(), &activOp, activ_mode);
        }

        // Compile
        miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, ptr_fusionplan.get());
        if(miopenError != miopenStatusSuccess)
        {
            if(bias_mode)
                if(tactiv)
                {
                    std::cerr << "Conv+Bias+BatchNorm+Activation Inference plan not supported."
                              << std::endl;
                }
                else
                {
                    std::cerr << "Conv+Bias+BatchNorm Inference plan not supported." << std::endl;
                }
            else
            {
                if(tactiv)
                {
                    std::cerr << "Conv+BatchNorm+Activation Inference plan not supported."
                              << std::endl;
                }
                else
                {
                    std::cerr << "Conv+BatchNorm Inference plan not supported." << std::endl;
                }
            }
        }
        else if(input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(1) &&
                wei_h > 2 * fpad_h && wei_w > 2 * fpad_w && input_h >= (2 * fpad_h + wei_h) &&
                input_w >= (2 * fpad_w + wei_w))
        {
            output = get_output_tensor(filter, input, weights);
            if(bias_mode)
            {
                // create activation descriptor here
                verify(verify_forward_conv_bias_batchnorm_activ<T>{ptr_fusionplan.get(),
                                                                   input,
                                                                   weights,
                                                                   filter,
                                                                   bias_mode,
                                                                   bias,
                                                                   ptr_activdesc.get(),
                                                                   tactiv,
                                                                   scale,
                                                                   shift,
                                                                   estMean,
                                                                   estVariance,
                                                                   bnmode});
            }
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<cbna_fusion_driver>(argc, argv); }
