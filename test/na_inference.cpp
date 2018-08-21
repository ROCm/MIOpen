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
// #include <miopen/batch_norm_activ.hpp>

template <class T>
struct verify_inference_batchnorm_activ
{
    tensor<T> input;
    miopenTensorDescriptor_t inputDesc{};
    miopenActivationDescriptor_t activDesc{};
    miopenTensorDescriptor_t biasScaleTensor{};
    tensor<T> bnscale{};
    tensor<T> bnbias{};
    tensor<T> estMean{};
    tensor<T> estVariance{};
    miopenBatchNormMode_t bnmode;
    double epsilon;

    verify_inference_batchnorm_activ(tensor<T>& pinput,
                                     miopenActivationDescriptor_t& pactivDesc,
                                     tensor<T>& pbnscale,
                                     tensor<T>& pbnbias,
                                     tensor<T>& pestMean,
                                     tensor<T>& pestVariance,
                                     miopenBatchNormMode_t pbnmode)
    {
        input           = pinput;
        inputDesc       = &pinput.desc;
        activDesc       = pactivDesc;
        biasScaleTensor = &pbnscale.desc;
        bnscale         = pbnscale;
        bnbias          = pbnbias;
        estMean         = pestMean;
        estVariance     = pestVariance;
        bnmode          = pbnmode;
        epsilon         = 1.0e-5;
    }

    tensor<T> cpu() const
    {

        auto&& handle = get_handle();
        auto bout     = input;
        std::fill(bout.begin(), bout.end(), 0.);
        auto aout = input;
        std::fill(aout.begin(), aout.end(), 0.);
        miopenFusionPlanDescriptor_t fusePlanDesc;
        miopenFusionOpDescriptor_t bNormOp = nullptr;
        miopenFusionOpDescriptor_t activOp = nullptr;

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
        miopenCreateOpBatchNormInference(fusePlanDesc, &bNormOp, bnmode, biasScaleTensor);
        miopenCreateOpActivationForward(fusePlanDesc, &activOp, activ_mode);

        miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, fusePlanDesc);
        if(miopenError != miopenStatusSuccess)
        {
            std::cerr << "BatchNorm+Activation Inference plan not supported." << std::endl;
        }
        else
        {
            if(bnmode == miopenBNPerActivation)
            {
                batchNormPerActivHostInference(
                    input, bout, bnscale, bnbias, epsilon, estMean, estVariance);
            }
            else
            {
                batchNormSpatialHostInference(
                    input, bout, bnscale, bnbias, epsilon, estMean, estVariance);
            }

            activationHostInfer(
                activ_mode, activ_gamma, activ_beta, activ_alpha, bout.data, aout.data);
        }

        miopenDestroyFusionPlanDescriptor(fusePlanDesc);
        return aout;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto baout    = input;
        std::fill(baout.begin(), baout.end(), 0.);
        auto in_dev          = handle.Write(input.data);
        auto out_dev         = handle.Write(baout.data);
        auto bnscale_dev     = handle.Write(bnscale.data);
        auto bnbias_dev      = handle.Write(bnbias.data);
        auto estMean_dev     = handle.Write(estMean.data);
        auto estVariance_dev = handle.Write(estVariance.data);

        miopenFusionPlanDescriptor_t fusePlanDesc;
        miopenFusionOpDescriptor_t bNormOp = nullptr;
        miopenFusionOpDescriptor_t activOp = nullptr;
        miopenOperatorArgs_t fusionArgs;

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        double alpha = 1., beta = 0.;
        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputDesc);
        miopenCreateOperatorArgs(&fusionArgs);
        miopenCreateOpBatchNormInference(fusePlanDesc, &bNormOp, bnmode, biasScaleTensor);
        miopenCreateOpActivationForward(fusePlanDesc, &activOp, activ_mode);

        miopenStatus_t miopenError = miopenCompileFusionPlan(&handle, fusePlanDesc);
        if(miopenError != miopenStatusSuccess)
        {
            std::cerr << "BatchNorm+Activation Inference plan not supported." << std::endl;
        }
        else
        {
            miopenSetOpArgsBatchNormInference(fusionArgs,
                                              bNormOp,
                                              &alpha,
                                              &beta,
                                              bnscale_dev.get(),
                                              bnbias_dev.get(),
                                              estMean_dev.get(),
                                              estVariance_dev.get(),
                                              epsilon);
            miopenSetOpArgsActivForward(
                fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
            miopenExecuteFusionPlan(&handle,
                                    fusePlanDesc,
                                    inputDesc,
                                    in_dev.get(),
                                    inputDesc,
                                    out_dev.get(),
                                    fusionArgs);
            baout.data = handle.Read<T>(out_dev, baout.data.size());
        }
        miopenDestroyFusionPlanDescriptor(fusePlanDesc);
        return baout;
    }

    void fail(float = 0) const { std::cerr << "BatchNorm+Activation Inference:" << std::endl; }
};

template <class T>
struct na_fusion_driver : test_driver
{
    tensor<T> input;
    tensor<T> scale;
    tensor<T> shift;
    tensor<T> estMean;
    tensor<T> estVariance;
    miopen::ActivationDescriptor activDesc{};
    miopenActivationMode_t activ_mode = miopenActivationRELU;
    std::string amode;
    miopenBatchNormMode_t bnmode{};
    int batchnormMode = 0;

    unsigned long max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;
    double alpha = 0., beta = 0., gamma = 0.;

    na_fusion_driver()
    {
        add(input, "input", get_input_tensor());
        add(alpha, "alpha", generate_data({/*1.,*/ 0.5}));
        add(beta, "beta", generate_data({/*0.,*/ 0.5}));
        add(gamma, "gamma", generate_data({/*1.,*/ 0.5}));
        add(amode, "amode", generate_data({"RELU", "LOGISTIC", "c"})); // get_activation_modes());
        add(batchnormMode, "batch-norm-mode", generate_data({0, 1}));
    }
    ~na_fusion_driver()
    {
        // todo: realease all the
    }

    void run()
    {
        if(amode == "a")
        {
            activ_mode = miopenActivationPASTHRU;
        }
        else if(amode == "b")
        {
            activ_mode = miopenActivationLOGISTIC;
        }

        // switch(amode)
        // {
        // case 0: activ_mode = miopenActivationPASTHRU; break;
        // case 1: activ_mode = miopenActivationLOGISTIC; break;
        // case 2: activ_mode = miopenActivationTANH; break;
        // case 3: activ_mode = miopenActivationRELU; break;
        // case 4: activ_mode = miopenActivationSOFTRELU; break;
        // case 5: activ_mode = miopenActivationABS; break;
        // case 6: activ_mode = miopenActivationPOWER; break;
        // case 7: activ_mode = miopenActivationCLIPPEDRELU; break;
        // case 8: activ_mode = miopenActivationLEAKYRELU; break;
        // case 9: activ_mode = miopenActivationELU;
        // }

        int input_c, input_h, input_w;
        std::tie(std::ignore, input_c, input_h, input_w) = miopen::tien<4>(input.desc.GetLengths());

        miopenSetActivationDescriptor(&activDesc, activ_mode, alpha, beta, gamma);

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
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, bnmode);
        std::tie(ssn, ssc, ssh, ssw) = miopen::tien<4>(derivedBnDesc.GetLengths());

        if(input.desc.GetType() == miopenFloat)
        {
            scale       = tensor<T>{ssn, ssc, ssh, ssw}.generate(rand_gen{});
            shift       = tensor<T>{ssn, ssc, ssh, ssw}.generate(rand_gen{});
            estMean     = tensor<T>{ssn, ssc, ssh, ssw}.generate(rand_gen{});
            estVariance = tensor<T>{ssn, ssc, ssh, ssw}.generate(rand_gen{});
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
                scale[i]       = (((rand() % 2) == 1) ? -1 : 1) * 1e-4 * T(rand() % 100);
                shift[i]       = (((rand() % 2) == 1) ? -1 : 1) * 1e-4 * T(rand() % 100);
                estMean[i]     = (((rand() % 2) == 1) ? -1 : 1) * 1e-4 * T(rand() % 100);
                estVariance[i] = std::fabs((((rand() % 2) == 1) ? -1 : 1) * 1e-2 * T(rand() % 100));
            }
            for(int i = 0; i < input.desc.GetElementSize(); i++)
            {
                input[i] = (((rand() % 2) == 1) ? -1 : 1) * T(rand() % 100);
            }
        }
        verify(verify_inference_batchnorm_activ<T>{
            input, &activDesc, scale, shift, estMean, estVariance, bnmode});
    }
};

int main(int argc, const char* argv[]) { test_drive<na_fusion_driver>(argc, argv); }
