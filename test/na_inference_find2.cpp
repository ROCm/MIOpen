/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#include "miopen/miopen.h"
#include "test.hpp"
#include "driver.hpp"
#include "fusionHost.hpp"
#include "random.hpp"
#include <miopen/stringutils.hpp>

#include <array>

#define MIO_BN_USE_MIX_PREC 1
#if MIO_BN_USE_MIX_PREC == 1
#define PREC_TYPE float
#else
#define PREC_TYPE T
#endif

using ptr_FusionPlanDesc = MIOPEN_MANAGE_PTR(miopenFusionPlanDescriptor_t, miopenDestroyFusionPlan);
using ptr_FusionPlanArgs = MIOPEN_MANAGE_PTR(miopenOperatorArgs_t, miopenDestroyOperatorArgs);
using ptr_ActivationDesc = MIOPEN_MANAGE_PTR(miopenActivationDescriptor_t,
                                             miopenDestroyActivationDescriptor);

using ManagedFindOptions = std::unique_ptr<std::remove_pointer_t<miopenFindOptions_t>,
                                           miopenStatus_t (*)(miopenFindOptions_t)>;

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

static inline auto MakeFindOtions() -> ManagedFindOptions
{
    miopenFindOptions_t find_options_raw;
    miopenCreateFindOptions(&find_options_raw);
    return {find_options_raw, &miopenDestroyFindOptions};
}

static inline void AddAndFuse(miopenProblem_t left,
                              std::function<void(miopenProblem_t*)>&& make_right_problem)
{
    miopenProblem_t right;
    make_right_problem(&right);
    EXPECT_EQUAL(miopenStatusSuccess, miopenFuseProblems(left, right));
    EXPECT_EQUAL(miopenStatusSuccess, miopenDestroyProblem(right));
};

template <class T, class U>
struct verify_inference_batchnorm_activ
{
    tensor<T> input;
    miopenTensorDescriptor_t inputDesc{};
    miopenActivationDescriptor_t activDesc{};
    miopenTensorDescriptor_t biasScaleTensor{};
    tensor<U> bnscale{};
    tensor<U> bnbias{};
    tensor<U> estMean{};
    tensor<U> estVariance{};
    miopenBatchNormMode_t bnmode;
    miopenProblem_t m_problem;
    double epsilon;

    verify_inference_batchnorm_activ(miopenProblem_t problem,
                                     tensor<T>& pinput,
                                     miopenActivationDescriptor_t pactivDesc,
                                     tensor<U>& pbnscale,
                                     const tensor<U>& pbnbias,
                                     const tensor<U>& pestMean,
                                     const tensor<U>& pestVariance,
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
        m_problem       = problem;
        epsilon         = 1.0e-5;
    }

    tensor<T> cpu() const
    {
        auto bout = input;
        std::fill(bout.begin(), bout.end(), 0.);
        auto aout = input;
        std::fill(aout.begin(), aout.end(), 0.);

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

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
        activationHostInfer(activ_mode, activ_gamma, activ_beta, activ_alpha, bout.data, aout.data);
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

        double activ_alpha, activ_beta, activ_gamma;
        miopenActivationMode_t activ_mode;
        miopenGetActivationDescriptor(
            activDesc, &activ_mode, &activ_alpha, &activ_beta, &activ_gamma);

        auto mut_eps = epsilon;

        const auto find_options = MakeFindOtions();
        // clang-format off
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormX, in_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorActivationY, out_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormScale, bnscale_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormBias, bnbias_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormEstimatedMean, estMean_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenTensorBatchnormEstimatedVariance, estVariance_dev.get());
        miopenSetFindOptionPreallocatedTensor(find_options.get(), miopenScalarBatchnormEpsilon, &mut_eps);
        // clang-format on

        const auto run_tensors = std::array{
            // clang-format off
            miopenTensorArgument_t{miopenTensorBatchnormX, nullptr, in_dev.get()},
            miopenTensorArgument_t{miopenTensorActivationY, nullptr, out_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormScale, nullptr, bnscale_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormBias, nullptr, bnbias_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormEstimatedMean, nullptr, estMean_dev.get()},
            miopenTensorArgument_t{miopenTensorBatchnormEstimatedVariance, nullptr, estVariance_dev.get()},
            miopenTensorArgument_t{miopenScalarBatchnormEpsilon, nullptr, (void*)&mut_eps},
            // clang-format on
        };

        std::vector<miopenSolution_t> solutions;
        solutions.resize(1);

        {
            std::size_t solutions_found;
            const auto find_ret = miopenFindSolutions(&handle,
                                                      m_problem,
                                                      find_options.get(),
                                                      solutions.data(),
                                                      &solutions_found,
                                                      solutions.size());

            EXPECT_EQUAL(find_ret, miopenStatusSuccess);
            solutions.resize(solutions_found);
        }

        for(const auto solution : solutions)
        {
            const auto run_ret = miopenRunSolution(
                &handle, solution, run_tensors.size(), run_tensors.data(), nullptr, 0);
            EXPECT_EQUAL(run_ret, miopenStatusSuccess);
        }

        /*
        double alpha = 1., beta = 0.;
        auto ptr_fusionargs = GetManageFusionPlanArgs();
        miopenSetOpArgsBatchNormInference(ptr_fusionargs.get(),
                                          bNormOp,
                                          &alpha,
                                          &beta,
                                          bnscale_dev.get(),
                                          bnbias_dev.get(),
                                          estMean_dev.get(),
                                          estVariance_dev.get(),
                                          epsilon);
        miopenSetOpArgsActivForward(
            ptr_fusionargs.get(), activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
        miopenExecuteFusionPlan(&handle,
                                fusionplan,
                                inputDesc,
                                in_dev.get(),
                                inputDesc,
                                out_dev.get(),
                                ptr_fusionargs.get());
        */

        baout.data = handle.Read<T>(out_dev, baout.data.size());
        return baout;
    }

    void fail(float = 0) const { std::cerr << "BatchNorm+Activation Inference:" << std::endl; }
};

static std::string transform_mode(std::string s)
{
    return miopen::RemovePrefix(miopen::ToUpper(s), "MIOPENACTIVATION");
}

template <class T>
struct na_fusion_driver : test_driver
{
    tensor<T> input;
    tensor<PREC_TYPE> scale;
    tensor<PREC_TYPE> shift;
    tensor<PREC_TYPE> estMean;
    tensor<PREC_TYPE> estVariance;
    ptr_ActivationDesc ptr_activdesc = nullptr;

    miopenActivationMode_t activ_mode = miopenActivationRELU;
    std::string amode;
    miopenBatchNormMode_t bnmode{};
    int batchnormMode = 1;

    uint64_t max_value = miopen_type<T>{} == miopenHalf ? 3 : 17;
    double alpha = 0., beta = 0., gamma = 0.;

    na_fusion_driver()
    {
        add(input, "input", get_input_tensor());
        add(alpha, "alpha", generate_data({/*1.,*/ 0.5}));
        add(beta, "beta", generate_data({/*0.,*/ 0.5}));
        add(gamma, "gamma", generate_data({/*1.,*/ 0.5}));
        add(amode,
            "amode",
            generate_data(
                {"MIOPENACTIVATIONRELU" /*, "MIOPENACTIVATIONLOGISTIC", "MIOPENACTIVATIONABS"*/}));
        add(batchnormMode, "batch-norm-mode", generate_data({/*0, */ 1}));
    }

    void run()
    {
        amode = transform_mode(amode);

        // NOLINTBEGIN(*-braces-around-statements)
        if(amode == "PASSTHRU")
            activ_mode = miopenActivationPASTHRU;
        else if(amode == "LOGISTIC")
            activ_mode = miopenActivationLOGISTIC;
        else if(amode == "TANH")
            activ_mode = miopenActivationTANH;
        else if(amode == "RELU")
            activ_mode = miopenActivationRELU;
        else if(amode == "SOFTRELU")
            activ_mode = miopenActivationSOFTRELU;
        else if(amode == "ABS")
            activ_mode = miopenActivationABS;
        else if(amode == "POWER")
            activ_mode = miopenActivationPOWER;
        else if(amode == "CLIPPEDRELU")
            activ_mode = miopenActivationCLIPPEDRELU;
        else if(amode == "LEAKYRELU")
            activ_mode = miopenActivationLEAKYRELU;
        else if(amode == "ELU")
            activ_mode = miopenActivationELU;
        // NOLINTEND(*-braces-around-statements)

        int input_c, input_h, input_w;
        std::tie(std::ignore, input_c, input_h, input_w) = miopen::tien<4>(input.desc.GetLengths());
        ptr_activdesc                                    = GetManagedActivDesc();
        miopenSetActivationDescriptor(ptr_activdesc.get(), activ_mode, alpha, beta, gamma);

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

        scale = tensor<PREC_TYPE>{
            ssn, ssc, ssh, ssw}; //.generate(                tensor_elem_gen_integer{max_value});;
        shift = tensor<PREC_TYPE>{
            ssn, ssc, ssh, ssw}; //.generate(               tensor_elem_gen_integer{max_value});;
        estMean = tensor<PREC_TYPE>{
            ssn, ssc, ssh, ssw}; //.generate(                tensor_elem_gen_integer{max_value});;
        estVariance = tensor<PREC_TYPE>{
            ssn, ssc, ssh, ssw}; //.generate(                tensor_elem_gen_integer{max_value});;

        const double Data_scale = 0.01;
        for(std::size_t i = 0; i < scale.desc.GetElementSize(); i++)
        {
            scale[i]       = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
            shift[i]       = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
            estMean[i]     = prng::gen_descreet_uniform_sign<PREC_TYPE>(Data_scale, 100);
            estVariance[i] = static_cast<PREC_TYPE>(Data_scale * prng::gen_off_range(1, 100));
        }
        for(std::size_t i = 0; i < input.desc.GetElementSize(); i++)
        {
            input[i] = prng::gen_descreet_uniform_sign<T>(Data_scale, 100);
        }

        using ManagedProblem = std::unique_ptr<std::remove_pointer_t<miopenProblem_t>,
                                               miopenStatus_t (*)(miopenProblem_t)>;

        const auto problem = [&]() {
            miopenProblem_t problem;
            miopenCreateBatchnormProblem(&problem, bnmode, true, miopenProblemDirectionInference);

            // clang-format off
            miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormX, &input.desc);
            miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormScale, &scale.desc);
            miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormBias, &shift.desc);
            miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormEstimatedMean, &derivedBnDesc);
            miopenSetProblemTensorDescriptor(problem, miopenTensorBatchnormEstimatedVariance, &derivedBnDesc);
            // clang-format on

            AddAndFuse(problem, [&](auto activation) {
                miopenCreateActivationProblem(
                    activation, ptr_activdesc.get(), miopenProblemDirectionForward);
                miopenSetProblemTensorDescriptor(*activation, miopenTensorActivationX, &input.desc);
                miopenSetProblemTensorDescriptor(*activation, miopenTensorActivationY, &input.desc);
            });

            return ManagedProblem{problem, &miopenDestroyProblem};
        }();

        verify(verify_inference_batchnorm_activ<T, PREC_TYPE>{
            problem.get(), input, ptr_activdesc.get(), scale, shift, estMean, estVariance, bnmode});
    }
};

int main(int argc, const char* argv[]) { test_drive<na_fusion_driver>(argc, argv); }
