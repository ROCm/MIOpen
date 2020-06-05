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

#include <miopen/batch_norm.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/visit_float.hpp>

#include <cassert>
#include <chrono>
#include <iostream>

#define MIOPEN_BN_SYNCH 0

namespace miopen {

void DeriveBNTensorDescriptor(TensorDescriptor& derivedBnDesc,
                              const TensorDescriptor& xDesc,
                              miopenBatchNormMode_t bn_mode)
{

    auto lengths = xDesc.GetLengths();
    std::vector<int> newlens(lengths.size());
    newlens[1] = lengths[1];
    if(bn_mode == miopenBNSpatial)
    {
        newlens[0] = newlens[2] = newlens[3] = 1;
        if(lengths.size() == 5)
            newlens[4] = 1;
    }
    else
    {
        newlens[0] = 1;
        newlens[2] = lengths[2];
        newlens[3] = lengths[3];

        if(lengths.size() == 5)
            newlens[4] = lengths[4];
    }
    derivedBnDesc =
        TensorDescriptor(/* xDesc.GetType() */ miopenFloat, newlens.data(), xDesc.GetSize());
}

TensorDescriptor BuildReshaped4DTensorDescriptor(const miopen::TensorDescriptor& tDesc)
{
    auto dataType = tDesc.GetType();
    std::vector<size_t> dims(tDesc.GetLengths());

    // NxCxDxHxW -> NxCx(D*H)xW
    dims[2] *= dims[3];
    dims[3] = dims[4];
    dims.pop_back();

    return {dataType, dims};
}

void profileSequence(const Handle& handle, unsigned char select, float* ctime)
{

    float ktime = 0.;
    assert((select < 3) && "profileSequence case incorrect");
    switch(select)
    {

    case 0:
        if(handle.IsProfilingEnabled())
        {
            *ctime = 0.;
            handle.ResetKernelTime();
            ktime  = handle.GetKernelTime();
            *ctime = ktime;

#if(MIO_BN_CPP_PROF == 1)
            printf("ktime0: %lf\n", ktime);
            printf("ctime: %f\n", *ctime);
#endif
        }
#if(MIOPEN_BN_SYNCH == 1)
        else
        {
            handle.Finish();
        }
#endif
        break;
    case 1:
        if(handle.IsProfilingEnabled())
        {
            ktime = handle.GetKernelTime();
            *ctime += ktime;

#if(MIO_BN_CPP_PROF == 1)
            printf("ktime1: %lf\n", ktime);
            printf("ctime: %f\n", *ctime);
#endif
        }
#if(MIOPEN_BN_SYNCH == 1)
        else
        {
            handle.Finish();
        }
#endif
        break;

    case 2:
        if(handle.IsProfilingEnabled())
        {

#if(MIO_BN_CPP_PROF == 1)
            ktime = handle.GetKernelTime();
            handle.AccumKernelTime(*ctime);
            printf("ktime2: %lf\n", ktime);
            printf("ctime: %f\n", *ctime + ktime);
#else
            handle.GetKernelTime();
            handle.AccumKernelTime(*ctime);
#endif
        }
        break;
    default: assert(false);
    }
}

void bnFwdTrainSelectMulti(const Handle& handle,
                           miopenDataType_t dtype,
                           const std::string& program_name,
                           const std::string& algo_name,
                           const std::string& kernel_name,
                           const std::string& network_config,
                           const std::string& parms,
                           const std::vector<size_t>& vld,
                           const std::vector<size_t>& vgd,
                           ConstData_t x,
                           Data_t y,
                           ConstData_t bnScale,
                           ConstData_t bnBias,
                           bool resultsave,
                           bool resultrunning,
                           double expAvgFactor,
                           Data_t resultRunningMean,
                           Data_t resultRunningVariance,
                           double epsilon,
                           Data_t resultSaveMean,
                           Data_t resultSaveInvVariance,
                           float inhw)
{

    float ctime = 0.;
    std::string kernel_subname{};
    visit_float(dtype, [&](auto as_float) {
        if(resultsave && resultrunning)
        {
            kernel_subname = kernel_name + "MeanVariance";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 0)(x, y);
            profileSequence(handle, 0, &ctime);

            kernel_subname = kernel_name + "FinalMeanVariance";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 1)(
                y,
                as_float(inhw),
                expAvgFactor,
                resultRunningMean,
                resultRunningVariance,
                epsilon,
                resultSaveMean,
                resultSaveInvVariance);
            profileSequence(handle, 1, &ctime);

            kernel_subname = kernel_name + "Norm";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 2)(
                x, y, bnScale, bnBias);
            profileSequence(handle, 2, &ctime);
        }
        else if(resultsave)
        {

            kernel_subname = kernel_name + "MeanVariance";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 0)(x, y);
            profileSequence(handle, 0, &ctime);

            kernel_subname = kernel_name + "FinalMeanVariance";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 1)(
                y, as_float(inhw), epsilon, resultSaveMean, resultSaveInvVariance);
            profileSequence(handle, 1, &ctime);

            kernel_subname = kernel_name + "Norm";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 2)(
                x, y, bnScale, bnBias);
            profileSequence(handle, 2, &ctime);
        }
        else if(resultrunning)
        {

            kernel_subname = kernel_name + "MeanVariance";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 0)(x, y);
            profileSequence(handle, 0, &ctime);

            kernel_subname = kernel_name + "FinalMeanVariance";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 1)(
                y, as_float(inhw), expAvgFactor, resultRunningMean, resultRunningVariance, epsilon);
            profileSequence(handle, 1, &ctime);

            kernel_subname = kernel_name + "Norm";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 2)(
                x, y, bnScale, bnBias);
            profileSequence(handle, 2, &ctime);
        }
        else
        {

            kernel_subname = kernel_name + "MeanVariance";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 0)(x, y);
            profileSequence(handle, 0, &ctime);

            kernel_subname = kernel_name + "FinalMeanVariance";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 1)(
                y, as_float(inhw), epsilon);
            profileSequence(handle, 1, &ctime);

            kernel_subname = kernel_name + "Norm";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 2)(
                x, y, bnScale, bnBias);
            profileSequence(handle, 2, &ctime);
        }
    });
}

void bnFwdTrainSelectSingle(const Handle& handle,
                            miopenDataType_t dtype,
                            const std::string& program_name,
                            const std::string& algo_name,
                            const std::string& kernel_name,
                            const std::string& network_config,
                            const std::string& parms,
                            const std::vector<size_t>& vld,
                            const std::vector<size_t>& vgd,
                            ConstData_t x,
                            Data_t y,
                            ConstData_t bnScale,
                            ConstData_t bnBias,
                            bool resultsave,
                            bool resultrunning,
                            double expAvgFactor,
                            Data_t resultRunningMean,
                            Data_t resultRunningVariance,
                            double epsilon,
                            Data_t resultSaveMean,
                            Data_t resultSaveInvVariance,
                            float inhw)
{

#if(MIOPEN_BN_CPP_DEBUG == 1)
    printf("Inside %s.\n", __FUNCTION__);
#endif
    visit_float(dtype, [&](auto as_float) {

#if(MIOPEN_BN_CPP_DEBUG == 1)
        printf("sizeof cast inhw: %d, ", sizeof(as_float(inhw)));
        printf("inhw: %e\n", inhw);
#endif
        if(resultsave && resultrunning)
        {
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x,
                y,
                bnScale,
                bnBias,
                as_float(inhw),
                expAvgFactor,
                resultRunningMean,
                resultRunningVariance,
                epsilon,
                resultSaveMean,
                resultSaveInvVariance);
        }
        else if(resultsave)
        {
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x,
                y,
                bnScale,
                bnBias,
                as_float(inhw),
                epsilon,
                resultSaveMean,
                resultSaveInvVariance);
        }
        else if(resultrunning)
        {
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x,
                y,
                bnScale,
                bnBias,
                as_float(inhw),
                expAvgFactor,
                resultRunningMean,
                resultRunningVariance,
                epsilon);
        }
        else
        {
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x, y, bnScale, bnBias, as_float(inhw), epsilon);
        }
    });
}

void bnBwdTrainSelectSingle(const Handle& handle,
                            miopenDataType_t dtype,
                            const std::string& program_name,
                            const std::string& algo_name,
                            const std::string& kernel_name,
                            const std::string& network_config,
                            const std::string& parms,
                            const std::vector<size_t>& vld,
                            const std::vector<size_t>& vgd,
                            ConstData_t x,
                            ConstData_t dy,
                            Data_t dx,
                            ConstData_t bnScale,
                            Data_t dScale,
                            Data_t dBias,
                            bool useSaved,
                            double epsilon,
                            ConstData_t savedMean,
                            ConstData_t savedInvVariance,
                            float inhw)
{

    visit_float(dtype, [&](auto as_float) {
        if(useSaved)
        {
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x, dy, dx, bnScale, dScale, dBias, savedMean, savedInvVariance, as_float(inhw));
        }
        else
        {
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x, dy, dx, bnScale, dScale, dBias, epsilon, as_float(inhw));
        }
    });
}

void bnBwdTrainSelectMulti(const Handle& handle,
                           miopenDataType_t dtype,
                           const std::string& program_name,
                           const std::string& algo_name,
                           const std::string& kernel_name,
                           const std::string& network_config,
                           const std::string& parms,
                           const std::vector<size_t>& vld,
                           const std::vector<size_t>& vgd,
                           ConstData_t x,
                           ConstData_t dy,
                           Data_t dx,
                           ConstData_t bnScale,
                           Data_t dScale,
                           Data_t dBias,
                           bool useSaved,
                           double epsilon,
                           ConstData_t savedMean,
                           ConstData_t savedInvVariance,
                           float inhw)
{
    float ctime = 0.;
    std::string kernel_subname{};
    visit_float(dtype, [&](auto as_float) {
        if(useSaved)
        {

            kernel_subname = kernel_name + "DScaleDBias";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 0)(
                x, dy, dx, savedMean, savedInvVariance);
            profileSequence(handle, 0, &ctime);

            kernel_subname = kernel_name + "FinalDScaleDBias";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 1)(
                dx, dScale, dBias);
            profileSequence(handle, 1, &ctime);

            kernel_subname = kernel_name + "DX";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 2)(
                x, dy, dx, bnScale, dScale, dBias, savedMean, savedInvVariance, as_float(inhw));
            profileSequence(handle, 2, &ctime);
        }
        else
        {
            kernel_subname = kernel_name + "MeanVariance";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 0)(x, dx);
            profileSequence(handle, 0, &ctime);

            kernel_subname = kernel_name + "FinalMeanVariance";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 1)(
                dx, as_float(inhw), epsilon);
            profileSequence(handle, 1, &ctime);

            kernel_subname = kernel_name + "DScaleDBias";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 2)(
                x, dy, dx);
            profileSequence(handle, 1, &ctime);

            kernel_subname = kernel_name + "FinalDScaleDBias";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 3)(
                dx, dScale, dBias);
            profileSequence(handle, 1, &ctime);

            kernel_subname = kernel_name + "DX";
            handle.AddKernel(
                algo_name, network_config, program_name, kernel_subname, vld, vgd, parms, 4)(
                x, dy, dx, bnScale, dScale, dBias, as_float(inhw));
            profileSequence(handle, 2, &ctime);
        }
    });
}

} // namespace miopen
