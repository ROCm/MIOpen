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

#include <miopen/errors.hpp>
#include <miopen/batch_norm.hpp>
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
        newlens[0] = newlens[2] = newlens[3] = 1; // TODO: support 5D
    }
    else
    {
        newlens[0] = 1;
        newlens[2] = lengths[2];
        newlens[3] = lengths[3];
        ; // TODO: support 5D
    }
    derivedBnDesc = TensorDescriptor(xDesc.GetType(), newlens.data(), xDesc.GetSize());
}

inline void profileSequence(Handle& handle, unsigned char select)
{

    float ktime        = 0.;
    static float ctime = 0.; // TODO make this non-static parameter
    assert((select < 3) && "profileSequence case incorrect");
    switch(select)
    {

    case 0:
        if(handle.IsProfilingEnabled())
        {
            ctime = 0.;
            handle.ResetKernelTime();
            ktime = handle.GetKernelTime();
            ctime = ktime;

#if(MIO_BN_CPP_PROF == 1)
            printf("ktime: %f\n", ktime);
            printf("ctime: %f\n", ctime);
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
            ctime += ktime;

#if(MIO_BN_CPP_PROF == 1)
            printf("ktime: %f\n", ktime);
            printf("ctime: %f\n", ctime);
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
            handle.GetKernelTime();
            handle.AccumKernelTime(ctime);
        }
        break;
    }
}

void bnFwdTrainSelectMulti(Handle& handle,
                           std::string& program_name,
                           std::string& algo_name,
                           std::string& kernel_name,
                           std::string& network_config,
                           std::string& parms,
                           std::vector<size_t>& vld,
                           std::vector<size_t>& vgd,
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

    //#if(MIO_BN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
    //#endif

    std::string kernel_subname{};
    if(resultsave && resultrunning)
    {
        kernel_subname = kernel_name + "Mean";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y);
        profileSequence(handle, 0);

        kernel_subname = kernel_name + "FinalMean";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            y, inhw, expAvgFactor, resultRunningMean, resultSaveMean);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "Variance";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y);

        profileSequence(handle, 1);

        kernel_subname = kernel_name + "FinalVariance";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            y, inhw, expAvgFactor, resultRunningVariance, epsilon, resultSaveInvVariance);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "Norm";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y, bnScale, bnBias);
        profileSequence(handle, 2);
    }
    else if(resultsave)
    {

        kernel_subname = kernel_name + "Mean";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y);
        profileSequence(handle, 0);

        kernel_subname = kernel_name + "FinalMean";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            y, inhw, resultSaveMean);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "Variance";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "FinalVariance";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            y, inhw, epsilon, resultSaveInvVariance);

        profileSequence(handle, 1);

        kernel_subname = kernel_name + "Norm";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y, bnScale, bnBias);
        profileSequence(handle, 2);
    }
    else if(resultrunning)
    {

        kernel_subname = kernel_name + "Mean";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y);
        profileSequence(handle, 0);

        kernel_subname = kernel_name + "FinalMean";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            y, inhw, expAvgFactor, resultRunningMean);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "Variance";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "FinalVariance";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            y, inhw, expAvgFactor, resultRunningVariance, epsilon);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "Norm";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y, bnScale, bnBias);
        profileSequence(handle, 2);
    }
    else
    {

        kernel_subname = kernel_name + "Mean";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y);
        profileSequence(handle, 0);

        kernel_subname = kernel_name + "FinalMean";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            y, inhw);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "Variance";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "FinalVariance";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            y, inhw, epsilon);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "Norm";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, y, bnScale, bnBias);
        profileSequence(handle, 2);
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: KERN LAUNCHES: "
              << std::chrono::duration<double>(t_end - t_start).count() * 1000.0 << " ms."
              << std::endl;
}

void bnFwdTrainSelectSingle(Handle& handle,
                            std::string& program_name,
                            std::string& algo_name,
                            std::string& kernel_name,
                            std::string& network_config,
                            std::string& parms,
                            std::vector<size_t>& vld,
                            std::vector<size_t>& vgd,
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

    if(resultsave && resultrunning)
    {
        handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x,
            y,
            bnScale,
            bnBias,
            inhw,
            expAvgFactor,
            resultRunningMean,
            resultRunningVariance,
            epsilon,
            resultSaveMean,
            resultSaveInvVariance);
    }
    else if(resultsave)
    {
        handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x, y, bnScale, bnBias, inhw, epsilon, resultSaveMean, resultSaveInvVariance);
    }
    else if(resultrunning)
    {
        handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x,
            y,
            bnScale,
            bnBias,
            inhw,
            expAvgFactor,
            resultRunningMean,
            resultRunningVariance,
            epsilon);
    }
    else
    {
        handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x, y, bnScale, bnBias, inhw, epsilon);
    }
}

void bnBwdTrainSelectSingle(Handle& handle,
                            std::string& program_name,
                            std::string& algo_name,
                            std::string& kernel_name,
                            std::string& network_config,
                            std::string& parms,
                            std::vector<size_t>& vld,
                            std::vector<size_t>& vgd,
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

    if(useSaved)
    {
        handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x, dy, dx, bnScale, dScale, dBias, savedMean, savedInvVariance, inhw);
    }
    else
    {
        if(handle.GetDeviceName() == "gfx803")
            parms += " -DMIO_BN_NODPP=1";

        handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x, dy, dx, bnScale, dScale, dBias, epsilon, inhw);
    }
}

void bnBwdTrainSelectMulti(Handle& handle,
                           std::string& program_name,
                           std::string& algo_name,
                           std::string& kernel_name,
                           std::string& network_config,
                           std::string& parms,
                           std::vector<size_t>& vld,
                           std::vector<size_t>& vgd,
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
{ // TODO use this param somewhere

    //#if(MIO_BN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
    //#endif

    std::string kernel_subname{};
    if(useSaved)
    {

        kernel_subname = kernel_name + "DBias";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            dy, dx);
        profileSequence(handle, 0);

        kernel_subname = kernel_name + "DScale";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, dy, dx, savedMean, savedInvVariance);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "FinalDBias";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            dx, dBias);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "FinalDScale";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            dx, dScale);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "DX";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, dy, dx, bnScale, dScale, dBias, savedMean, savedInvVariance, inhw);
        profileSequence(handle, 2);
    }
    else
    {
        if(handle.GetDeviceName() == "gfx803")
            parms += " -DMIO_BN_NODPP=1";

        kernel_subname = kernel_name + "Mean";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, dx);
        profileSequence(handle, 0);

        kernel_subname = kernel_name + "DBias";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            dy, dx);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "FinalDBias";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            dx, dBias);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "FinalMean";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            dx, inhw);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "Variance";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, dx);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "FinalVariance";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            dx, inhw, epsilon);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "DScale";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, dy, dx);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "FinalDScale";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            dx, dScale);
        profileSequence(handle, 1);

        kernel_subname = kernel_name + "DX";
        handle.GetKernel(algo_name, network_config, program_name, kernel_subname, vld, vgd, parms)(
            x, dy, dx, bnScale, dScale, dBias, inhw);
        profileSequence(handle, 2);
    }
    handle.Finish();
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Wall clock: KERN LAUNCHES: "
              << std::chrono::duration<double>(t_end - t_start).count() * 1000.0 << " ms."
              << std::endl;
}

} // namespace miopen
