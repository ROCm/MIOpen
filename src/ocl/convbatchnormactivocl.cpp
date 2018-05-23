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
#include <miopen/batch_norm_activ.hpp>
#include <miopen/solver.hpp>
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/logger.hpp>
#include <chrono>

namespace miopen {

void DirectConvInference(Handle& handle,
                        const void* alpha,
                        const TensorDescriptor& xDesc,
                        ConstData_t x,
                        const TensorDescriptor& wDesc,
                        ConstData_t w,
                        const void* beta,
                        const TensorDescriptor& yDesc,
                        Data_t y)
{
    int pad_h = 0;
    int pad_w = 0;
    int u = 1;
    int v = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    if(x == nullptr || w == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != wDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != wDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    //    if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1]) {
    //        MIOPEN_THROW(miopenStatusBadParm);
    //    }
    if(xDesc.GetSize() < 3)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Only alpha=1 and beta=0 is supported");
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, wDesc, w);
    }

    //MIOPEN_LOG_I("workspace = " << workSpaceSize);
    if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1])
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    // TODO(paul): Replicating code for now.
    mlo_construct_direct2D construct_params(1); // forward
    construct_params.setOutputDescFromMLDesc(yDesc);
    construct_params.setInputDescFromMLDesc(xDesc);
    construct_params.setWeightDescFromMLDesc(wDesc);
    construct_params.setConvDescr(pad_h, pad_w, u, v, dilation_h, dilation_w);
    construct_params.setStream(&handle);

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);

    std::string algorithm_name = "miopenConvolutionFwdAlgoDirect";
    float padding_val          = 0;
    auto kernel                = handle.GetKernel(algorithm_name, network_config);

    visit_float(xDesc.GetType(), [&](auto as_float) {
        {
            ConvolutionContext context;
            construct_params.mloCopyTo(context);
            context.n_passes = true;

            Db db(context.GetPerfDbPath());
            solver::ConvSolution solution =
                FindSolution(solver::ConvOclDirectFwd11x11{}, context, db);

            if(solution.passes == 1)
            {
                kernel(x, w, y, as_float(padding_val));
            }
            else
            {
                // second kernel has
                network_config += "x1";
                auto kernel2 = handle.GetKernel(algorithm_name + "_pass2", network_config);

                handle.ResetKernelTime();
                kernel(x, w, y, as_float(padding_val));

                float time0 = handle.GetKernelTime();
                kernel2(x, w, y, as_float(padding_val));

                handle.AccumKernelTime(time0);
            }
        }
    });

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
}
}
