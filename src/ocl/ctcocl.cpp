/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include <miopen/config.h>
#include <miopen/activ.hpp>
#include <miopen/softmax.hpp>
#include <miopen/ctc.hpp>
#include <miopen/db.hpp>
#include <miopen/env.hpp>
#include <miopen/find_db.hpp>
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/check_numerics.hpp>
#include <vector>
#include <numeric>
#include <algorithm>

#define MAX_ACTIVE_THREADS (64 * 4 * 64)
#define MAX_LOCAL_MEM 65536

namespace miopen {

void CTCLossDescriptor::CTCLoss(Handle& handle,
                                const TensorDescriptor& probsDesc,
                                ConstData_t probs,
                                const int* labels,
                                const int* labelLengths,
                                const int* inputLengths,
                                Data_t losses,
                                const TensorDescriptor& gradientsDesc,
                                Data_t gradients,
                                miopenCTCLossAlgo_t algo,
                                Data_t workSpace,
                                size_t workSpaceSize) const
{
    (void)algo;
    (void)workSpaceSize;

    if(probsDesc.GetType() != miopenFloat && probsDesc.GetType() != miopenHalf)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(probsDesc.GetLengths()[0] != gradientsDesc.GetLengths()[0] ||
       probsDesc.GetLengths()[1] != gradientsDesc.GetLengths()[1] ||
       probsDesc.GetLengths()[2] != gradientsDesc.GetLengths()[2])
    {
        MIOPEN_THROW("probs tensor's dimension does not match gradients tensor's dimension");
    }

    int class_sz      = probsDesc.GetLengths()[2];
    int batch_size    = probsDesc.GetLengths()[1];
    int max_time_step = probsDesc.GetLengths()[0];
    std::vector<int> repeat(batch_size, 0);
    std::vector<int> labels_offset(batch_size, 0);
    int max_label_len   = 0;
    int total_label_len = 0;

    for(int i = 0; i < batch_size; i++)
    {
        if(inputLengths[i] > max_time_step)
        {
            MIOPEN_THROW("Wrong input time step");
        }
        max_label_len = std::max(max_label_len, labelLengths[i]);
        total_label_len += labelLengths[i];
        labels_offset[i] = i == 0 ? 0 : (labels_offset[i - 1] + labelLengths[i - 1]);

        for(int j = 0; j < labelLengths[i]; j++)
        {
            if(labels[labels_offset[i] + j] >= class_sz)
            {
                MIOPEN_THROW("Wrong label id");
            }
            if(j > 0)
            {
                if(labels[labels_offset[i] + j] == labels[labels_offset[i] + j - 1])
                    repeat[i]++;
            }
        }

        if(labelLengths[i] + repeat[i] > inputLengths[i])
        {
            MIOPEN_THROW("Error: label length exceeds input time step");
        }
    }

    int max_S_len       = 2 * max_label_len + 1;
    int lb_prime_offset = 4 * batch_size + total_label_len;
    int problog_offset  = lb_prime_offset + batch_size * max_S_len;

    if(probsDesc.GetType() == miopenHalf)
        problog_offset *= 2;

    int alpha_offset = problog_offset + class_sz * batch_size * max_time_step;
    int beta_offset  = alpha_offset + max_time_step * batch_size * max_S_len;
    int batch_bytes  = 4 * batch_size; // batch size multiples sizeof(int)

#if MIOPEN_BACKEND_OPENCL
    auto q = handle.GetStream();

    cl_context ctx;
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);

    clEnqueueWriteBuffer(q, workSpace, CL_FALSE, 0, batch_bytes, inputLengths, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(
        q, workSpace, CL_FALSE, batch_bytes, batch_bytes, labelLengths, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(q,
                         workSpace,
                         CL_FALSE,
                         2ULL * batch_bytes,
                         batch_bytes,
                         labels_offset.data(),
                         0,
                         nullptr,
                         nullptr);
    clEnqueueWriteBuffer(q,
                         workSpace,
                         CL_FALSE,
                         3ULL * batch_bytes,
                         batch_bytes,
                         repeat.data(),
                         0,
                         nullptr,
                         nullptr);
    clEnqueueWriteBuffer(q,
                         workSpace,
                         CL_FALSE,
                         4ULL * batch_bytes,
                         total_label_len * sizeof(int),
                         labels,
                         0,
                         nullptr,
                         nullptr);

#elif MIOPEN_BACKEND_HIP

    hipMemcpy(static_cast<int*>(workSpace), inputLengths, batch_bytes, hipMemcpyHostToDevice);
    hipMemcpy(static_cast<int*>(workSpace) + batch_size,
              labelLengths,
              batch_bytes,
              hipMemcpyHostToDevice);
    hipMemcpy(static_cast<int*>(workSpace) + 2 * static_cast<ptrdiff_t>(batch_size),
              labels_offset.data(),
              batch_bytes,
              hipMemcpyHostToDevice);
    hipMemcpy(static_cast<int*>(workSpace) + 3 * static_cast<ptrdiff_t>(batch_size),
              repeat.data(),
              batch_bytes,
              hipMemcpyHostToDevice);
    hipMemcpy(static_cast<int*>(workSpace) + 4 * static_cast<ptrdiff_t>(batch_size),
              labels,
              total_label_len * sizeof(int),
              hipMemcpyHostToDevice);
#endif

    std::string program_name = "MIOpenCTCLoss.cl";
    std::string kernel_name  = "CTCLossGPU";

    std::string network_config =
        "t" + std::to_string(max_time_step) + "n" + std::to_string(batch_size) + "a" +
        std::to_string(class_sz) + "mlb" + std::to_string(max_label_len) + "tlb" +
        std::to_string(total_label_len) + "sfm" +
        std::to_string(static_cast<int>(apply_softmax_layer)) + "b" +
        std::to_string(blank_label_id); // max timestep, batch, alphabet, max label length, total
                                        // label length, softmax layer indicator, blank ID

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    float time = 0.;
    if(apply_softmax_layer)
    {
        std::vector<int> sfm_size(4, 1);
        sfm_size[0]   = max_time_step * batch_size;
        sfm_size[1]   = class_sz;
        auto sfm_desc = miopen::TensorDescriptor(probsDesc.GetType(), sfm_size);

        float alpha = 1;
        float beta  = 0;
        SoftmaxForward(handle,
                       &alpha,
                       &beta,
                       sfm_desc,
                       probs,
                       sfm_desc,
                       workSpace,
                       MIOPEN_SOFTMAX_LOG,
                       MIOPEN_SOFTMAX_MODE_CHANNEL,
                       0,
                       problog_offset);
        if(handle.IsProfilingEnabled())
            time += handle.GetKernelTime();
    }

    if(!kernels.empty())
    {
        auto kernel = kernels.front();

        kernel(probs, workSpace, workSpace, losses, gradients);
    }
    else
    {
        std::string params;

        size_t work_per_grp = batch_size <= 64 ? 256 : batch_size <= 128 ? 128 : 64;
        assert(512 >= work_per_grp && work_per_grp > 0);
        size_t glb_sz  = batch_size < static_cast<size_t>(MAX_ACTIVE_THREADS) / work_per_grp
                             ? batch_size * work_per_grp
                             : static_cast<size_t>(MAX_ACTIVE_THREADS);
        size_t grp_num = glb_sz / work_per_grp;

        size_t lcl_mem_per_grp = MAX_LOCAL_MEM / 2 / (512 / work_per_grp);

        params += " -DCLASS_SZ=" + std::to_string(class_sz) +
                  " -DBATCH_SZ=" + std::to_string(batch_size) +
                  " -DMAX_TSTEP=" + std::to_string(max_time_step) +
                  " -DMAX_LB_LEN=" + std::to_string(max_label_len) +
                  " -DTOTAL_LB_LEN=" + std::to_string(total_label_len) +
                  " -DMAX_S_LEN=" + std::to_string(max_S_len) +
                  " -DLB_PRIME_OFFSET=" + std::to_string(lb_prime_offset) +
                  " -DPROBLOG_OFFSET=" + std::to_string(problog_offset) +
                  " -DALPHA_OFFSET=" + std::to_string(alpha_offset) +
                  " -DBETA_OFFSET=" + std::to_string(beta_offset) +
                  " -DWORK_PER_GRP=" + std::to_string(work_per_grp) +
                  " -DGRP_NUM=" + std::to_string(grp_num) +
                  " -DBLANK_LB_ID=" + std::to_string(blank_label_id);

        if(!probsDesc.IsPacked())
        {
            params += " -DPROBS_STRIDE0=" + std::to_string(probsDesc.GetStrides()[0]) +
                      " -DPROBS_STRIDE1=" + std::to_string(probsDesc.GetStrides()[1]);
        }

        if(!gradientsDesc.IsPacked())
        {
            params += " -DGRADS_STRIDE0=" + std::to_string(gradientsDesc.GetStrides()[0]) +
                      " -DGRADS_STRIDE1=" + std::to_string(gradientsDesc.GetStrides()[1]);
        }

        params += " -DSOFTMAX_APPLIED=" + std::to_string(static_cast<int>(apply_softmax_layer)) +
                  " -DSOFTMAX_LEN=" + std::to_string(class_sz);

#if MIOPEN_BACKEND_OPENCL
        if(class_sz <= lcl_mem_per_grp)
            params += " -DOPT_LCL_MEM_GRAD";
#endif

        if(static_cast<size_t>(max_S_len) * 2
#if MIOPEN_BACKEND_OPENCL
               + class_sz
#endif
           <= lcl_mem_per_grp)
        {
            params += " -DOPT_LCL_MEM_BETA";
        }

        if(static_cast<size_t>(max_S_len) * 3
#if MIOPEN_BACKEND_OPENCL
               + class_sz
#endif
           <= lcl_mem_per_grp)
        {
            params += " -DOPT_LCL_MEM_LB";
        }

        if(probsDesc.GetType() == miopenHalf)
            params += " -DMIOPEN_USE_FP16=1";
        else
            params += " -DMIOPEN_USE_FP32=1 -DOPT_ATOMIC_LOGADDEXP";

#if MIOPEN_BACKEND_HIP
        params += " -DUSE_HIP_BACKEND=1";
#elif MIOPEN_BACKEND_OPENCL
        params += " -DUSE_OCL_BACKEND=1";
#endif

        const std::vector<size_t> vld{work_per_grp, 1, 1};
        const std::vector<size_t> vgd{glb_sz, 1, 1};

        handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
            probs, workSpace, workSpace, losses, gradients);
    }
    if(handle.IsProfilingEnabled())
        handle.AccumKernelTime(time);
}

} // namespace miopen
