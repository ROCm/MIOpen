// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/ctc.hpp>
#include <miopen/find_db.hpp>

namespace miopen {

CTCLossDescriptor::CTCLossDescriptor()
{
    dataType            = miopenFloat;
    apply_softmax_layer = true;
    blank_label_id      = 0;
}

void CTCLossDescriptor::CTCLoss(const Handle& handle,
                                const TensorDescriptor& probsDesc,
                                ConstData_t probs,
                                const int* labels,
                                const int* labelLengths,
                                const int* inputLengths,
                                Data_t losses,
                                const TensorDescriptor& gradientsDesc,
                                Data_t gradients,
                                [[maybe_unused]] miopenCTCLossAlgo_t algo,
                                Data_t workSpace,
                                [[maybe_unused]] size_t workSpaceSize) const
{
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
                {
                    repeat[i]++;
                }
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
    {
        problog_offset *= 2;
    }

    int alpha_offset = problog_offset + class_sz * batch_size * max_time_step;
    int beta_offset  = alpha_offset + max_time_step * batch_size * max_S_len;
    int batch_bytes  = 4 * batch_size; // batch size multiples sizeof(int)

    hipMemcpyWithStream(static_cast<int*>(workSpace),
                        inputLengths,
                        batch_bytes,
                        hipMemcpyHostToDevice,
                        handle.GetStream());
    hipMemcpyWithStream(static_cast<int*>(workSpace) + batch_size,
                        labelLengths,
                        batch_bytes,
                        hipMemcpyHostToDevice,
                        handle.GetStream());
    hipMemcpyWithStream(static_cast<int*>(workSpace) + 2 * static_cast<ptrdiff_t>(batch_size),
                        labels_offset.data(),
                        batch_bytes,
                        hipMemcpyHostToDevice,
                        handle.GetStream());
    hipMemcpyWithStream(static_cast<int*>(workSpace) + 3 * static_cast<ptrdiff_t>(batch_size),
                        repeat.data(),
                        batch_bytes,
                        hipMemcpyHostToDevice,
                        handle.GetStream());
    hipMemcpyWithStream(static_cast<int*>(workSpace) + 4 * static_cast<ptrdiff_t>(batch_size),
                        labels,
                        total_label_len * sizeof(int),
                        hipMemcpyHostToDevice,
                        handle.GetStream());

    std::string network_config =
        "t" + std::to_string(max_time_step) + "n" + std::to_string(batch_size) + "a" +
        std::to_string(class_sz) + "mlb" + std::to_string(max_label_len) + "tlb" +
        std::to_string(total_label_len) + "sfm" +
        std::to_string(static_cast<int>(apply_softmax_layer)) + "b" +
        std::to_string(blank_label_id); // max timestep, batch, alphabet, max label length, total
                                        // label length, softmax layer indicator, blank ID

    std::string kernel_name = "CTCLossGPU";
    auto&& kernels          = handle.GetKernels(kernel_name, network_config);

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
        {
            time += handle.GetKernelTime();
        }
    }

    if(!kernels.empty())
    {
        auto kernel = kernels.front();
        kernel(probs, workSpace, workSpace, losses, gradients);
    }
    else
    {
        // Work out group-size first, then global-size as the amount of work-groups, to
        // ensure a uniform number of total threads.
        size_t group_size = batch_size <= 64 ? 256 : (batch_size <= 128) ? 128 : 64;
        constexpr size_t max_active_threads = static_cast<size_t>(64 * 4 * 64); // 16384
        size_t max_num_groups               = max_active_threads / group_size;
        size_t global_size =
            (batch_size < max_num_groups) ? batch_size * group_size : max_active_threads;
        size_t num_groups = global_size / group_size;

        constexpr size_t max_local_mem = 65536;
        size_t lcl_mem_per_grp         = max_local_mem / 2 / (512 / group_size);

        int blank_label;
        if(blank_label_id < 0)
        {
            blank_label = 0;
        }
        else if(blank_label_id >= class_sz)
        {
            blank_label = class_sz - 1;
        }
        else
        {
            blank_label = blank_label_id;
        }
        // clang-format off
        std::string params = " -DCLASS_SZ=" + std::to_string(class_sz) +
                             " -DBATCH_SZ=" + std::to_string(batch_size) +
                             " -DMAX_TSTEP=" + std::to_string(max_time_step) +
                             " -DMAX_LB_LEN=" + std::to_string(max_label_len) +
                             " -DTOTAL_LB_LEN=" + std::to_string(total_label_len) +
                             " -DMAX_S_LEN=" + std::to_string(max_S_len) +
                             " -DLB_PRIME_OFFSET=" + std::to_string(lb_prime_offset) +
                             " -DPROBLOG_OFFSET=" + std::to_string(problog_offset) +
                             " -DALPHA_OFFSET=" + std::to_string(alpha_offset) +
                             " -DBETA_OFFSET=" + std::to_string(beta_offset) +
                             " -DWORK_PER_GRP=" + std::to_string(group_size) +
                             " -DGRP_NUM=" + std::to_string(num_groups) +
                             " -DBLANK_LB=" + std::to_string(blank_label) +
                             " -DSOFTMAX_APPLIED=" +
                             std::to_string(static_cast<int>(apply_softmax_layer));
        // clang-format on
        if(apply_softmax_layer || probsDesc.IsPacked())
        {
            params += " -DPROBS_STRIDE0=" + std::to_string(class_sz * batch_size) +
                      " -DPROBS_STRIDE1=" + std::to_string(class_sz);
        }
        else
        {
            params += " -DPROBS_STRIDE0=" + std::to_string(probsDesc.GetStrides()[0]) +
                      " -DPROBS_STRIDE1=" + std::to_string(probsDesc.GetStrides()[1]);
        }

        if(!gradientsDesc.IsPacked())
        {
            params += " -DGRADS_STRIDE0=" + std::to_string(gradientsDesc.GetStrides()[0]) +
                      " -DGRADS_STRIDE1=" + std::to_string(gradientsDesc.GetStrides()[1]);
        }
        else
        {
            params += " -DGRADS_STRIDE0=" + std::to_string(class_sz * batch_size) +
                      " -DGRADS_STRIDE1=" + std::to_string(class_sz);
        }

        bool opt_lcl_mem_beta = static_cast<size_t>(max_S_len) * 2 <= lcl_mem_per_grp;
        params += " -DOPT_LCL_MEM_BETA=" + std::to_string(static_cast<int>(opt_lcl_mem_beta));

        bool opt_lcl_mem_lb = static_cast<size_t>(max_S_len) * 3 <= lcl_mem_per_grp;
        params += " -DOPT_LCL_MEM_LB=" + std::to_string(static_cast<int>(opt_lcl_mem_lb));

        if(probsDesc.GetType() == miopenHalf)
        {
            // FP16 is not yet fully supported
            // See https://github.com/ROCm/rocm-libraries/issues/2866
            params += " -DMIOPEN_USE_FP16=1 -DOPT_ATOMIC_LOGADDEXP=0";
        }
        else
        {
            params += " -DMIOPEN_USE_FP32=1 -DOPT_ATOMIC_LOGADDEXP=1";
        }

        const std::vector<size_t> vld{group_size, 1, 1};
        const std::vector<size_t> vgd{global_size, 1, 1};
        std::string program_name = "MIOpenCTCLoss.cpp";
        handle.AddKernel(kernel_name, network_config, program_name, kernel_name, vld, vgd, params)(
            probs, workSpace, workSpace, losses, gradients);
    }
    if(handle.IsProfilingEnabled())
    {
        handle.AccumKernelTime(time);
    }
}

size_t CTCLossDescriptor::GetCTCLossWorkspaceSize(const Handle& handle,
                                                  const TensorDescriptor& probsDesc,
                                                  const TensorDescriptor& gradientsDesc,
                                                  const int* labels,
                                                  const int* labelLengths,
                                                  const int* inputLengths,
                                                  miopenCTCLossAlgo_t algo) const
{
    (void)algo;

    if(probsDesc.GetLengths()[0] != gradientsDesc.GetLengths()[0] ||
       probsDesc.GetLengths()[1] != gradientsDesc.GetLengths()[1] ||
       probsDesc.GetLengths()[2] != gradientsDesc.GetLengths()[2])
    {
        MIOPEN_THROW(
            miopenStatusBadParm,
            "The probability tensor's dimensions do not match the gradient tensor's dimensions");
    }

    int class_sz        = probsDesc.GetLengths()[2];
    int batch_size      = probsDesc.GetLengths()[1];
    int max_time_step   = probsDesc.GetLengths()[0];
    int max_label_len   = 0;
    int total_label_len = 0;
    std::vector<int> repeat(batch_size, 0);
    std::vector<int> labels_offset(batch_size, 0);
    size_t wksp_sz_lb  = 0;
    size_t wksp_sz_dat = 0;

    for(int i = 0; i < batch_size; i++)
    {
        if(inputLengths[i] > max_time_step)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Wrong input time step");
        }
        max_label_len = std::max(max_label_len, labelLengths[i]);
        total_label_len += labelLengths[i];
        labels_offset[i] = i == 0 ? 0 : (labels_offset[i - 1] + labelLengths[i - 1]);

        for(int j = 0; j < labelLengths[i]; j++)
        {
            if(labels[labels_offset[i] + j] >= class_sz)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Wrong label id at batch");
            }
            if(j > 0)
            {
                if(labels[labels_offset[i] + j] == labels[labels_offset[i] + j - 1])
                    repeat[i]++;
            }
        }

        if(labelLengths[i] + repeat[i] > inputLengths[i])
        {
            MIOPEN_THROW(miopenStatusBadParm, "Error: label length exceeds input time step");
        }
    }

    // input length
    wksp_sz_lb += batch_size;

    // label length
    wksp_sz_lb += batch_size;

    // label offset
    wksp_sz_lb += batch_size;

    // label repeat
    wksp_sz_lb += batch_size;

    // labels
    wksp_sz_lb += total_label_len;

    // labels with blanks
    wksp_sz_lb += static_cast<size_t>(batch_size) * (2 * static_cast<size_t>(max_label_len) + 1);

    // logsoftmax of probs
    wksp_sz_dat += static_cast<size_t>(max_time_step) * batch_size * class_sz;

    // alphas
    wksp_sz_dat += static_cast<size_t>(max_time_step) * batch_size *
                   (2 * static_cast<size_t>(max_label_len) + 1);

    // beta buffer
    wksp_sz_dat +=
        2 * static_cast<size_t>(batch_size) * (2 * static_cast<size_t>(max_label_len) + 1);

    size_t total_size = wksp_sz_dat * sizeof(float) + wksp_sz_lb * sizeof(int);
    if(total_size > handle.GetMaxMemoryAllocSize())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Error: Workspace size exceeds GPU memory capacity");
    }
    return total_size;
}

std::ostream& operator<<(std::ostream& stream, const CTCLossDescriptor& r)
{
    stream << r.dataType << ", ";
    return stream;
}

} // namespace miopen
