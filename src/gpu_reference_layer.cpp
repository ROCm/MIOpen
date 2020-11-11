/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/gpu_reference_layer.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/datatype.hpp>
#include <miopen/problem_description.hpp>

namespace miopen {

static std::string getKernelName(const ProblemDescription& conv_param)
{
    std::string kernel_name("n/a");
    if(conv_param.direction.IsForward())
    {
        if(conv_param.in_layout == "NCHW")
        {
            if(conv_param.in_data_type == miopenFloat)
            {
                kernel_name = "naive_conv_fwd_nchw_fp32";
            }
            else if(conv_param.in_data_type == miopenHalf)
            {
                kernel_name = "naive_conv_fwd_nchw_fp16";
            }
            else
            {
                MIOPEN_LOG_E("unsupported datatype:" << conv_param.in_data_type);
            }
        }
        else if(conv_param.in_layout == "NCDHW")
        {
            if(conv_param.in_data_type == miopenFloat)
            {
                kernel_name = "naive_conv_fwd_ncdhw_fp32";
            }
            else if(conv_param.in_data_type == miopenHalf)
            {
                kernel_name = "naive_conv_fwd_ncdhw_fp16";
            }
            else
            {
                MIOPEN_LOG_E("unsupported datatype:" << conv_param.in_data_type);
            }
        }
    }
    else if(conv_param.direction.IsBackwardData())
    {
        if(conv_param.in_layout == "NCHW")
        {
            if(conv_param.in_data_type == miopenFloat)
            {
                kernel_name = "naive_conv_bwd_nchw_fp32";
            }
            else if(conv_param.in_data_type == miopenHalf)
            {
                kernel_name = "naive_conv_bwd_nchw_fp16";
            }
            else
            {
                MIOPEN_LOG_E("unsupported datatype:" << conv_param.in_data_type);
            }
        }
        else if(conv_param.in_layout == "NCDHW")
        {
            if(conv_param.in_data_type == miopenFloat)
            {
                kernel_name = "naive_conv_bwd_ncdhw_fp32";
            }
            else if(conv_param.in_data_type == miopenHalf)
            {
                kernel_name = "naive_conv_bwd_ncdhw_fp16";
            }
            else
            {
                MIOPEN_LOG_E("unsupported datatype:" << conv_param.in_data_type);
            }
        }
    }
    else if(conv_param.direction.IsBackwardWrW())
    {
        if(conv_param.in_layout == "NCHW")
        {
            if(conv_param.in_data_type == miopenFloat)
            {
                kernel_name = "naive_conv_wrw_nchw_fp32";
            }
            else if(conv_param.in_data_type == miopenHalf)
            {
                kernel_name = "naive_conv_wrw_nchw_fp16";
            }
            else
            {
                MIOPEN_LOG_E("unsupported datatype:" << conv_param.in_data_type);
            }
        }
        else if(conv_param.in_layout == "NCDHW")
        {
            if(conv_param.in_data_type == miopenFloat)
            {
                kernel_name = "naive_conv_wrw_ncdhw_fp32";
            }
            else if(conv_param.in_data_type == miopenHalf)
            {
                kernel_name = "naive_conv_wrw_ncdhw_fp16";
            }
            else
            {
                MIOPEN_LOG_E("unsupported datatype:" << conv_param.in_data_type);
            }
        }
    }
    return kernel_name;
}

void GPUReferenceConvolutionForward(const Handle& handle,
                                    const ProblemDescription& conv_param,
                                    ConstData_t input_data,
                                    ConstData_t weight_data,
                                    Data_t output_data)
{
#if MIOPEN_BACKEND_OPENCL
    MIOPEN_LOG_E("currently only support hip backend");
    return;
#endif
    if(!conv_param.IsLayoutDefault())
    {
        MIOPEN_LOG_E("not default layout, will not run");
        return;
    }
    if(!conv_param.direction.IsForward())
    {
        MIOPEN_THROW("should be used in forward direction");
    }
    // clang-format on
    int di          = conv_param.in_depth;
    int hi          = conv_param.in_height;
    int wi          = conv_param.in_width;
    int n           = conv_param.batch_sz;
    int k           = conv_param.n_outputs;
    int c           = conv_param.n_inputs;
    int do_         = conv_param.out_depth;
    int ho          = conv_param.out_height;
    int wo          = conv_param.out_width;
    int sz          = conv_param.kernel_stride_d;
    int sy          = conv_param.kernel_stride_h;
    int sx          = conv_param.kernel_stride_w;
    int dz          = conv_param.kernel_dilation_d;
    int dy          = conv_param.kernel_dilation_h;
    int dx          = conv_param.kernel_dilation_w;
    int pz          = conv_param.pad_d;
    int py          = conv_param.pad_h;
    int px          = conv_param.pad_w;
    int fz          = conv_param.kernel_size_d;
    int fy          = conv_param.kernel_size_h;
    int fx          = conv_param.kernel_size_w;
    int group       = conv_param.group_counts;
    int c_per_group = c / group;
    int k_per_group = k / group;
    // clang-format off

    std::string program_name      = "naive_conv.cpp";
    std::string kernel_name       =  getKernelName(conv_param);
    const size_t block_size = 256;
    const size_t grid_size = block_size * n * k;
    const std::vector<size_t> vld = {block_size, size_t{1}, size_t{1}};
    const std::vector<size_t> vgd = {grid_size, size_t{1}, size_t{1}};
    auto kernel = handle.AddKernel("GPUReferenceConvolutionForward", "", program_name, kernel_name, vld, vgd, "");

    if(conv_param.in_layout == "NCHW"){
        // clang-format off
        kernel( input_data,
                weight_data,
                output_data,
                hi,
                wi,
                n,
                k_per_group,
                c_per_group,
                ho,
                wo,
                sy,
                sx,
                dy,
                dx,
                py,
                px,
                fy,
                fx,
                group);
        // clang-format on
    }
    else if(conv_param.in_layout == "NCDHW")
    {
        // clang-format off
        kernel( input_data,
                weight_data,
                output_data,
                di,
                hi,
                wi,
                n,
                k_per_group,
                c_per_group,
                do_,
                ho,
                wo,
                sz,
                sy,
                sx,
                dz,
                dy,
                dx,
                pz,
                py,
                px,
                fz,
                fy,
                fx,
                group);
        // clang-format on
    }
    else { MIOPEN_LOG_E("unsupported layout:" << conv_param.in_layout); }
}

void GPUReferenceConvolutionBackwardData(const Handle& handle,
                                         const ProblemDescription& conv_param,
                                         Data_t input_data,
                                         ConstData_t weight_data,
                                         ConstData_t output_data)
{
#if MIOPEN_BACKEND_OPENCL
    MIOPEN_LOG_E("currently only support hip backend");
    return;
#endif
    if(!conv_param.IsLayoutDefault())
    {
        MIOPEN_LOG_E("not default layout, will not run");
        return;
    }
    if(!conv_param.direction.IsBackwardData())
    {
        MIOPEN_THROW("should be used in backward_data direction");
    }
    // clang-format on
    int di          = conv_param.out_depth;
    int hi          = conv_param.out_height;
    int wi          = conv_param.out_width;
    int n           = conv_param.batch_sz;
    int k           = conv_param.n_inputs;
    int c           = conv_param.n_outputs;
    int do_         = conv_param.in_depth;
    int ho          = conv_param.in_height;
    int wo          = conv_param.in_width;
    int sz          = conv_param.in_depth > 1 ? conv_param.kernel_stride_d : 1;
    int sy          = conv_param.in_height > 1 ? conv_param.kernel_stride_h : 1;
    int sx          = conv_param.in_width > 1 ? conv_param.kernel_stride_w : 1;
    int dz          = conv_param.kernel_size_d > 1 ? conv_param.kernel_dilation_d : 1;
    int dy          = conv_param.kernel_size_h > 1 ? conv_param.kernel_dilation_h : 1;
    int dx          = conv_param.kernel_size_w > 1 ? conv_param.kernel_dilation_w : 1;
    int pz          = conv_param.pad_d;
    int py          = conv_param.pad_h;
    int px          = conv_param.pad_w;
    int fz          = conv_param.kernel_size_d;
    int fy          = conv_param.kernel_size_h;
    int fx          = conv_param.kernel_size_w;
    int group       = conv_param.group_counts;
    int c_per_group = c / group;
    int k_per_group = k / group;
    // clang-format off

    std::string program_name      = "naive_conv.cpp";
    std::string kernel_name       =  getKernelName(conv_param);
    const size_t block_size = 256;
    const size_t grid_size = block_size * n * c;
    const std::vector<size_t> vld = {block_size, size_t{1}, size_t{1}};
    const std::vector<size_t> vgd = {grid_size, size_t{1}, size_t{1}};
    auto kernel = handle.AddKernel("GPUReferenceConvolutionBackwardData", "", program_name, kernel_name, vld, vgd, "");

    if(conv_param.in_layout == "NCHW"){
        // clang-format off
        kernel( input_data,
                weight_data,
                output_data,
                hi,
                wi,
                n,
                k_per_group,
                c_per_group,
                ho,
                wo,
                sy,
                sx,
                dy,
                dx,
                py,
                px,
                fy,
                fx,
                group);
        // clang-format on
    }
    else if(conv_param.in_layout == "NCDHW")
    {
        // clang-format off
        kernel( input_data,
                weight_data,
                output_data,
                di,
                hi,
                wi,
                n,
                k_per_group,
                c_per_group,
                do_,
                ho,
                wo,
                sz,
                sy,
                sx,
                dz,
                dy,
                dx,
                pz,
                py,
                px,
                fz,
                fy,
                fx,
                group);
        // clang-format on
    }
    else { MIOPEN_LOG_E("unsupported layout:" << conv_param.in_layout); }
}

void GPUReferenceConvolutionBackwardWeight(const Handle& handle,
                                           const ProblemDescription& conv_param,
                                           ConstData_t input_data,
                                           Data_t weight_data,
                                           ConstData_t output_data)
{
#if MIOPEN_BACKEND_OPENCL
    MIOPEN_LOG_E("currently only support hip backend");
    return;
#endif
    if(!conv_param.IsLayoutDefault())
    {
        MIOPEN_LOG_E("not default layout, will not run");
        return;
    }
    if(!conv_param.direction.IsBackwardWrW())
    {
        MIOPEN_THROW("should be used in backward_weight direction");
    }
    // clang-format on
    int di          = conv_param.out_depth;
    int hi          = conv_param.out_height;
    int wi          = conv_param.out_width;
    int n           = conv_param.batch_sz;
    int k           = conv_param.n_inputs;
    int c           = conv_param.n_outputs;
    int do_         = conv_param.in_depth;
    int ho          = conv_param.in_height;
    int wo          = conv_param.in_width;
    int sz          = conv_param.in_depth > 1 ? conv_param.kernel_stride_d : 1;
    int sy          = conv_param.in_height > 1 ? conv_param.kernel_stride_h : 1;
    int sx          = conv_param.in_width > 1 ? conv_param.kernel_stride_w : 1;
    int dz          = conv_param.kernel_size_d > 1 ? conv_param.kernel_dilation_d : 1;
    int dy          = conv_param.kernel_size_h > 1 ? conv_param.kernel_dilation_h : 1;
    int dx          = conv_param.kernel_size_w > 1 ? conv_param.kernel_dilation_w : 1;
    int pz          = conv_param.pad_d;
    int py          = conv_param.pad_h;
    int px          = conv_param.pad_w;
    int fz          = conv_param.kernel_size_d;
    int fy          = conv_param.kernel_size_h;
    int fx          = conv_param.kernel_size_w;
    int group       = conv_param.group_counts;
    int c_per_group = c / group;
    int k_per_group = k / group;
    // clang-format off

    std::string program_name      = "naive_conv.cpp";
    std::string kernel_name       =  getKernelName(conv_param);
    const size_t block_size = 256;
    const size_t grid_size = block_size * k;
    const std::vector<size_t> vld = {block_size, size_t{1}, size_t{1}};
    const std::vector<size_t> vgd = {grid_size, size_t{1}, size_t{1}};
    auto kernel = handle.AddKernel("GPUReferenceConvolutionBackwardWeight", "", program_name, kernel_name, vld, vgd, "");

    if(conv_param.in_layout == "NCHW"){
        // clang-format off
        kernel( input_data,
                weight_data,
                output_data,
                hi,
                wi,
                n,
                k_per_group,
                c_per_group,
                ho,
                wo,
                sy,
                sx,
                dy,
                dx,
                py,
                px,
                fy,
                fx,
                group);
        // clang-format on
    }
    else if(conv_param.in_layout == "NCDHW")
    {
        // clang-format off
        kernel( input_data,
                weight_data,
                output_data,
                di,
                hi,
                wi,
                n,
                k_per_group,
                c_per_group,
                do_,
                ho,
                wo,
                sz,
                sy,
                sx,
                dz,
                dy,
                dx,
                pz,
                py,
                px,
                fz,
                fy,
                fx,
                group);
        // clang-format on
    }
    else { MIOPEN_LOG_E("unsupported layout:" << conv_param.in_layout); }
}
} // namespace miopen
