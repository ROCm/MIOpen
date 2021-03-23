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

#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/convolution.hpp>
#include <miopen/problem_description.hpp>
#include <miopen/tensor.hpp>
#include <miopen/bfloat16.hpp>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <type_traits>
#include <half.hpp>
#include "test.hpp"
#include "driver.hpp"
#include "tensor_holder.hpp"
#include "cpu_conv.hpp"
#include "tensor_layout.hpp"

enum tensor_layout_t
{
    miopen_tensor_layout_nchw,
    miopen_tensor_layout_ncdhw,
    miopen_tensor_layout_nhwc,
    miopen_tensor_layout_ndhwc,
};

std::string tensor_layout_to_string(tensor_layout_t layout)
{
    std::string layout_string("N/A");
    if(layout == miopen_tensor_layout_nchw)
        layout_string = "NCHW";
    else if(layout == miopen_tensor_layout_ncdhw)
        layout_string = "NCDHW";
    else if(layout == miopen_tensor_layout_nhwc)
        layout_string = "NHWC";
    else if(layout == miopen_tensor_layout_ndhwc)
        layout_string = "NDHWC";
    else
        MIOPEN_THROW("Unsupported tensor layout");
    return layout_string;
}

static int gen_rand_integer()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static int inited = 0;
    if(inited == 0)
    {
        std::srand(std::time(nullptr));
        inited = 1;
    }
    return std::rand();
}

struct gpu_reference_kernel_base
{
    miopenHandle_t handle{};
#if MIOPEN_BACKEND_OPENCL
    cl_command_queue q{};
#endif

    gpu_reference_kernel_base()
    {
        miopenCreate(&handle);
#if MIOPEN_BACKEND_OPENCL
        miopenGetStream(handle, &q);
#endif
    }
    ~gpu_reference_kernel_base() { miopenDestroy(handle); }

    static int conv_out_size(int in_size, int pad, int dilation, int ksize, int stride)
    {
        return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
    }

    static std::vector<int> get_image_depth() { return {8, 10}; }

    static std::vector<int> get_image_size() { return {9, 14}; }

    static std::vector<int> get_channel_size() { return {3, 8}; }

    static std::vector<int> get_filter_depth() { return {1, 3}; }

    static std::vector<int> get_filter_size() { return {1, 3}; }

    static std::vector<int> get_stride_depth() { return {1, 2}; }

    static std::vector<int> get_dilation_depth() { return {1}; }

    static std::vector<int> get_stride_dilation_size() { return {1, 2}; }

    static std::vector<int> get_pad_depth() { return {0, 1}; }

    static std::vector<int> get_pad_size() { return {0, 1}; }

    static std::vector<int> get_group_size() { return {1, 2}; }

    static std::vector<int> get_batch_size() { return {1, 2}; }

    template <typename F>
    void iterate_conv_2d(F f)
    {
        for(int c : get_channel_size())
        {
            for(int hi : get_image_size())
            {
                for(int wi : get_image_size())
                {
                    for(int fy : get_filter_size())
                    {
                        for(int fx : get_filter_size())
                        {
                            for(int py : get_pad_size())
                            {
                                for(int px : get_pad_size())
                                {
                                    int n  = get_batch_size()[gen_rand_integer() % 2];
                                    int g  = get_group_size()[gen_rand_integer() % 2];
                                    int k  = get_channel_size()[gen_rand_integer() % 2];
                                    int sy = get_stride_dilation_size()[gen_rand_integer() % 2];
                                    int sx = get_stride_dilation_size()[gen_rand_integer() % 2];
                                    int dy = get_stride_dilation_size()[gen_rand_integer() % 2];
                                    int dx = get_stride_dilation_size()[gen_rand_integer() % 2];
                                    int ho = conv_out_size(hi, py, dy, fy, sy);
                                    int wo = conv_out_size(wi, px, dx, fx, sx);

                                    if(fy > hi || fx > wi || (fy - 1) < py || (fx - 1) < px ||
                                       ho <= 0 || wo <= 0 || c % g != 0 || k % g != 0)
                                        continue;
                                    if((fx == 3 && fy == 5) || (fx == 5 && fy == 3))
                                        continue;
                                    f(n, wi, hi, c, k, fx, fy, px, py, sx, sy, dx, dy, g);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename F>
    void iterate_conv_3d(F f)
    {
        for(int c : get_channel_size())
        {
            for(int fy : get_filter_size())
            {
                for(int fx : get_filter_size())
                {
                    for(int py : get_pad_size())
                    {
                        for(int px : get_pad_size())
                        {
                            int n   = get_batch_size()[gen_rand_integer() % 2];
                            int g   = get_group_size()[gen_rand_integer() % 2];
                            int k   = get_channel_size()[gen_rand_integer() % 2];
                            int di  = get_image_depth()[gen_rand_integer() % 2];
                            int hi  = get_image_size()[gen_rand_integer() % 2];
                            int wi  = get_image_size()[gen_rand_integer() % 2];
                            int fz  = get_filter_depth()[gen_rand_integer() % 2];
                            int pz  = get_pad_depth()[gen_rand_integer() % 2];
                            int sx  = get_stride_dilation_size()[gen_rand_integer() % 2];
                            int sy  = get_stride_dilation_size()[gen_rand_integer() % 2];
                            int sz  = get_stride_depth()[gen_rand_integer() % 2];
                            int dx  = get_stride_dilation_size()[gen_rand_integer() % 2];
                            int dy  = get_stride_dilation_size()[gen_rand_integer() % 2];
                            int dz  = get_dilation_depth()[0];
                            int ho  = conv_out_size(hi, py, dy, fy, sy);
                            int wo  = conv_out_size(wi, px, dx, fx, sx);
                            int do_ = conv_out_size(di, pz, dz, fz, sz);
                            if(fy > hi || fx > wi || fz > di || (fy - 1) < py || (fx - 1) < px ||
                               (fz - 1) < pz || ho <= 0 || wo <= 0 || do_ <= 0 || c % g != 0 ||
                               k % g != 0)
                                continue;
                            if((fx == 3 && fy == 5) || (fx == 5 && fy == 3))
                                continue;
                            f(n,
                              di,
                              wi,
                              hi,
                              c,
                              k,
                              fz,
                              fx,
                              fy,
                              pz,
                              px,
                              py,
                              sz,
                              sx,
                              sy,
                              dz,
                              dx,
                              dy,
                              g);
                        }
                    }
                }
            }
        }
    }
};

#define RAND_INTEGER_MAX 5
#define RAND_INTEGER_MIN (-4)
#define MAX_INTEGER_INTERVAL 4.0

/*
* for half, if we use integer, half can express -2048 ~ 2048 without data-loss.
* e.g. 2049 can not expressed by half.
* from 2048~4096, half can only express 1/2 the number. number 2049, 2051, 2053, 2055.... can not be
* expressed. (max interval is 2)
* from 4096~8192, half can only express 1/4 the number. number 4097, 4098, 4099, 4101, 4102, 4103,
* 4105, 4106, 4107, 4109...
*               can not expressd. (max interval is 4)
* from 8192~16384, half can only express 1/8 the number. (max interval is 8)
*/
template <typename T>
void rand_tensor_integer(tensor<T>& t, int max = RAND_INTEGER_MAX, int min = RAND_INTEGER_MIN)
{
    // use integer to random.
    for(int i = 0; i < t.data.size(); i++)
        t[i]  = static_cast<T>(gen_rand_integer() % (max - min) + min);
}

template <typename T>
bool verify_tensor(tensor<T>& t_gpu,
                   tensor<T>& t_cpu,
                   float integer_interval = MAX_INTEGER_INTERVAL)
{
    if(t_gpu.data.size() != t_cpu.data.size())
    {
        MIOPEN_LOG_E("size not equal, should not happen");
        return false;
    }
    auto idx          = miopen::mismatch_idx(t_gpu.data, t_cpu.data, miopen::float_equal);
    bool valid_result = true;
    if(idx < miopen::range_distance(t_cpu))
    {
        // give a re-try chance for half_float
        // max gemm_k is wrw, max_n=2, max_ho/wo=14, max integer=4, max value=2*14*14*4*4 = 6272.
        // hence max integer interval is 4
        // for gpu we cast value to float, for cpu we cast value to double. hence inside kernel
        // precision is guaranteed.
        // the problem is cast the result back.
        // round-to-nearest(default rounding mode) seems will have little difference when doing
        // double->half, compare to float->half.
        // hence we give a chance to calculate if the difference is still following our experience,
        // while doing integer computation.
        auto max_diff = miopen::max_diff(t_gpu, t_cpu);
        if(max_diff > integer_interval)
            valid_result = false;
    }

    if(!valid_result)
    {
        std::cout << "diff at:" << idx << ", gpu:" << t_gpu[idx] << ", cpu:" << t_cpu[idx]
                  << std::endl;
    }
    return valid_result;
}

static std::string direction_to_string(miopen::conv::Direction direction)
{
    if(direction == miopen::conv::Direction::Forward)
        return "fwd";
    if(direction == miopen::conv::Direction::BackwardData)
        return "bwd";
    if(direction == miopen::conv::Direction::BackwardWeights)
        return "wrw";
    return "n/a";
}

static std::string miopen_type_to_string(miopenDataType_t type)
{
    if(type == miopenHalf)
        return "fp16";
    if(type == miopenFloat)
        return "fp32";
    if(type == miopenInt32)
        return "int32";
    if(type == miopenInt8)
        return "int8";
    if(type == miopenInt8x4)
        return "int8x4";
    if(type == miopenBFloat16)
        return "bf16";
    return "n/a";
}

template <miopen::conv::Direction direction, typename TRef, tensor_layout_t tensor_layout>
struct gpu_reference_conv_2d : gpu_reference_kernel_base
{
    void run()
    {
        auto run_conv_2d = [&](int n,
                               int wi,
                               int hi,
                               int c,
                               int k,
                               int fx,
                               int fy,
                               int px,
                               int py,
                               int sx,
                               int sy,
                               int dx,
                               int dy,
                               int g) {
            miopenConvolutionDescriptor_t convDesc;
            miopenTensorDescriptor_t inDesc, weiDesc, outDesc;

            int pads[]      = {py, px};
            int strides[]   = {sy, sx};
            int dilations[] = {dy, dx};
            int ho          = conv_out_size(hi, py, dy, fy, sy);
            int wo          = conv_out_size(wi, px, dx, fx, sx);
            int c_per_group = c / g;
            int k_per_group = k / g;

            int in_sz  = g * n * c_per_group * hi * wi;
            int wei_sz = g * k_per_group * c_per_group * fy * fx;
            int out_sz = g * n * k_per_group * ho * wo;

            std::vector<int> in_len({n, c, hi, wi});
            std::vector<int> wei_len({k, c_per_group, fy, fx});
            std::vector<int> out_len({n, k, ho, wo});

            std::vector<int> in_strides;
            std::vector<int> wei_strides;
            std::vector<int> out_strides;

            std::string layout_default = tensor_layout_get_default(4);
            std::string layout_string  = tensor_layout_to_string(tensor_layout);

            tensor_layout_to_strides(in_len, layout_default, layout_string, in_strides);
            tensor_layout_to_strides(wei_len, layout_default, layout_string, wei_strides);
            tensor_layout_to_strides(out_len, layout_default, layout_string, out_strides);

            tensor<TRef> in(in_len, in_strides);
            tensor<TRef> wei(wei_len, wei_strides);
            tensor<TRef> out(out_len, out_strides);
#if MIOPEN_BACKEND_OPENCL
            cl_context ctx;
            clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
            cl_int status = CL_SUCCESS;
            cl_mem in_dev =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(TRef) * in_sz, nullptr, &status);
            cl_mem wei_dev =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(TRef) * wei_sz, nullptr, nullptr);
            cl_mem out_dev =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(TRef) * out_sz, nullptr, nullptr);
            EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
            void* in_dev;
            void* wei_dev;
            void* out_dev;
            EXPECT(hipMalloc(&in_dev, sizeof(TRef) * in_sz) == hipSuccess);
            EXPECT(hipMalloc(&wei_dev, sizeof(TRef) * wei_sz) == hipSuccess);
            EXPECT(hipMalloc(&out_dev, sizeof(TRef) * out_sz) == hipSuccess);
#endif
            EXPECT(miopenCreateConvolutionDescriptor(&convDesc) == miopenStatusSuccess);
            EXPECT(miopenInitConvolutionNdDescriptor(convDesc,
                                                     2,
                                                     static_cast<int*>(pads),
                                                     static_cast<int*>(strides),
                                                     static_cast<int*>(dilations),
                                                     miopenConvolution) == miopenStatusSuccess);
            EXPECT(miopenSetConvolutionGroupCount(convDesc, g) == miopenStatusSuccess);

            EXPECT(miopenCreateTensorDescriptor(&inDesc) == miopenStatusSuccess);
            EXPECT(miopenCreateTensorDescriptor(&weiDesc) == miopenStatusSuccess);
            EXPECT(miopenCreateTensorDescriptor(&outDesc) == miopenStatusSuccess);

            EXPECT(
                miopenSetTensorDescriptor(
                    inDesc, miopen_type<TRef>{}, in_len.size(), in_len.data(), in_strides.data()) ==
                miopenStatusSuccess);
            EXPECT(miopenSetTensorDescriptor(weiDesc,
                                             miopen_type<TRef>{},
                                             wei_len.size(),
                                             wei_len.data(),
                                             wei_strides.data()) == miopenStatusSuccess);
            EXPECT(miopenSetTensorDescriptor(outDesc,
                                             miopen_type<TRef>{},
                                             out_len.size(),
                                             out_len.data(),
                                             out_strides.data()) == miopenStatusSuccess);

            bool valid_result = false;

            if(direction == miopen::conv::Direction::Forward)
            {
                // initialize data with integer
                rand_tensor_integer(in);
                rand_tensor_integer(wei);
#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueWriteBuffer(q,
                                              in_dev,
                                              CL_TRUE,
                                              0,
                                              sizeof(TRef) * in_sz,
                                              in.data.data(),
                                              0,
                                              nullptr,
                                              nullptr);
                status |= clEnqueueWriteBuffer(q,
                                               wei_dev,
                                               CL_TRUE,
                                               0,
                                               sizeof(TRef) * wei_sz,
                                               wei.data.data(),
                                               0,
                                               nullptr,
                                               nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(
                           in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice) ==
                       hipSuccess);
                EXPECT(hipMemcpy(wei_dev,
                                 wei.data.data(),
                                 sizeof(TRef) * wei_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
#endif
                cpu_convolution_forward(miopen::deref(convDesc).GetSpatialDimension(),
                                        in,
                                        wei,
                                        out,
                                        miopen::deref(convDesc).GetConvPads(),
                                        miopen::deref(convDesc).GetConvStrides(),
                                        miopen::deref(convDesc).GetConvDilations(),
                                        miopen::deref(convDesc).GetGroupCount());

                EXPECT(miopenConvolutionForwardImmediate(
                           handle,
                           weiDesc,
                           wei_dev,
                           inDesc,
                           in_dev,
                           convDesc,
                           outDesc,
                           out_dev,
                           nullptr,
                           0,
                           miopen::solver::Id("ConvDirectNaiveConvFwd").Value()) ==
                       miopenStatusSuccess);

                tensor<TRef> out_host(out_len, out_strides);
#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueReadBuffer(q,
                                             out_dev,
                                             CL_TRUE,
                                             0,
                                             sizeof(TRef) * out_sz,
                                             out_host.data.data(),
                                             0,
                                             nullptr,
                                             nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(out_host.data.data(),
                                 out_dev,
                                 sizeof(TRef) * out_sz,
                                 hipMemcpyDeviceToHost) == hipSuccess);
#endif
                // we expect excact match, since use integer
                valid_result = verify_tensor(out_host, out);
            }
            else if(direction == miopen::conv::Direction::BackwardData)
            {
                // initialize data with integer
                rand_tensor_integer(out);
                rand_tensor_integer(wei);
#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueWriteBuffer(q,
                                              out_dev,
                                              CL_TRUE,
                                              0,
                                              sizeof(TRef) * out_sz,
                                              out.data.data(),
                                              0,
                                              nullptr,
                                              nullptr);
                status |= clEnqueueWriteBuffer(q,
                                               wei_dev,
                                               CL_TRUE,
                                               0,
                                               sizeof(TRef) * wei_sz,
                                               wei.data.data(),
                                               0,
                                               nullptr,
                                               nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(out_dev,
                                 out.data.data(),
                                 sizeof(TRef) * out_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
                EXPECT(hipMemcpy(wei_dev,
                                 wei.data.data(),
                                 sizeof(TRef) * wei_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
#endif
                cpu_convolution_backward_data(miopen::deref(convDesc).GetSpatialDimension(),
                                              in,
                                              wei,
                                              out,
                                              miopen::deref(convDesc).GetConvPads(),
                                              miopen::deref(convDesc).GetConvStrides(),
                                              miopen::deref(convDesc).GetConvDilations(),
                                              miopen::deref(convDesc).GetGroupCount());

                EXPECT(miopenConvolutionBackwardDataImmediate(
                           handle,
                           outDesc,
                           out_dev,
                           weiDesc,
                           wei_dev,
                           convDesc,
                           inDesc,
                           in_dev,
                           nullptr,
                           0,
                           miopen::solver::Id("ConvDirectNaiveConvBwd").Value()) ==
                       miopenStatusSuccess);

                tensor<TRef> in_host(in_len, in_strides);
#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueReadBuffer(q,
                                             in_dev,
                                             CL_TRUE,
                                             0,
                                             sizeof(TRef) * in_sz,
                                             in_host.data.data(),
                                             0,
                                             nullptr,
                                             nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(in_host.data.data(),
                                 in_dev,
                                 sizeof(TRef) * in_sz,
                                 hipMemcpyDeviceToHost) == hipSuccess);
#endif

                // we expect excact match, since use integer
                valid_result = verify_tensor(in_host, in);
            }
            else if(direction == miopen::conv::Direction::BackwardWeights)
            {
                rand_tensor_integer(in);
                rand_tensor_integer(out);
#if MIOPEN_BACKEND_OPENCL
                status |= clEnqueueWriteBuffer(q,
                                               in_dev,
                                               CL_TRUE,
                                               0,
                                               sizeof(TRef) * in_sz,
                                               in.data.data(),
                                               0,
                                               nullptr,
                                               nullptr);
                status |= clEnqueueWriteBuffer(q,
                                               out_dev,
                                               CL_TRUE,
                                               0,
                                               sizeof(TRef) * out_sz,
                                               out.data.data(),
                                               0,
                                               nullptr,
                                               nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(
                           in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice) ==
                       hipSuccess);
                EXPECT(hipMemcpy(out_dev,
                                 out.data.data(),
                                 sizeof(TRef) * out_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
#endif
                cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                                in,
                                                wei,
                                                out,
                                                miopen::deref(convDesc).GetConvPads(),
                                                miopen::deref(convDesc).GetConvStrides(),
                                                miopen::deref(convDesc).GetConvDilations(),
                                                miopen::deref(convDesc).GetGroupCount());

                EXPECT(miopenConvolutionBackwardWeightsImmediate(
                           handle,
                           outDesc,
                           out_dev,
                           inDesc,
                           in_dev,
                           convDesc,
                           weiDesc,
                           wei_dev,
                           nullptr,
                           0,
                           miopen::solver::Id("ConvDirectNaiveConvWrw").Value()) ==
                       miopenStatusSuccess);

                tensor<TRef> wei_host(wei_len, wei_strides);
#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueReadBuffer(q,
                                             wei_dev,
                                             CL_TRUE,
                                             0,
                                             sizeof(TRef) * wei_sz,
                                             wei_host.data.data(),
                                             0,
                                             nullptr,
                                             nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(wei_host.data.data(),
                                 wei_dev,
                                 sizeof(TRef) * wei_sz,
                                 hipMemcpyDeviceToHost) == hipSuccess);
#endif

                // we expect excact match, since use integer
                valid_result = verify_tensor(wei_host, wei);
            }

            // auto error        = miopen::rms_range(out_host.data, out.data);
            // auto tolerance = get_default_tolerence<TRef>();
            // bool valid_result = error <= tolerance;
            std::cout << "n:" << n << ", c:" << c << ", hi:" << hi << ", wi:" << wi << ", k:" << k
                      << ", ho:" << ho << ", wo:" << wo << ", fy:" << fy << ",fx:" << fx
                      << ", py:" << py << ", px:" << px << ", sy:" << sy << ", sx:" << sx
                      << ", dy:" << dy << ",dx:" << dx << ", g:" << g
                      << ", dir:" << direction_to_string(direction)
                      << ", type:" << miopen_type_to_string(miopen_type<TRef>{})
                      << ", layout:" << layout_string << ", valid:" << valid_result << std::endl;
            EXPECT(valid_result == true);

            miopenDestroyConvolutionDescriptor(convDesc);
            miopenDestroyTensorDescriptor(inDesc);
            miopenDestroyTensorDescriptor(weiDesc);
            miopenDestroyTensorDescriptor(outDesc);
#if MIOPEN_BACKEND_OPENCL
            clReleaseMemObject(in_dev);
            clReleaseMemObject(wei_dev);
            clReleaseMemObject(out_dev);
#elif MIOPEN_BACKEND_HIP
            hipFree(in_dev);
            hipFree(wei_dev);
            hipFree(out_dev);
#endif
        };

        iterate_conv_2d(run_conv_2d);
    }
};

template <miopen::conv::Direction direction, typename TRef, tensor_layout_t tensor_layout>
struct gpu_reference_conv_3d : gpu_reference_kernel_base
{
    void run()
    {
        auto run_conv_3d = [&](int n,
                               int di,
                               int wi,
                               int hi,
                               int c,
                               int k,
                               int fz,
                               int fx,
                               int fy,
                               int pz,
                               int px,
                               int py,
                               int sz,
                               int sx,
                               int sy,
                               int dz,
                               int dx,
                               int dy,
                               int g) {
            miopenConvolutionDescriptor_t convDesc;
            miopenTensorDescriptor_t inDesc, weiDesc, outDesc;

            int pads[]      = {pz, py, px};
            int strides[]   = {sz, sy, sx};
            int dilations[] = {dz, dy, dx};
            int ho          = conv_out_size(hi, py, dy, fy, sy);
            int wo          = conv_out_size(wi, px, dx, fx, sx);
            int do_         = conv_out_size(di, pz, dz, fz, sz);
            int c_per_group = c / g;
            int k_per_group = k / g;

            int in_sz  = g * n * c_per_group * di * hi * wi;
            int wei_sz = g * k_per_group * c_per_group * fz * fy * fx;
            int out_sz = g * n * k_per_group * do_ * ho * wo;

            std::vector<int> in_len({n, c, di, hi, wi});
            std::vector<int> wei_len({k, c_per_group, fz, fy, fx});
            std::vector<int> out_len({n, k, do_, ho, wo});

            std::vector<int> in_strides;
            std::vector<int> wei_strides;
            std::vector<int> out_strides;

            std::string layout_default = tensor_layout_get_default(5);
            std::string layout_string  = tensor_layout_to_string(tensor_layout);

            tensor_layout_to_strides(in_len, layout_default, layout_string, in_strides);
            tensor_layout_to_strides(wei_len, layout_default, layout_string, wei_strides);
            tensor_layout_to_strides(out_len, layout_default, layout_string, out_strides);

            tensor<TRef> in(in_len, in_strides);
            tensor<TRef> wei(wei_len, wei_strides);
            tensor<TRef> out(out_len, out_strides);
#if MIOPEN_BACKEND_OPENCL
            cl_context ctx;
            clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
            cl_int status = CL_SUCCESS;
            cl_mem in_dev =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(TRef) * in_sz, nullptr, &status);
            cl_mem wei_dev =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(TRef) * wei_sz, nullptr, nullptr);
            cl_mem out_dev =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(TRef) * out_sz, nullptr, nullptr);
            EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
            void* in_dev;
            void* wei_dev;
            void* out_dev;
            EXPECT(hipMalloc(&in_dev, sizeof(TRef) * in_sz) == hipSuccess);
            EXPECT(hipMalloc(&wei_dev, sizeof(TRef) * wei_sz) == hipSuccess);
            EXPECT(hipMalloc(&out_dev, sizeof(TRef) * out_sz) == hipSuccess);
#endif
            EXPECT(miopenCreateConvolutionDescriptor(&convDesc) == miopenStatusSuccess);
            EXPECT(miopenInitConvolutionNdDescriptor(convDesc,
                                                     3,
                                                     static_cast<int*>(pads),
                                                     static_cast<int*>(strides),
                                                     static_cast<int*>(dilations),
                                                     miopenConvolution) == miopenStatusSuccess);
            EXPECT(miopenSetConvolutionGroupCount(convDesc, g) == miopenStatusSuccess);

            EXPECT(miopenCreateTensorDescriptor(&inDesc) == miopenStatusSuccess);
            EXPECT(miopenCreateTensorDescriptor(&weiDesc) == miopenStatusSuccess);
            EXPECT(miopenCreateTensorDescriptor(&outDesc) == miopenStatusSuccess);

            EXPECT(
                miopenSetTensorDescriptor(
                    inDesc, miopen_type<TRef>{}, in_len.size(), in_len.data(), in_strides.data()) ==
                miopenStatusSuccess);
            EXPECT(miopenSetTensorDescriptor(weiDesc,
                                             miopen_type<TRef>{},
                                             wei_len.size(),
                                             wei_len.data(),
                                             wei_strides.data()) == miopenStatusSuccess);
            EXPECT(miopenSetTensorDescriptor(outDesc,
                                             miopen_type<TRef>{},
                                             out_len.size(),
                                             out_len.data(),
                                             out_strides.data()) == miopenStatusSuccess);

            bool valid_result = false;

            if(direction == miopen::conv::Direction::Forward)
            {
                // initialize data with integer
                rand_tensor_integer(in);
                rand_tensor_integer(wei);
#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueWriteBuffer(q,
                                              in_dev,
                                              CL_TRUE,
                                              0,
                                              sizeof(TRef) * in_sz,
                                              in.data.data(),
                                              0,
                                              nullptr,
                                              nullptr);
                status |= clEnqueueWriteBuffer(q,
                                               wei_dev,
                                               CL_TRUE,
                                               0,
                                               sizeof(TRef) * wei_sz,
                                               wei.data.data(),
                                               0,
                                               nullptr,
                                               nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(
                           in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice) ==
                       hipSuccess);
                EXPECT(hipMemcpy(wei_dev,
                                 wei.data.data(),
                                 sizeof(TRef) * wei_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
#endif

                cpu_convolution_forward(miopen::deref(convDesc).GetSpatialDimension(),
                                        in,
                                        wei,
                                        out,
                                        miopen::deref(convDesc).GetConvPads(),
                                        miopen::deref(convDesc).GetConvStrides(),
                                        miopen::deref(convDesc).GetConvDilations(),
                                        miopen::deref(convDesc).GetGroupCount());

                EXPECT(miopenConvolutionForwardImmediate(
                           handle,
                           weiDesc,
                           wei_dev,
                           inDesc,
                           in_dev,
                           convDesc,
                           outDesc,
                           out_dev,
                           nullptr,
                           0,
                           miopen::solver::Id("ConvDirectNaiveConvFwd").Value()) ==
                       miopenStatusSuccess);

                tensor<TRef> out_host(out_len, out_strides);
#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueReadBuffer(q,
                                             out_dev,
                                             CL_TRUE,
                                             0,
                                             sizeof(TRef) * out_sz,
                                             out_host.data.data(),
                                             0,
                                             nullptr,
                                             nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(out_host.data.data(),
                                 out_dev,
                                 sizeof(TRef) * out_sz,
                                 hipMemcpyDeviceToHost) == hipSuccess);
#endif

                // we expect excact match, since use integer
                valid_result = verify_tensor(out_host, out);
            }
            else if(direction == miopen::conv::Direction::BackwardData)
            {
                // initialize data with integer
                rand_tensor_integer(out);
                rand_tensor_integer(wei);
#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueWriteBuffer(q,
                                              out_dev,
                                              CL_TRUE,
                                              0,
                                              sizeof(TRef) * out_sz,
                                              out.data.data(),
                                              0,
                                              nullptr,
                                              nullptr);
                status |= clEnqueueWriteBuffer(q,
                                               wei_dev,
                                               CL_TRUE,
                                               0,
                                               sizeof(TRef) * wei_sz,
                                               wei.data.data(),
                                               0,
                                               nullptr,
                                               nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(out_dev,
                                 out.data.data(),
                                 sizeof(TRef) * out_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
                EXPECT(hipMemcpy(wei_dev,
                                 wei.data.data(),
                                 sizeof(TRef) * wei_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
#endif
                cpu_convolution_backward_data(miopen::deref(convDesc).GetSpatialDimension(),
                                              in,
                                              wei,
                                              out,
                                              miopen::deref(convDesc).GetConvPads(),
                                              miopen::deref(convDesc).GetConvStrides(),
                                              miopen::deref(convDesc).GetConvDilations(),
                                              miopen::deref(convDesc).GetGroupCount());

                EXPECT(miopenConvolutionBackwardDataImmediate(
                           handle,
                           outDesc,
                           out_dev,
                           weiDesc,
                           wei_dev,
                           convDesc,
                           inDesc,
                           in_dev,
                           nullptr,
                           0,
                           miopen::solver::Id("ConvDirectNaiveConvBwd").Value()) ==
                       miopenStatusSuccess);

                tensor<TRef> in_host(in_len, in_strides);
#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueReadBuffer(q,
                                             in_dev,
                                             CL_TRUE,
                                             0,
                                             sizeof(TRef) * in_sz,
                                             in_host.data.data(),
                                             0,
                                             nullptr,
                                             nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(in_host.data.data(),
                                 in_dev,
                                 sizeof(TRef) * in_sz,
                                 hipMemcpyDeviceToHost) == hipSuccess);
#endif

                // we expect excact match, since use integer
                valid_result = verify_tensor(in_host, in);
            }
            else if(direction == miopen::conv::Direction::BackwardWeights)
            {
                rand_tensor_integer(in, 3, -2);
                rand_tensor_integer(out, 3, -2);
#if MIOPEN_BACKEND_OPENCL
                status |= clEnqueueWriteBuffer(q,
                                               in_dev,
                                               CL_TRUE,
                                               0,
                                               sizeof(TRef) * in_sz,
                                               in.data.data(),
                                               0,
                                               nullptr,
                                               nullptr);
                status |= clEnqueueWriteBuffer(q,
                                               out_dev,
                                               CL_TRUE,
                                               0,
                                               sizeof(TRef) * out_sz,
                                               out.data.data(),
                                               0,
                                               nullptr,
                                               nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(
                           in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice) ==
                       hipSuccess);
                EXPECT(hipMemcpy(out_dev,
                                 out.data.data(),
                                 sizeof(TRef) * out_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
#endif
                cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                                in,
                                                wei,
                                                out,
                                                miopen::deref(convDesc).GetConvPads(),
                                                miopen::deref(convDesc).GetConvStrides(),
                                                miopen::deref(convDesc).GetConvDilations(),
                                                miopen::deref(convDesc).GetGroupCount());

                EXPECT(miopenConvolutionBackwardWeightsImmediate(
                           handle,
                           outDesc,
                           out_dev,
                           inDesc,
                           in_dev,
                           convDesc,
                           weiDesc,
                           wei_dev,
                           nullptr,
                           0,
                           miopen::solver::Id("ConvDirectNaiveConvWrw").Value()) ==
                       miopenStatusSuccess);

                tensor<TRef> wei_host(wei_len, wei_strides);
#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueReadBuffer(q,
                                             wei_dev,
                                             CL_TRUE,
                                             0,
                                             sizeof(TRef) * wei_sz,
                                             wei_host.data.data(),
                                             0,
                                             nullptr,
                                             nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(wei_host.data.data(),
                                 wei_dev,
                                 sizeof(TRef) * wei_sz,
                                 hipMemcpyDeviceToHost) == hipSuccess);
#endif

                // we expect excact match, since use integer
                valid_result = verify_tensor(wei_host, wei, 8.0); // max possible int
                                                                  // 2*14*14*10*(2*2) = 15680, hence
                                                                  // int interval might be 8
            }

            // auto error        = miopen::rms_range(out_host.data, out.data);
            // auto tolerance = get_default_tolerence<TRef>();
            // bool valid_result = error <= tolerance;
            std::cout << "n:" << n << ", c:" << c << ", di:" << di << ", hi:" << hi << ", wi:" << wi
                      << ", k:" << k << ", do:" << do_ << ", ho:" << ho << ", wo:" << wo
                      << ", fz:" << fz << ", fy:" << fy << ",fx:" << fx << ", pz:" << pz
                      << ", py:" << py << ", px:" << px << ", sz:" << sz << ", sy:" << sy
                      << ", sx:" << sx << ", dz:" << dz << ", dy:" << dy << ", dx:" << dx
                      << ", g:" << g << ", dir:" << direction_to_string(direction)
                      << ", type:" << miopen_type_to_string(miopen_type<TRef>{})
                      << ", layout:" << layout_string << ", valid:" << valid_result << std::endl;
            EXPECT(valid_result == true);

            miopenDestroyConvolutionDescriptor(convDesc);
            miopenDestroyTensorDescriptor(inDesc);
            miopenDestroyTensorDescriptor(weiDesc);
            miopenDestroyTensorDescriptor(outDesc);

#if MIOPEN_BACKEND_OPENCL
            clReleaseMemObject(in_dev);
            clReleaseMemObject(wei_dev);
            clReleaseMemObject(out_dev);
#elif MIOPEN_BACKEND_HIP
            hipFree(in_dev);
            hipFree(wei_dev);
            hipFree(out_dev);
#endif
        };

        iterate_conv_3d(run_conv_3d);
    }
};

int main()
{
    // 2d NCHW
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::Forward,
                                   float,
                                   miopen_tensor_layout_nchw>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::Forward,
                                   half_float::half,
                                   miopen_tensor_layout_nchw>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::Forward,
                                   bfloat16,
                                   miopen_tensor_layout_nchw>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardData,
                                   float,
                                   miopen_tensor_layout_nchw>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardData,
                                   half_float::half,
                                   miopen_tensor_layout_nchw>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardData,
                                   bfloat16,
                                   miopen_tensor_layout_nchw>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights,
                                   float,
                                   miopen_tensor_layout_nchw>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights,
                                   half_float::half,
                                   miopen_tensor_layout_nchw>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights,
                                   bfloat16,
                                   miopen_tensor_layout_nchw>>();

    // 3d NCDHW
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
                                   float,
                                   miopen_tensor_layout_ncdhw>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
                                   half_float::half,
                                   miopen_tensor_layout_ncdhw>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
                                   bfloat16,
                                   miopen_tensor_layout_ncdhw>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
                                   float,
                                   miopen_tensor_layout_ncdhw>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
                                   half_float::half,
                                   miopen_tensor_layout_ncdhw>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
                                   bfloat16,
                                   miopen_tensor_layout_ncdhw>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                                   float,
                                   miopen_tensor_layout_ncdhw>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                                   half_float::half,
                                   miopen_tensor_layout_ncdhw>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                                   bfloat16,
                                   miopen_tensor_layout_ncdhw>>();

    // 2d NHWC
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::Forward,
                                   float,
                                   miopen_tensor_layout_nhwc>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::Forward,
                                   half_float::half,
                                   miopen_tensor_layout_nhwc>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::Forward,
                                   bfloat16,
                                   miopen_tensor_layout_nhwc>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardData,
                                   float,
                                   miopen_tensor_layout_nhwc>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardData,
                                   half_float::half,
                                   miopen_tensor_layout_nhwc>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardData,
                                   bfloat16,
                                   miopen_tensor_layout_nhwc>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights,
                                   float,
                                   miopen_tensor_layout_nhwc>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights,
                                   half_float::half,
                                   miopen_tensor_layout_nhwc>>();
    run_test<gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights,
                                   bfloat16,
                                   miopen_tensor_layout_nhwc>>();

    // 3d NDHWC
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
                                   float,
                                   miopen_tensor_layout_ndhwc>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
                                   half_float::half,
                                   miopen_tensor_layout_ndhwc>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
                                   bfloat16,
                                   miopen_tensor_layout_ndhwc>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
                                   float,
                                   miopen_tensor_layout_ndhwc>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
                                   half_float::half,
                                   miopen_tensor_layout_ndhwc>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
                                   bfloat16,
                                   miopen_tensor_layout_ndhwc>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                                   float,
                                   miopen_tensor_layout_ndhwc>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                                   half_float::half,
                                   miopen_tensor_layout_ndhwc>>();
    run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                                   bfloat16,
                                   miopen_tensor_layout_ndhwc>>();
}
