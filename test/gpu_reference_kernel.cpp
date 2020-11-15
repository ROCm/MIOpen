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
#include <miopen/gpu_reference_kernel.hpp>
#include <miopen/miopen.h>
#include <miopen/convolution.hpp>
#include <miopen/problem_description.hpp>
#include <miopen/tensor.hpp>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <type_traits>
#include <half.hpp>
#include "test.hpp"
#include "driver.hpp"
#include "tensor_holder.hpp"
#include "cpu_conv.hpp"

struct gpu_reference_kernel_base
{
    miopenHandle_t handle{};
    gpu_reference_kernel_base() { miopenCreate(&handle); }
    ~gpu_reference_kernel_base() { miopenDestroy(handle); }

    static int conv_out_size(int in_size, int pad, int dilation, int ksize, int stride)
    {
        return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
    }

    static std::vector<int> get_image_size() { return {9, 14}; }

    static std::vector<int> get_channel_size() { return {3, 8}; }

    static std::vector<int> get_filter_size() { return {1, 3}; }

    static std::vector<int> get_stride_dilation_size() { return {1, 2}; }

    static std::vector<int> get_pad_size() { return {0, 1}; }

    static std::vector<int> get_group_size() { return {2}; }

    static std::vector<int> get_batch_size() { return {2}; }

    template <typename F>
    void iterate_conv_2d(F f)
    {
        for(int n : get_batch_size())
        {
            for(int g : get_group_size())
            {
                for(int c : get_channel_size())
                {
                    for(int hi : get_image_size())
                    {
                        for(int wi : get_image_size())
                        {
                            for(int k : get_channel_size())
                            {
                                for(int fy : get_filter_size())
                                {
                                    for(int fx : get_filter_size())
                                    {
                                        for(int py : get_pad_size())
                                        {
                                            for(int px : get_pad_size())
                                            {
                                                for(int sy : get_stride_dilation_size())
                                                {
                                                    for(int sx : get_stride_dilation_size())
                                                    {
                                                        for(int dy : get_stride_dilation_size())
                                                        {
                                                            for(int dx : get_stride_dilation_size())
                                                            {
                                                                int ho = conv_out_size(
                                                                    hi, py, dy, fy, sy);
                                                                int wo = conv_out_size(
                                                                    wi, px, dx, fx, sx);
                                                                if(fy > hi || fx > wi ||
                                                                   (fy - 1) < py || (fx - 1) < px ||
                                                                   ho <= 0 || wo <= 0 ||
                                                                   c % g != 0 || k % g != 0)
                                                                    continue;
                                                                if((fx == 3 && fy == 5) ||
                                                                   (fx == 5 && fy == 3))
                                                                    continue;
                                                                f(n,
                                                                  wi,
                                                                  hi,
                                                                  c,
                                                                  k,
                                                                  fx,
                                                                  fy,
                                                                  px,
                                                                  py,
                                                                  sx,
                                                                  sy,
                                                                  dx,
                                                                  dy,
                                                                  g);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

#define RAND_INTEGER_MAX 5
#define RAND_INTEGER_MIN -4
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
void rand_tensor_integer(tensor<T>& t)
{
    // use integer to random.
    static int inited = 0;
    if(!inited)
    {
        std::srand(std::time(nullptr));
        inited = 1;
    }
    for(int i = 0; i < t.data.size(); i++)
        t[i] =
            static_cast<T>(std::rand() % (RAND_INTEGER_MAX - RAND_INTEGER_MIN) + RAND_INTEGER_MIN);
}

template <typename T>
bool verify_tensor(tensor<T>& t_gpu, tensor<T>& t_cpu)
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
        if(max_diff > MAX_INTEGER_INTERVAL)
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

template <miopen::conv::Direction direction, typename TRef>
struct gpu_reference_conv_nchw : gpu_reference_kernel_base
{
    void run()
    {
        auto run_fwd = [&](int n,
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

            void* in_dev;
            void* wei_dev;
            void* out_dev;

            tensor<TRef> in(n, c, hi, wi);
            tensor<TRef> wei(k, c / g, fy, fx); // refer to ConvDriver<Tgpu,
                                                // Tref>::GetWeightTensorLengthsFromCmdLine() to
                                                // deal with group_count
            tensor<TRef> out(n, k, ho, wo);

            EXPECT(hipMalloc(&in_dev, sizeof(TRef) * in_sz) == hipSuccess);
            EXPECT(hipMalloc(&wei_dev, sizeof(TRef) * wei_sz) == hipSuccess);
            EXPECT(hipMalloc(&out_dev, sizeof(TRef) * out_sz) == hipSuccess);
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
            EXPECT(miopenSet4dTensorDescriptor(inDesc, miopen_type<TRef>{}, n, c, hi, wi) ==
                   miopenStatusSuccess);
            EXPECT(miopenSet4dTensorDescriptor(weiDesc, miopen_type<TRef>{}, k, c, fy, fx) ==
                   miopenStatusSuccess);
            EXPECT(miopenSet4dTensorDescriptor(outDesc, miopen_type<TRef>{}, n, k, ho, wo) ==
                   miopenStatusSuccess);

            bool valid_result = false;

            if(direction == miopen::conv::Direction::Forward)
            {
                // initialize data with integer
                rand_tensor_integer(in);
                rand_tensor_integer(wei);
                EXPECT(hipMemcpy(
                           in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice) ==
                       hipSuccess);
                EXPECT(hipMemcpy(wei_dev,
                                 wei.data.data(),
                                 sizeof(TRef) * wei_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);

                cpu_convolution_forward(miopen::deref(convDesc).GetSpatialDimension(),
                                        in,
                                        wei,
                                        out,
                                        miopen::deref(convDesc).GetConvPads(),
                                        miopen::deref(convDesc).GetConvStrides(),
                                        miopen::deref(convDesc).GetConvDilations(),
                                        miopen::deref(convDesc).GetGroupCount());

                const auto problem = miopen::ProblemDescription{in.desc,
                                                                wei.desc,
                                                                out.desc,
                                                                miopen::deref(convDesc),
                                                                miopen::conv::Direction::Forward};
                GPUReferenceConvolutionForward(
                    miopen::deref(handle), problem, in_dev, wei_dev, out_dev);

                tensor<TRef> out_host(n, k, ho, wo);
                EXPECT(hipMemcpy(out_host.data.data(),
                                 out_dev,
                                 sizeof(TRef) * out_sz,
                                 hipMemcpyDeviceToHost) == hipSuccess);

                // we expect excact match, since use integer
                valid_result = verify_tensor(out_host, out);
            }
            else if(direction == miopen::conv::Direction::BackwardData)
            {
                // initialize data with integer
                rand_tensor_integer(out);
                rand_tensor_integer(wei);
                EXPECT(hipMemcpy(out_dev,
                                 out.data.data(),
                                 sizeof(TRef) * out_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
                EXPECT(hipMemcpy(wei_dev,
                                 wei.data.data(),
                                 sizeof(TRef) * wei_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
                cpu_convolution_backward_data(miopen::deref(convDesc).GetSpatialDimension(),
                                              in,
                                              wei,
                                              out,
                                              miopen::deref(convDesc).GetConvPads(),
                                              miopen::deref(convDesc).GetConvStrides(),
                                              miopen::deref(convDesc).GetConvDilations(),
                                              miopen::deref(convDesc).GetGroupCount());

                const auto problem =
                    miopen::ProblemDescription{in.desc,
                                               wei.desc,
                                               out.desc,
                                               miopen::deref(convDesc),
                                               miopen::conv::Direction::BackwardData};
                GPUReferenceConvolutionBackwardData(
                    miopen::deref(handle), problem, in_dev, wei_dev, out_dev);

                tensor<TRef> in_host(n, c, hi, wi);
                EXPECT(hipMemcpy(in_host.data.data(),
                                 in_dev,
                                 sizeof(TRef) * in_sz,
                                 hipMemcpyDeviceToHost) == hipSuccess);

                // we expect excact match, since use integer
                valid_result = verify_tensor(in_host, in);
            }
            else if(direction == miopen::conv::Direction::BackwardWeights)
            {
                rand_tensor_integer(in);
                rand_tensor_integer(out);
                EXPECT(hipMemcpy(
                           in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice) ==
                       hipSuccess);
                EXPECT(hipMemcpy(out_dev,
                                 out.data.data(),
                                 sizeof(TRef) * out_sz,
                                 hipMemcpyHostToDevice) == hipSuccess);
                cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                                in,
                                                wei,
                                                out,
                                                miopen::deref(convDesc).GetConvPads(),
                                                miopen::deref(convDesc).GetConvStrides(),
                                                miopen::deref(convDesc).GetConvDilations(),
                                                miopen::deref(convDesc).GetGroupCount());

                const auto problem =
                    miopen::ProblemDescription{in.desc,
                                               wei.desc,
                                               out.desc,
                                               miopen::deref(convDesc),
                                               miopen::conv::Direction::BackwardWeights};
                GPUReferenceConvolutionBackwardWeights(
                    miopen::deref(handle), problem, in_dev, wei_dev, out_dev);

                tensor<TRef> wei_host(k, c / g, fy, fx);
                EXPECT(hipMemcpy(wei_host.data.data(),
                                 wei_dev,
                                 sizeof(TRef) * wei_sz,
                                 hipMemcpyDeviceToHost) == hipSuccess);

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
                      << ", valid:" << valid_result << std::endl;
            EXPECT(valid_result == true);

            miopenDestroyConvolutionDescriptor(convDesc);
            miopenDestroyTensorDescriptor(inDesc);
            miopenDestroyTensorDescriptor(weiDesc);
            miopenDestroyTensorDescriptor(outDesc);

            hipFree(in_dev);
            hipFree(wei_dev);
            hipFree(out_dev);
        };

        iterate_conv_2d(run_fwd);
    }
};

int main()
{
    run_test<gpu_reference_conv_nchw<miopen::conv::Direction::Forward, float>>();
    run_test<gpu_reference_conv_nchw<miopen::conv::Direction::Forward, half_float::half>>();
    run_test<gpu_reference_conv_nchw<miopen::conv::Direction::BackwardData, float>>();
    run_test<gpu_reference_conv_nchw<miopen::conv::Direction::BackwardData, half_float::half>>();
    run_test<gpu_reference_conv_nchw<miopen::conv::Direction::BackwardWeights, float>>();
    run_test<gpu_reference_conv_nchw<miopen::conv::Direction::BackwardWeights, half_float::half>>();
}
