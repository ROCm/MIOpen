/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#define GUARD_TEST_TEST_HPP_

template <class T>
void run_test()
{
}
#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include <gtest/gtest_common.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>

#include <cstdlib>
#include <ctime>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

#include <miopen/convolution.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>
#include <miopen/bfloat16.hpp>
#include <half/half.hpp>
#include "driver.hpp"
#include "tensor_holder.hpp"
#include "cpu_conv.hpp"

using float16 = half_float::half;

class ReferenceConvBase : public ::testing::Test
{
protected:
    miopenHandle_t handle{};

    virtual void SetUp() override { miopenCreate(&this->handle); }

    virtual void TearDown() override { miopenDestroy(this->handle); }

    static int conv_out_size(int in_size, int pad, int dilation, int ksize, int stride)
    {
        return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
    }

    static std::vector<int> get_image_depth() { return {8, 10}; }

    static std::vector<int> get_image_size() { return {9, 14}; }

    // Warning: Channel size must be multiple of group size
    static std::vector<int> get_channel_size() { return {4, 8}; }

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
                                    int n = get_batch_size()[prng::gen_canonical<size_t>()];
                                    int g = get_group_size()[prng::gen_canonical<size_t>()];
                                    int k = get_channel_size()[prng::gen_canonical<size_t>()];
                                    int sy =
                                        get_stride_dilation_size()[prng::gen_canonical<size_t>()];
                                    int sx =
                                        get_stride_dilation_size()[prng::gen_canonical<size_t>()];
                                    int dy =
                                        get_stride_dilation_size()[prng::gen_canonical<size_t>()];
                                    int dx =
                                        get_stride_dilation_size()[prng::gen_canonical<size_t>()];
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
                            int n   = get_batch_size()[prng::gen_canonical<size_t>()];
                            int g   = get_group_size()[prng::gen_canonical<size_t>()];
                            int k   = get_channel_size()[prng::gen_canonical<size_t>()];
                            int di  = get_image_depth()[prng::gen_canonical<size_t>()];
                            int hi  = get_image_size()[prng::gen_canonical<size_t>()];
                            int wi  = get_image_size()[prng::gen_canonical<size_t>()];
                            int fz  = get_filter_depth()[prng::gen_canonical<size_t>()];
                            int pz  = get_pad_depth()[prng::gen_canonical<size_t>()];
                            int sx  = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
                            int sy  = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
                            int sz  = get_stride_depth()[prng::gen_canonical<size_t>()];
                            int dx  = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
                            int dy  = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
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

static constexpr int RAND_INTEGER_MAX       = 5;
static constexpr int RAND_INTEGER_MIN       = -4;
static constexpr float MAX_INTEGER_INTERVAL = 4.f;

/*
 * for half, if we use integer, half can express -2048 ~ 2048 without data-loss.
 * e.g. 2049 can not expressed by half.
 * from 2048~4096, half can only express 1/2 the number. number 2049, 2051, 2053, 2055.... can not
 * be expressed. (max interval is 2) from 4096~8192, half can only express 1/4 the number. number
 * 4097, 4098, 4099, 4101, 4102, 4103, 4105, 4106, 4107, 4109... can not expressd. (max interval is
 * 4) from 8192~16384, half can only express 1/8 the number. (max interval is 8)
 */
template <typename T>
void rand_tensor_integer(tensor<T>& t, int max = RAND_INTEGER_MAX, int min = RAND_INTEGER_MIN)
{
    // use integer to random.
    for(size_t i = 0; i < t.data.size(); i++)
        t[i] = static_cast<T>(prng::gen_A_to_B(min, max));
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
    if(type == miopenBFloat16)
        return "bf16";
    return "n/a";
}

/// input: a vector of lengths of dims in a tensor
/// multiply each element with a random constant integer
void pad_tensor_strides(std::vector<int>& strides)
{
    constexpr int min_stride_multiplier = 1;
    constexpr int max_stride_multiplier = 5;

    auto c = prng::gen_A_to_B(min_stride_multiplier, max_stride_multiplier);
    for(auto& v : strides)
    {
        // cppcheck-suppress useStlAlgorithm
        v = v * c;
    }
}

template <miopen::conv::Direction direction,
          typename TREF,
          typename TOUT,
          miopenTensorLayout_t tensor_layout>
struct TypeDefs
{
    static constexpr miopen::conv::Direction get_direction() { return direction; };
    typedef TREF TRef;
    typedef TOUT Tout;
    static constexpr miopenTensorLayout_t get_tensor_layout() { return tensor_layout; };
};

template <class Types>
class ReferenceConv2d : public ReferenceConvBase
{
protected:
    template <miopen::conv::Direction direction,
              typename TRef,
              typename Tout,
              miopenTensorLayout_t tensor_layout>
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
            int ho          = this->conv_out_size(hi, py, dy, fy, sy);
            int wo          = this->conv_out_size(wi, px, dx, fx, sx);
            int c_per_group = c / g;

            std::vector<int> in_len({n, c, hi, wi});
            std::vector<int> wei_len({k, c_per_group, fy, fx});
            std::vector<int> out_len({n, k, ho, wo});

            std::vector<int> in_strides;
            std::vector<int> wei_strides;
            std::vector<int> out_strides;

            std::string layout_default = miopen::tensor_layout_get_default(4);
            std::string layout_string  = miopen::TensorDescriptor::GetLayoutStr(tensor_layout);

            miopen::tensor_layout_to_strides(in_len, layout_default, layout_string, in_strides);
            miopen::tensor_layout_to_strides(wei_len, layout_default, layout_string, wei_strides);
            miopen::tensor_layout_to_strides(out_len, layout_default, layout_string, out_strides);

            pad_tensor_strides(in_strides);
            pad_tensor_strides(wei_strides);
            pad_tensor_strides(out_strides);

            tensor<TRef> in(in_len, in_strides);
            tensor<TRef> wei(wei_len, wei_strides);
            tensor<Tout> out(out_len, out_strides);

            auto in_sz  = in.data.size();
            auto wei_sz = wei.data.size();
            auto out_sz = out.data.size();

            void* in_dev;
            void* wei_dev;
            void* out_dev;
            EXPECT_TRUE(hipMalloc(&in_dev, sizeof(TRef) * in_sz) == hipSuccess);
            EXPECT_TRUE(hipMalloc(&wei_dev, sizeof(TRef) * wei_sz) == hipSuccess);
            EXPECT_TRUE(hipMalloc(&out_dev, sizeof(Tout) * out_sz) == hipSuccess);

            EXPECT_TRUE(miopenCreateConvolutionDescriptor(&convDesc) == miopenStatusSuccess);
            EXPECT_TRUE(miopenInitConvolutionNdDescriptor(convDesc,
                                                          2,
                                                          static_cast<int*>(pads),
                                                          static_cast<int*>(strides),
                                                          static_cast<int*>(dilations),
                                                          miopenConvolution) ==
                        miopenStatusSuccess);
            EXPECT_TRUE(miopenSetConvolutionGroupCount(convDesc, g) == miopenStatusSuccess);

            EXPECT_TRUE(miopenCreateTensorDescriptor(&inDesc) == miopenStatusSuccess);
            EXPECT_TRUE(miopenCreateTensorDescriptor(&weiDesc) == miopenStatusSuccess);
            EXPECT_TRUE(miopenCreateTensorDescriptor(&outDesc) == miopenStatusSuccess);

            EXPECT_TRUE(
                miopenSetTensorDescriptor(
                    inDesc, miopen_type<TRef>{}, in_len.size(), in_len.data(), in_strides.data()) ==
                miopenStatusSuccess);
            EXPECT_TRUE(miopenSetTensorDescriptor(weiDesc,
                                                  miopen_type<TRef>{},
                                                  wei_len.size(),
                                                  wei_len.data(),
                                                  wei_strides.data()) == miopenStatusSuccess);
            EXPECT_TRUE(miopenSetTensorDescriptor(outDesc,
                                                  miopen_type<Tout>{},
                                                  out_len.size(),
                                                  out_len.data(),
                                                  out_strides.data()) == miopenStatusSuccess);

            bool valid_result = false;

            if(direction == miopen::conv::Direction::Forward)
            {
                // initialize data with integer
                rand_tensor_integer(in);
                rand_tensor_integer(wei);
                /// \ref copy_non_packed_output_before_convolution
                rand_tensor_integer(out);

                EXPECT_TRUE(hipMemcpy(in_dev,
                                      in.data.data(),
                                      sizeof(TRef) * in_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                EXPECT_TRUE(hipMemcpy(wei_dev,
                                      wei.data.data(),
                                      sizeof(TRef) * wei_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                /// \anchor copy_non_packed_output_before_convolution
                /// \note Output is a non-packed tensor, which means there are
                /// elements that convolution will not update. In order to verify
                /// the convolution result, the GPU buffer should have the same
                /// data as the CPU in both update and not-updated elements.
                /// Therefore, we copy the output to the GPU buffer after
                /// initializing it with random values.
                ///
                EXPECT_TRUE(hipMemcpy(out_dev,
                                      out.data.data(),
                                      sizeof(Tout) * out_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);

                cpu_convolution_forward(miopen::deref(convDesc).GetSpatialDimension(),
                                        in,
                                        wei,
                                        out,
                                        miopen::deref(convDesc).GetConvPads(),
                                        miopen::deref(convDesc).GetConvStrides(),
                                        miopen::deref(convDesc).GetConvDilations(),
                                        miopen::deref(convDesc).GetGroupCount());

                EXPECT_TRUE(miopenConvolutionForwardImmediate(
                                this->handle,
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

                tensor<Tout> out_host(out_len, out_strides);
                EXPECT_TRUE(hipMemcpy(out_host.data.data(),
                                      out_dev,
                                      sizeof(Tout) * out_sz,
                                      hipMemcpyDeviceToHost) == hipSuccess);

                // we expect excact match, since use integer
                valid_result = verify_tensor(out_host, out);
            }
            else if(direction == miopen::conv::Direction::BackwardData)
            {
                // initialize data with integer
                rand_tensor_integer(out);
                rand_tensor_integer(wei);
                /// \ref copy_non_packed_output_before_convolution
                rand_tensor_integer(in);

                /// \ref copy_non_packed_output_before_convolution
                EXPECT_TRUE(hipMemcpy(in_dev,
                                      in.data.data(),
                                      sizeof(TRef) * in_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                EXPECT_TRUE(hipMemcpy(out_dev,
                                      out.data.data(),
                                      sizeof(Tout) * out_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                EXPECT_TRUE(hipMemcpy(wei_dev,
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

                EXPECT_TRUE(miopenConvolutionBackwardDataImmediate(
                                this->handle,
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

                EXPECT_TRUE(hipMemcpy(in_host.data.data(),
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
                /// \ref copy_non_packed_output_before_convolution
                rand_tensor_integer(wei);

                EXPECT_TRUE(hipMemcpy(in_dev,
                                      in.data.data(),
                                      sizeof(TRef) * in_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                /// \ref copy_non_packed_output_before_convolution
                EXPECT_TRUE(hipMemcpy(wei_dev,
                                      wei.data.data(),
                                      sizeof(TRef) * wei_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                EXPECT_TRUE(hipMemcpy(out_dev,
                                      out.data.data(),
                                      sizeof(Tout) * out_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);

                cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                                in,
                                                wei,
                                                out,
                                                miopen::deref(convDesc).GetConvPads(),
                                                miopen::deref(convDesc).GetConvStrides(),
                                                miopen::deref(convDesc).GetConvDilations(),
                                                miopen::deref(convDesc).GetGroupCount());

                EXPECT_TRUE(miopenConvolutionBackwardWeightsImmediate(
                                this->handle,
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

                EXPECT_TRUE(hipMemcpy(wei_host.data.data(),
                                      wei_dev,
                                      sizeof(TRef) * wei_sz,
                                      hipMemcpyDeviceToHost) == hipSuccess);

                // we expect excact match, since use integer
                valid_result = verify_tensor(wei_host, wei);
            }

            EXPECT_TRUE(valid_result == true);

            miopenDestroyConvolutionDescriptor(convDesc);
            miopenDestroyTensorDescriptor(inDesc);
            miopenDestroyTensorDescriptor(weiDesc);
            miopenDestroyTensorDescriptor(outDesc);

            hipFree(in_dev);
            hipFree(wei_dev);
            hipFree(out_dev);
        };

        this->iterate_conv_2d(run_conv_2d);
    }
};

// clang-format off
typedef ::testing::Types<
        TypeDefs<miopen::conv::Direction::Forward,          float,       float,      miopenTensorNCHW>,
        TypeDefs<miopen::conv::Direction::Forward,          float16,     float16,    miopenTensorNCHW>,
        TypeDefs<miopen::conv::Direction::Forward,          bfloat16,    bfloat16,   miopenTensorNCHW>,
        TypeDefs<miopen::conv::Direction::Forward,          int8_t,      int32_t,    miopenTensorNCHW>,
        TypeDefs<miopen::conv::Direction::Forward,          int8_t,      float,      miopenTensorNCHW>,

        TypeDefs<miopen::conv::Direction::BackwardData,     float,       float,      miopenTensorNCHW>,
        TypeDefs<miopen::conv::Direction::BackwardData,     float16,     float16,    miopenTensorNCHW>,
        TypeDefs<miopen::conv::Direction::BackwardData,     bfloat16,    bfloat16,   miopenTensorNCHW>,

        TypeDefs<miopen::conv::Direction::BackwardWeights,  float,       float,      miopenTensorNCHW>,
        TypeDefs<miopen::conv::Direction::BackwardWeights,  float16,     float16,    miopenTensorNCHW>,
        TypeDefs<miopen::conv::Direction::BackwardWeights,  bfloat16,    bfloat16,   miopenTensorNCHW>,

        TypeDefs<miopen::conv::Direction::Forward,          float,       float,      miopenTensorNHWC>,
        TypeDefs<miopen::conv::Direction::Forward,          float16,     float16,    miopenTensorNHWC>,
        TypeDefs<miopen::conv::Direction::Forward,          bfloat16,    bfloat16,   miopenTensorNHWC>,
        TypeDefs<miopen::conv::Direction::Forward,          int8_t,      int32_t,    miopenTensorNHWC>,
        TypeDefs<miopen::conv::Direction::Forward,          int8_t,      float,      miopenTensorNHWC>,

        TypeDefs<miopen::conv::Direction::BackwardData,     float,       float,      miopenTensorNHWC>,
        TypeDefs<miopen::conv::Direction::BackwardData,     float16,     float16,    miopenTensorNHWC>,
        TypeDefs<miopen::conv::Direction::BackwardData,     bfloat16,    bfloat16,   miopenTensorNHWC>,

        TypeDefs<miopen::conv::Direction::BackwardWeights,  float,       float,      miopenTensorNHWC>,
        TypeDefs<miopen::conv::Direction::BackwardWeights,  float16,     float16,    miopenTensorNHWC>,
        TypeDefs<miopen::conv::Direction::BackwardWeights,  bfloat16,    bfloat16,   miopenTensorNHWC>
> Implementations2d;
// clang-format on

TYPED_TEST_CASE(ReferenceConv2d, Implementations2d);

TYPED_TEST(ReferenceConv2d, Forward2dNCHW)
{
    typedef typename TypeParam::TRef TRef;
    typedef typename TypeParam::Tout Tout;

    this->template run<TypeParam::get_direction(), TRef, Tout, TypeParam::get_tensor_layout()>();
}

template <class Types>
struct ReferenceConv3d : ReferenceConvBase
{
    template <miopen::conv::Direction direction,
              typename TRef,
              typename Tout,
              miopenTensorLayout_t tensor_layout>
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

            std::vector<int> in_len({n, c, di, hi, wi});
            std::vector<int> wei_len({k, c_per_group, fz, fy, fx});
            std::vector<int> out_len({n, k, do_, ho, wo});

            std::vector<int> in_strides;
            std::vector<int> wei_strides;
            std::vector<int> out_strides;

            std::string layout_default = miopen::tensor_layout_get_default(5);
            std::string layout_string  = miopen::TensorDescriptor::GetLayoutStr(tensor_layout);

            miopen::tensor_layout_to_strides(in_len, layout_default, layout_string, in_strides);
            miopen::tensor_layout_to_strides(wei_len, layout_default, layout_string, wei_strides);
            miopen::tensor_layout_to_strides(out_len, layout_default, layout_string, out_strides);

            pad_tensor_strides(in_strides);
            pad_tensor_strides(wei_strides);
            pad_tensor_strides(out_strides);

            tensor<TRef> in(in_len, in_strides);
            tensor<TRef> wei(wei_len, wei_strides);
            tensor<Tout> out(out_len, out_strides);

            auto in_sz  = in.data.size();
            auto wei_sz = wei.data.size();
            auto out_sz = out.data.size();

            void* in_dev;
            void* wei_dev;
            void* out_dev;

            EXPECT_TRUE(hipMalloc(&in_dev, sizeof(TRef) * in_sz) == hipSuccess);
            EXPECT_TRUE(hipMalloc(&wei_dev, sizeof(TRef) * wei_sz) == hipSuccess);
            EXPECT_TRUE(hipMalloc(&out_dev, sizeof(Tout) * out_sz) == hipSuccess);

            EXPECT_TRUE(miopenCreateConvolutionDescriptor(&convDesc) == miopenStatusSuccess);
            EXPECT_TRUE(miopenInitConvolutionNdDescriptor(convDesc,
                                                          3,
                                                          static_cast<int*>(pads),
                                                          static_cast<int*>(strides),
                                                          static_cast<int*>(dilations),
                                                          miopenConvolution) ==
                        miopenStatusSuccess);
            EXPECT_TRUE(miopenSetConvolutionGroupCount(convDesc, g) == miopenStatusSuccess);

            EXPECT_TRUE(miopenCreateTensorDescriptor(&inDesc) == miopenStatusSuccess);
            EXPECT_TRUE(miopenCreateTensorDescriptor(&weiDesc) == miopenStatusSuccess);
            EXPECT_TRUE(miopenCreateTensorDescriptor(&outDesc) == miopenStatusSuccess);

            EXPECT_TRUE(
                miopenSetTensorDescriptor(
                    inDesc, miopen_type<TRef>{}, in_len.size(), in_len.data(), in_strides.data()) ==
                miopenStatusSuccess);
            EXPECT_TRUE(miopenSetTensorDescriptor(weiDesc,
                                                  miopen_type<TRef>{},
                                                  wei_len.size(),
                                                  wei_len.data(),
                                                  wei_strides.data()) == miopenStatusSuccess);
            EXPECT_TRUE(miopenSetTensorDescriptor(outDesc,
                                                  miopen_type<Tout>{},
                                                  out_len.size(),
                                                  out_len.data(),
                                                  out_strides.data()) == miopenStatusSuccess);

            bool valid_result = false;

            if(direction == miopen::conv::Direction::Forward)
            {
                // initialize data with integer
                rand_tensor_integer(in);
                rand_tensor_integer(wei);
                /// \ref copy_non_packed_output_before_convolution
                rand_tensor_integer(out);

                EXPECT_TRUE(hipMemcpy(in_dev,
                                      in.data.data(),
                                      sizeof(TRef) * in_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                /// \ref copy_non_packed_output_before_convolution
                EXPECT_TRUE(hipMemcpy(out_dev,
                                      out.data.data(),
                                      sizeof(Tout) * out_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                EXPECT_TRUE(hipMemcpy(wei_dev,
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

                EXPECT_TRUE(miopenConvolutionForwardImmediate(
                                this->handle,
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

                tensor<Tout> out_host(out_len, out_strides);

                EXPECT_TRUE(hipMemcpy(out_host.data.data(),
                                      out_dev,
                                      sizeof(Tout) * out_sz,
                                      hipMemcpyDeviceToHost) == hipSuccess);

                // we expect excact match, since use integer
                valid_result = verify_tensor(out_host, out);
            }
            else if(direction == miopen::conv::Direction::BackwardData)
            {
                // initialize data with integer
                rand_tensor_integer(out);
                rand_tensor_integer(wei);
                /// \ref copy_non_packed_output_before_convolution
                rand_tensor_integer(in);

                /// \ref copy_non_packed_output_before_convolution
                EXPECT_TRUE(hipMemcpy(in_dev,
                                      in.data.data(),
                                      sizeof(TRef) * in_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                EXPECT_TRUE(hipMemcpy(out_dev,
                                      out.data.data(),
                                      sizeof(Tout) * out_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                EXPECT_TRUE(hipMemcpy(wei_dev,
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

                EXPECT_TRUE(miopenConvolutionBackwardDataImmediate(
                                this->handle,
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

                EXPECT_TRUE(hipMemcpy(in_host.data.data(),
                                      in_dev,
                                      sizeof(TRef) * in_sz,
                                      hipMemcpyDeviceToHost) == hipSuccess);

                // we expect excact match, since use integer
                valid_result = verify_tensor(in_host, in);
            }
            else if(direction == miopen::conv::Direction::BackwardWeights)
            {
                rand_tensor_integer(in, 3, -2);
                rand_tensor_integer(out, 3, -2);
                /// \ref copy_non_packed_output_before_convolution
                rand_tensor_integer(wei);

                EXPECT_TRUE(hipMemcpy(in_dev,
                                      in.data.data(),
                                      sizeof(TRef) * in_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                /// \ref copy_non_packed_output_before_convolution
                EXPECT_TRUE(hipMemcpy(wei_dev,
                                      wei.data.data(),
                                      sizeof(TRef) * wei_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);
                EXPECT_TRUE(hipMemcpy(out_dev,
                                      out.data.data(),
                                      sizeof(Tout) * out_sz,
                                      hipMemcpyHostToDevice) == hipSuccess);

                cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                                in,
                                                wei,
                                                out,
                                                miopen::deref(convDesc).GetConvPads(),
                                                miopen::deref(convDesc).GetConvStrides(),
                                                miopen::deref(convDesc).GetConvDilations(),
                                                miopen::deref(convDesc).GetGroupCount());

                EXPECT_TRUE(miopenConvolutionBackwardWeightsImmediate(
                                this->handle,
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

                EXPECT_TRUE(hipMemcpy(wei_host.data.data(),
                                      wei_dev,
                                      sizeof(TRef) * wei_sz,
                                      hipMemcpyDeviceToHost) == hipSuccess);

                // we expect excact match, since use integer
                valid_result = verify_tensor(wei_host, wei, 8.0); // max possible int
                                                                  // 2*14*14*10*(2*2) = 15680, hence
                                                                  // int interval might be 8
            }

            EXPECT_TRUE(valid_result == true);

            miopenDestroyConvolutionDescriptor(convDesc);
            miopenDestroyTensorDescriptor(inDesc);
            miopenDestroyTensorDescriptor(weiDesc);
            miopenDestroyTensorDescriptor(outDesc);

            hipFree(in_dev);
            hipFree(wei_dev);
            hipFree(out_dev);
        };

        this->iterate_conv_3d(run_conv_3d);
    }
};

// clang-format off
typedef ::testing::Types<
        TypeDefs<miopen::conv::Direction::Forward,          float,       float,      miopenTensorNCDHW>,
        TypeDefs<miopen::conv::Direction::Forward,          float16,     float16,    miopenTensorNCDHW>,
        TypeDefs<miopen::conv::Direction::Forward,          bfloat16,    bfloat16,   miopenTensorNCDHW>,
        TypeDefs<miopen::conv::Direction::Forward,          int8_t,      int32_t,    miopenTensorNCDHW>,
        TypeDefs<miopen::conv::Direction::Forward,          int8_t,      float,      miopenTensorNCDHW>,

        TypeDefs<miopen::conv::Direction::BackwardData,     float,       float,      miopenTensorNCDHW>,
        TypeDefs<miopen::conv::Direction::BackwardData,     float16,     float16,    miopenTensorNCDHW>,
        TypeDefs<miopen::conv::Direction::BackwardData,     bfloat16,    bfloat16,   miopenTensorNCDHW>,

        TypeDefs<miopen::conv::Direction::BackwardWeights,  float,       float,      miopenTensorNCDHW>,
        TypeDefs<miopen::conv::Direction::BackwardWeights,  float16,     float16,    miopenTensorNCDHW>,
        TypeDefs<miopen::conv::Direction::BackwardWeights,  bfloat16,    bfloat16,   miopenTensorNCDHW>,

        TypeDefs<miopen::conv::Direction::Forward,          float,       float,      miopenTensorNDHWC>,
        TypeDefs<miopen::conv::Direction::Forward,          float16,     float16,    miopenTensorNDHWC>,
        TypeDefs<miopen::conv::Direction::Forward,          bfloat16,    bfloat16,   miopenTensorNDHWC>,
        TypeDefs<miopen::conv::Direction::Forward,          int8_t,      int32_t,    miopenTensorNDHWC>,
        TypeDefs<miopen::conv::Direction::Forward,          int8_t,      float,      miopenTensorNDHWC>,

        TypeDefs<miopen::conv::Direction::BackwardData,     float,       float,      miopenTensorNDHWC>,
        TypeDefs<miopen::conv::Direction::BackwardData,     float16,     float16,    miopenTensorNDHWC>,
        TypeDefs<miopen::conv::Direction::BackwardData,     bfloat16,    bfloat16,   miopenTensorNDHWC>,

        TypeDefs<miopen::conv::Direction::BackwardWeights,  float,       float,      miopenTensorNDHWC>,
        TypeDefs<miopen::conv::Direction::BackwardWeights,  float16,     float16,    miopenTensorNDHWC>,
        TypeDefs<miopen::conv::Direction::BackwardWeights,  bfloat16,    bfloat16,   miopenTensorNDHWC>
> Implementations3d;
// clang-format on

TYPED_TEST_CASE(ReferenceConv3d, Implementations3d);

TYPED_TEST(ReferenceConv3d, Forward3dNCHW)
{
    typedef typename TypeParam::TRef TRef;
    typedef typename TypeParam::Tout Tout;

    this->template run<TypeParam::get_direction(), TRef, Tout, TypeParam::get_tensor_layout()>();
}

TEST(CONV_IMMEDIATE, test_all)
{
    // 2d NCHW
    // run_test<ReferenceConv2d<miopen::conv::Direction::Forward,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNCHW>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::Forward,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNCHW>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::Forward,
    //                                int8_t,
    //                                int32_t,
    //                                miopenTensorNCHW>>();
    // run_test<
    //     ReferenceConv2d<miopen::conv::Direction::Forward, int8_t, float, miopenTensorNCHW>>();

    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardData,
    //                                float,
    //                                float,
    //                                miopenTensorNCHW>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardData,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNCHW>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardData,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNCHW>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardWeights,
    //                                float,
    //                                float,
    //                                miopenTensorNCHW>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardWeights,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNCHW>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardWeights,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNCHW>>();

    // 3d NCDHW
    // run_test<
    //     gpu_reference_conv_3d<miopen::conv::Direction::Forward, float, float, miopenTensorNCDHW>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNCDHW>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNCDHW>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
    //                                int8_t,
    //                                int32_t,
    //                                miopenTensorNCDHW>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
    //                                int8_t,
    //                                float,
    //                                miopenTensorNCDHW>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
    //                                float,
    //                                float,
    //                                miopenTensorNCDHW>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNCDHW>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNCDHW>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
    //                                float,
    //                                float,
    //                                miopenTensorNCDHW>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNCDHW>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNCDHW>>();

    // 2d NHWC
    // run_test<
    //     ReferenceConv2d<miopen::conv::Direction::Forward, float, float, miopenTensorNHWC>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::Forward,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNHWC>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::Forward,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNHWC>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::Forward,
    //                                int8_t,
    //                                int32_t,
    //                                miopenTensorNHWC>>();
    // run_test<
    //     ReferenceConv2d<miopen::conv::Direction::Forward, int8_t, float, miopenTensorNHWC>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardData,
    //                                float,
    //                                float,
    //                                miopenTensorNHWC>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardData,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNHWC>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardData,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNHWC>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardWeights,
    //                                float,
    //                                float,
    //                                miopenTensorNHWC>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardWeights,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNHWC>>();
    // run_test<ReferenceConv2d<miopen::conv::Direction::BackwardWeights,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNHWC>>();

    // 3d NDHWC
    // run_test<
    //     gpu_reference_conv_3d<miopen::conv::Direction::Forward, float, float, miopenTensorNDHWC>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNDHWC>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNDHWC>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
    //                                int8_t,
    //                                int32_t,
    //                                miopenTensorNDHWC>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::Forward,
    //                                int8_t,
    //                                float,
    //                                miopenTensorNDHWC>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
    //                                float,
    //                                float,
    //                                miopenTensorNDHWC>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNDHWC>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNDHWC>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
    //                                float,
    //                                float,
    //                                miopenTensorNDHWC>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
    //                                half_float::half,
    //                                half_float::half,
    //                                miopenTensorNDHWC>>();
    // run_test<gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
    //                                bfloat16,
    //                                bfloat16,
    //                                miopenTensorNDHWC>>();
}
