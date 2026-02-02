// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <miopen/convolution.hpp>
#include <miopen/tensor_layout.hpp>

#include "../cpu_conv.hpp"
#include "../random.hpp"
#include "../tensor_holder.hpp"
#include "../verify.hpp"

namespace {
struct TestCase2D
{
    int n;
    int wi;
    int hi;
    int c;
    int k;
    int fx;
    int fy;
    int px;
    int py;
    int sx;
    int sy;
    int dx;
    int dy;
    int g;
};

struct TestCase3D
{
    int n;
    int di;
    int wi;
    int hi;
    int c;
    int k;
    int fz;
    int fx;
    int fy;
    int pz;
    int px;
    int py;
    int sz;
    int sx;
    int sy;
    int dz;
    int dx;
    int dy;
    int g;
};

static constexpr int RAND_INTEGER_MAX       = 5;
static constexpr int RAND_INTEGER_MIN       = -4;
static constexpr float MAX_INTEGER_INTERVAL = 4.f;

auto NameGenerator(const ::testing::TestParamInfo<TestCase2D>& info)
{
    std::stringstream ss{};
    ss << "n" << info.param.n << "wi" << info.param.wi << "hi" << info.param.hi << "c"
       << info.param.c << "k" << info.param.k << "fx" << info.param.fx << "fy" << info.param.fy
       << "px" << info.param.px << "py" << info.param.py << "sx" << info.param.sx << "sy"
       << info.param.sy << "dx" << info.param.dx << "dy" << info.param.dy << "g" << info.param.g;
    return ss.str();
}

auto NameGenerator(const ::testing::TestParamInfo<TestCase3D>& info)
{
    std::stringstream ss{};
    ss << "n" << info.param.n << "di" << info.param.di << "wi" << info.param.wi << "hi"
       << info.param.hi << "c" << info.param.c << "k" << info.param.k << "fz" << info.param.fz
       << "fx" << info.param.fx << "fy" << info.param.fy << "pz" << info.param.pz << "px"
       << info.param.px << "py" << info.param.py << "sz" << info.param.sz << "sx" << info.param.sx
       << "sy" << info.param.sy << "dz" << info.param.dz << "dx" << info.param.dx << "dy"
       << info.param.dy << "g" << info.param.g;

    return ss.str();
}

int conv_out_size(int in_size, int pad, int dilation, int ksize, int stride)
{
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

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

std::vector<int> get_image_depth() { return {8, 10}; }

std::vector<int> get_image_size() { return {9, 14}; }

// Warning: Channel size must be multiple of group size
std::vector<int> get_channel_size() { return {4, 8}; }

std::vector<int> get_filter_depth() { return {1, 3}; }

std::vector<int> get_filter_size() { return {1, 3}; }

std::vector<int> get_stride_depth() { return {1, 2}; }

std::vector<int> get_dilation_depth() { return {1}; }

std::vector<int> get_stride_dilation_size() { return {1, 2}; }

std::vector<int> get_pad_depth() { return {0, 1}; }

std::vector<int> get_pad_size() { return {0, 1}; }

std::vector<int> get_group_size() { return {1, 2}; }

std::vector<int> get_batch_size() { return {1, 2}; }

std::vector<TestCase2D> generate_conv_2d()
{
    std::vector<TestCase2D> result{};
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
                                int n  = get_batch_size()[prng::gen_canonical<size_t>()];
                                int g  = get_group_size()[prng::gen_canonical<size_t>()];
                                int k  = get_channel_size()[prng::gen_canonical<size_t>()];
                                int sy = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
                                int sx = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
                                int dy = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
                                int dx = get_stride_dilation_size()[prng::gen_canonical<size_t>()];
                                int ho = conv_out_size(hi, py, dy, fy, sy);
                                int wo = conv_out_size(wi, px, dx, fx, sx);

                                if(fy > hi || fx > wi || (fy - 1) < py || (fx - 1) < px ||
                                   ho <= 0 || wo <= 0 || c % g != 0 || k % g != 0)
                                    continue;
                                if((fx == 3 && fy == 5) || (fx == 5 && fy == 3))
                                    continue;
                                result.push_back(
                                    TestCase2D{n, wi, hi, c, k, fx, fy, px, py, sx, sy, dx, dy, g});
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

auto GenCases2D()
{
    static const auto cases = generate_conv_2d();
    return cases;
}

std::vector<TestCase3D> generate_conv_3d()
{
    std::vector<TestCase3D> result{};
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
                        result.push_back(TestCase3D{n,
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
                                                    g});
                    }
                }
            }
        }
    }

    return result;
}

auto GenCases3D()
{
    static const auto cases = generate_conv_3d();
    return cases;
}

template <typename T>
bool verify_tensor(tensor<T>& t_gpu,
                   tensor<T>& t_cpu,
                   float integer_interval = MAX_INTEGER_INTERVAL)
{
    EXPECT_EQ(t_gpu.data.size(), t_cpu.data.size()) << "size not equal, should not happen\n";
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

    EXPECT_TRUE(valid_result) << "diff at:" << idx << ", gpu:" << t_gpu[idx]
                              << ", cpu:" << t_cpu[idx] << std::endl;
    return valid_result;
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

std::string direction_to_string(miopen::conv::Direction direction)
{
    if(direction == miopen::conv::Direction::Forward)
        return "fwd";
    if(direction == miopen::conv::Direction::BackwardData)
        return "bwd";
    if(direction == miopen::conv::Direction::BackwardWeights)
        return "wrw";
    return "n/a";
}

std::string miopen_type_to_string(miopenDataType_t type)
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

template <miopen::conv::Direction direction,
          typename TRef,
          typename Tout,
          miopenTensorLayout_t tensor_layout>
struct gpu_reference_conv_2d : public ::testing::TestWithParam<TestCase2D>
{
    miopenHandle_t handle{};

    void SetUp() override
    {
        prng::reset_seed();
        miopenCreate(&handle);
    }

    void TearDown() override { miopenDestroy(handle); }

    void Run()
    {
        auto const& param = GetParam();
        miopenConvolutionDescriptor_t convDesc;
        miopenTensorDescriptor_t inDesc, weiDesc, outDesc;

        int pads[]      = {param.py, param.px};
        int strides[]   = {param.sy, param.sx};
        int dilations[] = {param.dy, param.dx};
        int ho          = conv_out_size(param.hi, param.py, param.dy, param.fy, param.sy);
        int wo          = conv_out_size(param.wi, param.px, param.dx, param.fx, param.sx);
        int c_per_group = param.c / param.g;

        std::vector<int> in_len({param.n, param.c, param.hi, param.wi});
        std::vector<int> wei_len({param.k, c_per_group, param.fy, param.fx});
        std::vector<int> out_len({param.n, param.k, ho, wo});

        std::vector<int> in_strides;
        std::vector<int> wei_strides;
        std::vector<int> out_strides;

        std::string layout_default = miopen::tensor_layout_get_default(4);
        std::string layout_string  = miopen::TensorDescriptor::LayoutEnumToStr(tensor_layout);

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

        ASSERT_EQ(hipMalloc(&in_dev, sizeof(TRef) * in_sz), hipSuccess);
        ASSERT_EQ(hipMalloc(&wei_dev, sizeof(TRef) * wei_sz), hipSuccess);
        ASSERT_EQ(hipMalloc(&out_dev, sizeof(Tout) * out_sz), hipSuccess);

        ASSERT_EQ(miopenCreateConvolutionDescriptor(&convDesc), miopenStatusSuccess);
        ASSERT_EQ(miopenInitConvolutionNdDescriptor(convDesc,
                                                    2,
                                                    static_cast<int*>(pads),
                                                    static_cast<int*>(strides),
                                                    static_cast<int*>(dilations),
                                                    miopenConvolution),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSetConvolutionGroupCount(convDesc, param.g), miopenStatusSuccess);

        ASSERT_EQ(miopenCreateTensorDescriptor(&inDesc), miopenStatusSuccess);
        ASSERT_EQ(miopenCreateTensorDescriptor(&weiDesc), miopenStatusSuccess);
        ASSERT_EQ(miopenCreateTensorDescriptor(&outDesc), miopenStatusSuccess);

        ASSERT_EQ(miopenSetTensorDescriptor(
                      inDesc, miopen_type<TRef>{}, in_len.size(), in_len.data(), in_strides.data()),
                  miopenStatusSuccess);
        ASSERT_EQ(
            miopenSetTensorDescriptor(
                weiDesc, miopen_type<TRef>{}, wei_len.size(), wei_len.data(), wei_strides.data()),
            miopenStatusSuccess);
        ASSERT_EQ(
            miopenSetTensorDescriptor(
                outDesc, miopen_type<Tout>{}, out_len.size(), out_len.data(), out_strides.data()),
            miopenStatusSuccess);

        bool valid_result = false;

        if(direction == miopen::conv::Direction::Forward)
        {
            // initialize data with integer
            rand_tensor_integer(in);
            rand_tensor_integer(wei);
            /// \ref copy_non_packed_output_before_convolution
            rand_tensor_integer(out);
            ASSERT_EQ(
                hipMemcpy(in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(wei_dev, wei.data.data(), sizeof(TRef) * wei_sz, hipMemcpyHostToDevice),
                hipSuccess);
            /// \anchor copy_non_packed_output_before_convolution
            /// \note Output is a non-packed tensor, which means there are
            /// elements that convolution will not update. In order to verify
            /// the convolution result, the GPU buffer should have the same
            /// data as the CPU in both update and not-updated elements.
            /// Therefore, we copy the output to the GPU buffer after
            /// initializing it with random values.
            ///
            ASSERT_EQ(
                hipMemcpy(out_dev, out.data.data(), sizeof(Tout) * out_sz, hipMemcpyHostToDevice),
                hipSuccess);
            cpu_convolution_forward(miopen::deref(convDesc).GetSpatialDimension(),
                                    in,
                                    wei,
                                    out,
                                    miopen::deref(convDesc).GetConvPads(),
                                    miopen::deref(convDesc).GetConvStrides(),
                                    miopen::deref(convDesc).GetConvDilations(),
                                    miopen::deref(convDesc).GetGroupCount());

            ASSERT_EQ(miopenConvolutionForwardImmediate(
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
                          miopen::solver::Id("ConvDirectNaiveConvFwd").Value()),
                      miopenStatusSuccess);

            tensor<Tout> out_host(out_len, out_strides);
            ASSERT_EQ(
                hipMemcpy(
                    out_host.data.data(), out_dev, sizeof(Tout) * out_sz, hipMemcpyDeviceToHost),
                hipSuccess);

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
            ASSERT_EQ(
                hipMemcpy(in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(out_dev, out.data.data(), sizeof(Tout) * out_sz, hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(wei_dev, wei.data.data(), sizeof(TRef) * wei_sz, hipMemcpyHostToDevice),
                hipSuccess);
            cpu_convolution_backward_data(miopen::deref(convDesc).GetSpatialDimension(),
                                          in,
                                          wei,
                                          out,
                                          miopen::deref(convDesc).GetConvPads(),
                                          miopen::deref(convDesc).GetConvStrides(),
                                          miopen::deref(convDesc).GetConvDilations(),
                                          miopen::deref(convDesc).GetGroupCount());

            EXPECT_EQ(miopenConvolutionBackwardDataImmediate(
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
                          miopen::solver::Id("ConvDirectNaiveConvBwd").Value()),
                      miopenStatusSuccess);

            tensor<TRef> in_host(in_len, in_strides);

            ASSERT_EQ(
                hipMemcpy(in_host.data.data(), in_dev, sizeof(TRef) * in_sz, hipMemcpyDeviceToHost),
                hipSuccess);

            // we expect excact match, since use integer
            valid_result = verify_tensor(in_host, in);
        }
        else if(direction == miopen::conv::Direction::BackwardWeights)
        {
            rand_tensor_integer(in);
            rand_tensor_integer(out);
            /// \ref copy_non_packed_output_before_convolution
            rand_tensor_integer(wei);
            ASSERT_EQ(
                hipMemcpy(in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice),
                hipSuccess);
            /// \ref copy_non_packed_output_before_convolution
            ASSERT_EQ(
                hipMemcpy(wei_dev, wei.data.data(), sizeof(TRef) * wei_sz, hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(out_dev, out.data.data(), sizeof(Tout) * out_sz, hipMemcpyHostToDevice),
                hipSuccess);
            cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                            in,
                                            wei,
                                            out,
                                            miopen::deref(convDesc).GetConvPads(),
                                            miopen::deref(convDesc).GetConvStrides(),
                                            miopen::deref(convDesc).GetConvDilations(),
                                            miopen::deref(convDesc).GetGroupCount());

            EXPECT_EQ(miopenConvolutionBackwardWeightsImmediate(
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
                          miopen::solver::Id("ConvDirectNaiveConvWrw").Value()),
                      miopenStatusSuccess);

            tensor<TRef> wei_host(wei_len, wei_strides);

            ASSERT_EQ(
                hipMemcpy(
                    wei_host.data.data(), wei_dev, sizeof(TRef) * wei_sz, hipMemcpyDeviceToHost),
                hipSuccess);

            // we expect excact match, since use integer
            valid_result = verify_tensor(wei_host, wei);
        }

        std::cout << "n:" << param.n << ", c:" << param.c << ", hi:" << param.hi
                  << ", wi:" << param.wi << ", k:" << param.k << ", ho:" << ho << ", wo:" << wo
                  << ", fy:" << param.fy << ",fx:" << param.fx << ", py:" << param.py
                  << ", px:" << param.px << ", sy:" << param.sy << ", sx:" << param.sx
                  << ", dy:" << param.dy << ",dx:" << param.dx << ", g:" << param.g
                  << ", dir:" << direction_to_string(direction)
                  << ", type:" << miopen_type_to_string(miopen_type<TRef>{})
                  << ", layout:" << layout_string << ", valid:" << valid_result << std::endl;
        EXPECT_EQ(valid_result, true);

        miopenDestroyConvolutionDescriptor(convDesc);
        miopenDestroyTensorDescriptor(inDesc);
        miopenDestroyTensorDescriptor(weiDesc);
        miopenDestroyTensorDescriptor(outDesc);

        hipFree(in_dev);
        hipFree(wei_dev);
        hipFree(out_dev);
    }
};

template <miopen::conv::Direction direction,
          typename TRef,
          typename Tout,
          miopenTensorLayout_t tensor_layout>
struct gpu_reference_conv_3d : public ::testing::TestWithParam<TestCase3D>
{
    miopenHandle_t handle{};

    void SetUp() override
    {
        prng::reset_seed();
        miopenCreate(&handle);
    }

    void TearDown() override { miopenDestroy(handle); }

    void Run()
    {
        miopenConvolutionDescriptor_t convDesc;
        miopenTensorDescriptor_t inDesc, weiDesc, outDesc;

        auto const& param = GetParam();

        int pads[]      = {param.pz, param.py, param.px};
        int strides[]   = {param.sz, param.sy, param.sx};
        int dilations[] = {param.dz, param.dy, param.dx};
        int ho          = conv_out_size(param.hi, param.py, param.dy, param.fy, param.sy);
        int wo          = conv_out_size(param.wi, param.px, param.dx, param.fx, param.sx);
        int do_         = conv_out_size(param.di, param.pz, param.dz, param.fz, param.sz);
        int c_per_group = param.c / param.g;

        std::vector<int> in_len({param.n, param.c, param.di, param.hi, param.wi});
        std::vector<int> wei_len({param.k, c_per_group, param.fz, param.fy, param.fx});
        std::vector<int> out_len({param.n, param.k, do_, ho, wo});

        std::vector<int> in_strides;
        std::vector<int> wei_strides;
        std::vector<int> out_strides;

        std::string layout_default = miopen::tensor_layout_get_default(5);
        std::string layout_string  = miopen::TensorDescriptor::LayoutEnumToStr(tensor_layout);

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

        ASSERT_EQ(hipMalloc(&in_dev, sizeof(TRef) * in_sz), hipSuccess);
        ASSERT_EQ(hipMalloc(&wei_dev, sizeof(TRef) * wei_sz), hipSuccess);
        ASSERT_EQ(hipMalloc(&out_dev, sizeof(Tout) * out_sz), hipSuccess);
        ASSERT_EQ(miopenCreateConvolutionDescriptor(&convDesc), miopenStatusSuccess);
        ASSERT_EQ(miopenInitConvolutionNdDescriptor(convDesc,
                                                    3,
                                                    static_cast<int*>(pads),
                                                    static_cast<int*>(strides),
                                                    static_cast<int*>(dilations),
                                                    miopenConvolution),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenSetConvolutionGroupCount(convDesc, param.g), miopenStatusSuccess);

        ASSERT_EQ(miopenCreateTensorDescriptor(&inDesc), miopenStatusSuccess);
        ASSERT_EQ(miopenCreateTensorDescriptor(&weiDesc), miopenStatusSuccess);
        ASSERT_EQ(miopenCreateTensorDescriptor(&outDesc), miopenStatusSuccess);

        ASSERT_EQ(miopenSetTensorDescriptor(
                      inDesc, miopen_type<TRef>{}, in_len.size(), in_len.data(), in_strides.data()),
                  miopenStatusSuccess);
        ASSERT_EQ(
            miopenSetTensorDescriptor(
                weiDesc, miopen_type<TRef>{}, wei_len.size(), wei_len.data(), wei_strides.data()),
            miopenStatusSuccess);
        ASSERT_EQ(
            miopenSetTensorDescriptor(
                outDesc, miopen_type<Tout>{}, out_len.size(), out_len.data(), out_strides.data()),
            miopenStatusSuccess);

        bool valid_result = false;

        if(direction == miopen::conv::Direction::Forward)
        {
            // initialize data with integer
            rand_tensor_integer(in);
            rand_tensor_integer(wei);
            /// \ref copy_non_packed_output_before_convolution
            rand_tensor_integer(out);
            ASSERT_EQ(
                hipMemcpy(in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice),
                hipSuccess);
            /// \ref copy_non_packed_output_before_convolution
            ASSERT_EQ(
                hipMemcpy(out_dev, out.data.data(), sizeof(Tout) * out_sz, hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(wei_dev, wei.data.data(), sizeof(TRef) * wei_sz, hipMemcpyHostToDevice),
                hipSuccess);
            cpu_convolution_forward(miopen::deref(convDesc).GetSpatialDimension(),
                                    in,
                                    wei,
                                    out,
                                    miopen::deref(convDesc).GetConvPads(),
                                    miopen::deref(convDesc).GetConvStrides(),
                                    miopen::deref(convDesc).GetConvDilations(),
                                    miopen::deref(convDesc).GetGroupCount());

            ASSERT_EQ(miopenConvolutionForwardImmediate(
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
                          miopen::solver::Id("ConvDirectNaiveConvFwd").Value()),
                      miopenStatusSuccess);

            tensor<Tout> out_host(out_len, out_strides);

            ASSERT_EQ(
                hipMemcpy(
                    out_host.data.data(), out_dev, sizeof(Tout) * out_sz, hipMemcpyDeviceToHost),
                hipSuccess);

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
            ASSERT_EQ(
                hipMemcpy(in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(out_dev, out.data.data(), sizeof(Tout) * out_sz, hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(wei_dev, wei.data.data(), sizeof(TRef) * wei_sz, hipMemcpyHostToDevice),
                hipSuccess);
            cpu_convolution_backward_data(miopen::deref(convDesc).GetSpatialDimension(),
                                          in,
                                          wei,
                                          out,
                                          miopen::deref(convDesc).GetConvPads(),
                                          miopen::deref(convDesc).GetConvStrides(),
                                          miopen::deref(convDesc).GetConvDilations(),
                                          miopen::deref(convDesc).GetGroupCount());

            ASSERT_EQ(miopenConvolutionBackwardDataImmediate(
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
                          miopen::solver::Id("ConvDirectNaiveConvBwd").Value()),
                      miopenStatusSuccess);

            tensor<TRef> in_host(in_len, in_strides);

            ASSERT_EQ(
                hipMemcpy(in_host.data.data(), in_dev, sizeof(TRef) * in_sz, hipMemcpyDeviceToHost),
                hipSuccess);

            // we expect excact match, since use integer
            valid_result = verify_tensor(in_host, in);
        }
        else if(direction == miopen::conv::Direction::BackwardWeights)
        {
            rand_tensor_integer(in, 3, -2);
            rand_tensor_integer(out, 3, -2);
            /// \ref copy_non_packed_output_before_convolution
            rand_tensor_integer(wei);
            ASSERT_EQ(
                hipMemcpy(in_dev, in.data.data(), sizeof(TRef) * in_sz, hipMemcpyHostToDevice),
                hipSuccess);
            /// \ref copy_non_packed_output_before_convolution
            ASSERT_EQ(
                hipMemcpy(wei_dev, wei.data.data(), sizeof(TRef) * wei_sz, hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(out_dev, out.data.data(), sizeof(Tout) * out_sz, hipMemcpyHostToDevice),
                hipSuccess);
            cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                            in,
                                            wei,
                                            out,
                                            miopen::deref(convDesc).GetConvPads(),
                                            miopen::deref(convDesc).GetConvStrides(),
                                            miopen::deref(convDesc).GetConvDilations(),
                                            miopen::deref(convDesc).GetGroupCount());

            ASSERT_EQ(miopenConvolutionBackwardWeightsImmediate(
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
                          miopen::solver::Id("ConvDirectNaiveConvWrw").Value()),
                      miopenStatusSuccess);

            tensor<TRef> wei_host(wei_len, wei_strides);

            ASSERT_EQ(
                hipMemcpy(
                    wei_host.data.data(), wei_dev, sizeof(TRef) * wei_sz, hipMemcpyDeviceToHost),
                hipSuccess);

            // we expect excact match, since use integer
            valid_result = verify_tensor(wei_host, wei, 8.0); // max possible int
                                                              // 2*14*14*10*(2*2) = 15680, hence
                                                              // int interval might be 8
        }

        // auto error        = miopen::rms_range(out_host.data, out.data);
        // auto tolerance = get_default_tolerence<TRef>();
        // bool valid_result = error <= tolerance;
        std::cout << "n:" << param.n << ", c:" << param.c << ", di:" << param.di
                  << ", hi:" << param.hi << ", wi:" << param.wi << ", k:" << param.k
                  << ", do:" << do_ << ", ho:" << ho << ", wo:" << wo << ", fz:" << param.fz
                  << ", fy:" << param.fy << ",fx:" << param.fx << ", pz:" << param.pz
                  << ", py:" << param.py << ", px:" << param.px << ", sz:" << param.sz
                  << ", sy:" << param.sy << ", sx:" << param.sx << ", dz:" << param.dz
                  << ", dy:" << param.dy << ", dx:" << param.dx << ", g:" << param.g
                  << ", dir:" << direction_to_string(direction)
                  << ", type:" << miopen_type_to_string(miopen_type<TRef>{})
                  << ", layout:" << layout_string << ", valid:" << valid_result << std::endl;
        EXPECT_EQ(valid_result, true);

        miopenDestroyConvolutionDescriptor(convDesc);
        miopenDestroyTensorDescriptor(inDesc);
        miopenDestroyTensorDescriptor(weiDesc);
        miopenDestroyTensorDescriptor(outDesc);

        hipFree(in_dev);
        hipFree(wei_dev);
        hipFree(out_dev);
    }
};
} // namespace

// 2d NCHW
using GPU_reference_kernel_fwd_2d_NCHW_FP32_FP32 =
    gpu_reference_conv_2d<miopen::conv::Direction::Forward, float, float, miopenTensorNCHW>;
using GPU_reference_kernel_fwd_2d_NCHW_FP16_FP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::Forward,
                          half_float::half,
                          half_float::half,
                          miopenTensorNCHW>;
using GPU_reference_kernel_fwd_2d_NCHW_BFP16_BFP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::Forward, bfloat16, bfloat16, miopenTensorNCHW>;
using GPU_reference_kernel_fwd_2d_NCHW_I8_I32 =
    gpu_reference_conv_2d<miopen::conv::Direction::Forward, int8_t, int32_t, miopenTensorNCHW>;

using GPU_reference_kernel_fwd_2d_NCHW_I8_FP32 =
    gpu_reference_conv_2d<miopen::conv::Direction::Forward, int8_t, float, miopenTensorNCHW>;

using GPU_reference_kernel_bwd_2d_NCHW_FP32_FP32 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardData, float, float, miopenTensorNCHW>;

using GPU_reference_kernel_bwd_2d_NCHW_FP16_FP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardData,
                          half_float::half,
                          half_float::half,
                          miopenTensorNCHW>;

using GPU_reference_kernel_bwd_2d_NCHW_BFP16_BFP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardData,
                          bfloat16,
                          bfloat16,
                          miopenTensorNCHW>;

using GPU_reference_kernel_bww_2d_NCHW_FP32_FP32 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights, float, float, miopenTensorNCHW>;

using GPU_reference_kernel_bww_2d_NCHW_FP16_FP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights,
                          half_float::half,
                          half_float::half,
                          miopenTensorNCHW>;

using GPU_reference_kernel_bww_2d_NCHW_BFP16_BFP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights,
                          bfloat16,
                          bfloat16,
                          miopenTensorNCHW>;

// 3d NCDHW
using GPU_reference_kernel_fwd_3d_NCDHW_FP32_FP32 =
    gpu_reference_conv_3d<miopen::conv::Direction::Forward, float, float, miopenTensorNCDHW>;

using GPU_reference_kernel_fwd_3d_NCDHW_FP16_FP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::Forward,
                          half_float::half,
                          half_float::half,
                          miopenTensorNCDHW>;

using GPU_reference_kernel_fwd_3d_NCDHW_BFP16_BFP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::Forward, bfloat16, bfloat16, miopenTensorNCDHW>;

using GPU_reference_kernel_fwd_3d_NCDHW_I8_I32 =
    gpu_reference_conv_3d<miopen::conv::Direction::Forward, int8_t, int32_t, miopenTensorNCDHW>;

using GPU_reference_kernel_fwd_3d_NCDHW_I8_FP32 =
    gpu_reference_conv_3d<miopen::conv::Direction::Forward, int8_t, float, miopenTensorNCDHW>;

using GPU_reference_kernel_bwd_3d_NCDHW_FP32_FP32 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardData, float, float, miopenTensorNCDHW>;

using GPU_reference_kernel_bwd_3d_NCDHW_FP16_FP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
                          half_float::half,
                          half_float::half,
                          miopenTensorNCDHW>;

using GPU_reference_kernel_bwd_3d_NCDHW_BFP16_BFP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
                          bfloat16,
                          bfloat16,
                          miopenTensorNCDHW>;

using GPU_reference_kernel_bww_3d_NCDHW_FP32_FP32 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                          float,
                          float,
                          miopenTensorNCDHW>;

using GPU_reference_kernel_bww_3d_NCDHW_FP16_FP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                          half_float::half,
                          half_float::half,
                          miopenTensorNCDHW>;

using GPU_reference_kernel_bww_3d_NCDHW_BFP16_BFP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                          bfloat16,
                          bfloat16,
                          miopenTensorNCDHW>;

// 2d NHWC
using GPU_reference_kernel_fwd_2d_NHWC_FP32_FP32 =
    gpu_reference_conv_2d<miopen::conv::Direction::Forward, float, float, miopenTensorNHWC>;
using GPU_reference_kernel_fwd_2d_NHWC_FP16_FP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::Forward,
                          half_float::half,
                          half_float::half,
                          miopenTensorNHWC>;
using GPU_reference_kernel_fwd_2d_NHWC_BFP16_BFP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::Forward, bfloat16, bfloat16, miopenTensorNHWC>;
using GPU_reference_kernel_fwd_2d_NHWC_I8_I32 =
    gpu_reference_conv_2d<miopen::conv::Direction::Forward, int8_t, int32_t, miopenTensorNHWC>;

using GPU_reference_kernel_fwd_2d_NHWC_I8_FP32 =
    gpu_reference_conv_2d<miopen::conv::Direction::Forward, int8_t, float, miopenTensorNHWC>;

using GPU_reference_kernel_bwd_2d_NHWC_FP32_FP32 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardData, float, float, miopenTensorNHWC>;

using GPU_reference_kernel_bwd_2d_NHWC_FP16_FP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardData,
                          half_float::half,
                          half_float::half,
                          miopenTensorNHWC>;

using GPU_reference_kernel_bwd_2d_NHWC_BFP16_BFP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardData,
                          bfloat16,
                          bfloat16,
                          miopenTensorNHWC>;

using GPU_reference_kernel_bww_2d_NHWC_FP32_FP32 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights, float, float, miopenTensorNHWC>;

using GPU_reference_kernel_bww_2d_NHWC_FP16_FP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights,
                          half_float::half,
                          half_float::half,
                          miopenTensorNHWC>;

using GPU_reference_kernel_bww_2d_NHWC_BFP16_BFP16 =
    gpu_reference_conv_2d<miopen::conv::Direction::BackwardWeights,
                          bfloat16,
                          bfloat16,
                          miopenTensorNHWC>;

// 3d NCDHW
using GPU_reference_kernel_fwd_3d_NDHWC_FP32_FP32 =
    gpu_reference_conv_3d<miopen::conv::Direction::Forward, float, float, miopenTensorNDHWC>;

using GPU_reference_kernel_fwd_3d_NDHWC_FP16_FP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::Forward,
                          half_float::half,
                          half_float::half,
                          miopenTensorNDHWC>;

using GPU_reference_kernel_fwd_3d_NDHWC_BFP16_BFP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::Forward, bfloat16, bfloat16, miopenTensorNDHWC>;

using GPU_reference_kernel_fwd_3d_NDHWC_I8_I32 =
    gpu_reference_conv_3d<miopen::conv::Direction::Forward, int8_t, int32_t, miopenTensorNDHWC>;

using GPU_reference_kernel_fwd_3d_NDHWC_I8_FP32 =
    gpu_reference_conv_3d<miopen::conv::Direction::Forward, int8_t, float, miopenTensorNDHWC>;

using GPU_reference_kernel_bwd_3d_NDHWC_FP32_FP32 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardData, float, float, miopenTensorNDHWC>;

using GPU_reference_kernel_bwd_3d_NDHWC_FP16_FP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
                          half_float::half,
                          half_float::half,
                          miopenTensorNDHWC>;

using GPU_reference_kernel_bwd_3d_NDHWC_BFP16_BFP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardData,
                          bfloat16,
                          bfloat16,
                          miopenTensorNDHWC>;

using GPU_reference_kernel_bww_3d_NDHWC_FP32_FP32 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                          float,
                          float,
                          miopenTensorNDHWC>;

using GPU_reference_kernel_bww_3d_NDHWC_FP16_FP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                          half_float::half,
                          half_float::half,
                          miopenTensorNDHWC>;

using GPU_reference_kernel_bww_3d_NDHWC_BFP16_BFP16 =
    gpu_reference_conv_3d<miopen::conv::Direction::BackwardWeights,
                          bfloat16,
                          bfloat16,
                          miopenTensorNDHWC>;

TEST_P(GPU_reference_kernel_fwd_2d_NCHW_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_2d_NCHW_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_2d_NCHW_BFP16_BFP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_2d_NCHW_I8_I32, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_2d_NCHW_I8_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_2d_NCHW_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_2d_NCHW_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_2d_NCHW_BFP16_BFP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_2d_NCHW_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_2d_NCHW_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_2d_NCHW_BFP16_BFP16, Test) { Run(); }

TEST_P(GPU_reference_kernel_fwd_3d_NCDHW_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_3d_NCDHW_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_3d_NCDHW_BFP16_BFP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_3d_NCDHW_I8_I32, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_3d_NCDHW_I8_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_3d_NCDHW_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_3d_NCDHW_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_3d_NCDHW_BFP16_BFP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_3d_NCDHW_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_3d_NCDHW_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_3d_NCDHW_BFP16_BFP16, Test) { Run(); }

TEST_P(GPU_reference_kernel_fwd_2d_NHWC_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_2d_NHWC_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_2d_NHWC_BFP16_BFP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_2d_NHWC_I8_I32, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_2d_NHWC_I8_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_2d_NHWC_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_2d_NHWC_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_2d_NHWC_BFP16_BFP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_2d_NHWC_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_2d_NHWC_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_2d_NHWC_BFP16_BFP16, Test) { Run(); }

TEST_P(GPU_reference_kernel_fwd_3d_NDHWC_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_3d_NDHWC_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_3d_NDHWC_BFP16_BFP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_3d_NDHWC_I8_I32, Test) { Run(); }
TEST_P(GPU_reference_kernel_fwd_3d_NDHWC_I8_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_3d_NDHWC_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_3d_NDHWC_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bwd_3d_NDHWC_BFP16_BFP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_3d_NDHWC_FP32_FP32, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_3d_NDHWC_FP16_FP16, Test) { Run(); }
TEST_P(GPU_reference_kernel_bww_3d_NDHWC_BFP16_BFP16, Test) { Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_2d_NCHW_FP32_FP32,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_2d_NCHW_FP16_FP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_2d_NCHW_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_2d_NCHW_I8_I32,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_2d_NCHW_I8_FP32,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_2d_NCHW_FP32_FP32,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_2d_NCHW_FP16_FP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_2d_NCHW_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_2d_NCHW_FP32_FP32,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_2d_NCHW_FP16_FP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_2d_NCHW_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_3d_NCDHW_FP32_FP32,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_3d_NCDHW_FP16_FP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_3d_NCDHW_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_3d_NCDHW_I8_I32,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_3d_NCDHW_I8_FP32,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_3d_NCDHW_FP32_FP32,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_3d_NCDHW_FP16_FP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_3d_NCDHW_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_3d_NCDHW_FP32_FP32,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_3d_NCDHW_FP16_FP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_3d_NCDHW_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_2d_NHWC_FP32_FP32,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_2d_NHWC_FP16_FP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_2d_NHWC_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_2d_NHWC_I8_I32,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_2d_NHWC_I8_FP32,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_2d_NHWC_FP32_FP32,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_2d_NHWC_FP16_FP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_2d_NHWC_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_2d_NHWC_FP32_FP32,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_2d_NHWC_FP16_FP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_2d_NHWC_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases2D()),
                         [](const auto& info) { return NameGenerator(info); });

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_3d_NDHWC_FP32_FP32,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_3d_NDHWC_FP16_FP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_3d_NDHWC_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_3d_NDHWC_I8_I32,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_fwd_3d_NDHWC_I8_FP32,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_3d_NDHWC_FP32_FP32,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_3d_NDHWC_FP16_FP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bwd_3d_NDHWC_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_3d_NDHWC_FP32_FP32,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_3d_NDHWC_FP16_FP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_reference_kernel_bww_3d_NDHWC_BFP16_BFP16,
                         ::testing::ValuesIn(GenCases3D()),
                         [](const auto& info) { return NameGenerator(info); });
