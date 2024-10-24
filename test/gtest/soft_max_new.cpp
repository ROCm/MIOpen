#include "test.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
#include <miopen/softmax.hpp>
#include <miopen/tensor.hpp>
#include <utility>
#include <algorithm>
#include <sstream>

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "random.hpp"

#include <cstdint>
#include <tuple>
#include <gtest/gtest.h>

#define NEGATIVE_CUTOFF_VAL_FP32 (-1e20)
#define NEGATIVE_CUTOFF_VAL_FP16 (-1e4)

struct InputDimension
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;

    friend std::ostream& operator<<(std::ostream& os, const InputDimension& input)
    {
        return os << "(N: " << input.N << " C:" << input.C << " H:" << input.H << " W:" << input.W
                  << ")";
    }
};

struct Scales
{
    float alpha;
    float beta;

    friend std::ostream& operator<<(std::ostream& os, const Scales& ab)
    {
        return os << "(alpha: " << ab.alpha << " beta:" << ab.beta << ")";
    }
};
template <typename T>
T logaddexp(T x, T y, T neg_inf)
{
    T a = std::max(x, y);
    T b = std::min(x, y);
    T c = b - a;

    return c <= neg_inf ? std::max(a, neg_inf) : std::max(T(a + log(T(1) + exp(b - a))), neg_inf);
}
template <typename DataType>
class SoftMaxTest
    : public ::testing::TestWithParam<
          std::tuple<InputDimension, miopenSoftmaxAlgorithm_t, miopenSoftmaxMode_t, Scales>>
{
public:
    static std::vector<InputDimension> convertDim(const std::set<std::vector<int>>& input)
    {

        std::vector<InputDimension> result;
        result.reserve(input.size());
        std::transform(input.begin(),
                       input.end(),
                       std::back_inserter(result),
                       [](const std::vector<int>& inp) {
                           return InputDimension{static_cast<size_t>(inp[0]),
                                                 static_cast<size_t>(inp[1]),
                                                 static_cast<size_t>(inp[2]),
                                                 static_cast<size_t>(inp[3])};
                       });
        return result;
    }

protected:
    void SetUp() override
    {
        std::tie(input_dim, algo, mode, scales) = GetParam();
        auto&& handle                           = get_handle();

        // alocate input
        input = tensor<DataType>{input_dim.N, input_dim.C, input_dim.H, input_dim.W};
        assert(input_dim.N > 0 && input_dim.C > 0 && input_dim.H > 0 && input_dim.W > 0);

        // Initialize input with random values using prng
        std::generate(input.begin(), input.end(), []() {
            return static_cast<DataType>(
                prng::gen_A_to_B(-5.0, 5.0)); // Generiši vrednosti u opsegu [-5, 5]
        });

        output = tensor<DataType>{input};

        auto output_dev = handle.Write(output.data);

        ref_out = tensor<DataType>{input};
        std::fill(ref_out.begin(), ref_out.end(), std::numeric_limits<DataType>::quiet_NaN());
        auto ref_out_dev = handle.Write(ref_out.data);

        // backward pass
        bw_input = tensor<DataType>{input};
        std::fill(bw_input.begin(), bw_input.end(), std::numeric_limits<DataType>::quiet_NaN());
        bw_output = tensor<DataType>{input};
        std::fill(bw_output.begin(), bw_output.end(), std::numeric_limits<DataType>::quiet_NaN());
        bw_doutput = tensor<DataType>{input}; // Inicijalizuj gradient tensor
        std::fill(bw_doutput.begin(), bw_doutput.end(), std::numeric_limits<DataType>::quiet_NaN());
    }

    tensor<DataType> RunForwardPassCPU() const
    {
        auto out   = output;
        auto alpha = scales.alpha;
        auto beta  = scales.beta;

        std::cout << std::endl;

        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(input.desc.GetLengths());

        int in_nstr, in_cstr, in_hstr;
        std::tie(in_nstr, in_cstr, in_hstr, std::ignore) = miopen::tien<4>(input.desc.GetStrides());

        int out_nstr, out_cstr, out_hstr;
        std::tie(out_nstr, out_cstr, out_hstr, std::ignore) =
            miopen::tien<4>(out.desc.GetStrides());

        if(mode == MIOPEN_SOFTMAX_MODE_INSTANCE)
        {
            par_ford(in_n)([&](int o) {
                if(algo == MIOPEN_SOFTMAX_FAST)
                {
                    double sum = 0;
                    ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        double val =
                            static_cast<double>(input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                        sum += std::exp(val);
                    });
                    ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        double val = std::exp(static_cast<double>(
                                         input[o * in_nstr + w * in_cstr + i * in_hstr + j])) /
                                     sum;
                        out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                            alpha * val +
                            beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                    });
                }
                else
                {
                    DataType max_c = std::numeric_limits<DataType>::lowest();
                    ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                        max_c = std::max(max_c, input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });

                    if(algo == MIOPEN_SOFTMAX_LOG)
                    {
                        double neg_inf = (input.desc.GetType() == miopenHalf)
                                             ? NEGATIVE_CUTOFF_VAL_FP16
                                             : NEGATIVE_CUTOFF_VAL_FP32;
                        double sum     = neg_inf;
                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            sum = logaddexp(
                                static_cast<double>(
                                    input[o * in_nstr + w * in_cstr + i * in_hstr + j] - max_c),
                                sum,
                                neg_inf);
                        });

                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha * (static_cast<double>(
                                             input[o * in_nstr + w * in_cstr + i * in_hstr + j]) -
                                         max_c - sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                    else
                    {
                        double sum = 0;
                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            sum += std::exp(static_cast<double>(
                                input[o * in_nstr + w * in_cstr + i * in_hstr + j] - max_c));
                        });

                        ford(in_c, in_h, in_w)([&](int w, int i, int j) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha * (std::exp(
                                            static_cast<double>(
                                                input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                                max_c) /
                                            sum)) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                }
            });
        }
        else
        {
            par_ford(in_n, in_h, in_w)([&](int o, int i, int j) {
                if(algo == MIOPEN_SOFTMAX_FAST)
                {
                    double sum = 0;
                    ford(in_c)([&](int w) {
                        sum += std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });
                    ford(in_c)([&](int w) {
                        out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                            alpha * (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j]) /
                                     sum) +
                            beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                    });
                }
                else
                {
                    DataType max_c = std::numeric_limits<DataType>::lowest();
                    ford(in_c)([&](int w) {
                        max_c = std::max(max_c, input[o * in_nstr + w * in_cstr + i * in_hstr + j]);
                    });

                    if(algo == MIOPEN_SOFTMAX_LOG)
                    {
                        double neg_inf = input.desc.GetType() == miopenHalf
                                             ? NEGATIVE_CUTOFF_VAL_FP16
                                             : NEGATIVE_CUTOFF_VAL_FP32;
                        double sum     = neg_inf;
                        ford(in_c)([&](int w) {
                            sum = logaddexp(
                                double(input[o * in_nstr + w * in_cstr + i * in_hstr + j] - max_c),
                                sum,
                                neg_inf);
                        });

                        ford(in_c)([&](int w) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha * (input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                         max_c - sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                    else
                    {
                        double sum = 0;
                        ford(in_c)([&](int w) {
                            sum += std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                            max_c);
                        });

                        ford(in_c)([&](int w) {
                            out[o * out_nstr + w * out_cstr + i * out_hstr + j] =
                                alpha *
                                    (std::exp(input[o * in_nstr + w * in_cstr + i * in_hstr + j] -
                                              max_c) /
                                     sum) +
                                beta * out[o * out_nstr + w * out_cstr + i * out_hstr + j];
                        });
                    }
                }
            });
        }
        return out;
    }

    tensor<DataType> RunForwardPassGPU()
    {
        auto&& handle = get_handle();
        auto out      = output;

        // Alokacija i kopiranje podataka za forward pass
        auto input_dev  = handle.Write(input.data);
        auto output_dev = handle.Write(output.data); // Ako trebaš referencu na output

        // Izvršavanje forward pass-a na GPU
        miopen::SoftmaxForward(handle,
                               &scales.alpha,
                               &scales.beta,
                               input.desc,
                               input_dev.get(),
                               output.desc,
                               output_dev.get(),
                               algo,
                               mode);

        out.data = handle.Read<DataType>(output_dev, out.data.size());

        return out;

        std::cout << "Forward pass GPU executed successfully." << std::endl;
    }
    void RunBackwardPass() // gpu
    {
        auto&& handle = get_handle();
        // auto din = dinput; //cpu tek u verify

        // auto din_dev  = handle.Write(din.data); //cpu
        auto bw_doutput_dev = handle.Write(input.data);
        // backward pass on GPU
        auto bw_input_dev  = handle.Write(bw_input.data);
        auto bw_output_dev = handle.Write(bw_output.data);
        // Execute backward pass on GPU
        miopen::SoftmaxBackward(handle,
                                &scales.alpha,
                                bw_output.desc,
                                bw_output_dev.get(),
                                bw_input.desc,
                                bw_input_dev.get(),
                                &scales.beta,
                                input.desc,
                                bw_doutput_dev.get(),
                                algo,
                                mode);

        std::cout << "Backward pass GPU executed successfully." << std::endl;
    }

    void CompareResults(const tensor<DataType>& cpu_output,
                        const tensor<DataType>& gpu_output,
                        float tolerance = 1e-6)
    {
        ASSERT_EQ(cpu_output.desc.GetElementSize(), gpu_output.desc.GetElementSize())
            << "CPU and GPU output sizes differ!";

        for(size_t i = 0; i < cpu_output.desc.GetElementSize(); ++i)
        {
            EXPECT_NEAR(cpu_output[i], gpu_output[i], tolerance) << "Mismatch at index: " << i;
        }
    }

    void Print()
    {
        // print input test parameters
        std::cout << "Run test:" << std::endl;
        std::cout << "Input Dimensions: " << input_dim << std::endl;
        std::cout << "Input values: " << std::endl;
        for(size_t i = 0; i < input.desc.GetElementSize(); ++i)
        {
            std::cout << input[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Softmax Algorithm: " << algo << std::endl;
        std::cout << "Softmax Mode: " << mode << std::endl;
        std::cout << "Scales: " << scales << std::endl;
    }

    void TearDown() override
    {
        // probably deallocate anything if required
    }

private:
    InputDimension input_dim;
    tensor<DataType> input;
    tensor<DataType> output;
    tensor<DataType> ref_out;

    tensor<DataType> bw_input;   // Backward input tensor
    tensor<DataType> bw_output;  // Backward output tensor
    tensor<DataType> bw_doutput; // Backward gardian output tensor

    miopenSoftmaxAlgorithm_t algo;
    miopenSoftmaxMode_t mode;
    Scales scales;
};

// Main test case
using GPU_SoftMax_Fwd_FP32 = SoftMaxTest<float>;

TEST_P(GPU_SoftMax_Fwd_FP32, Test)
{
    Print();

    auto cpu_output = RunForwardPassCPU();
    auto gpu_output = RunForwardPassGPU();

    // Poredi rezultate CPU i GPU
    CompareResults(cpu_output, gpu_output);

    RunForwardPassGPU();
    RunBackwardPass();
};

// Define fixed input dimensions for testing (this could be dynamic)
std::set<std::vector<int>> GetOutputDimensions()
{
    return {
        {1, 2, 3, 4},
        {4, 3, 2, 1},

    };
}
// Instantiate the test cases
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_SoftMax_Fwd_FP32,
    testing::Combine(
        testing::ValuesIn(GPU_SoftMax_Fwd_FP32::convertDim(GetOutputDimensions())),
        testing::Values(MIOPEN_SOFTMAX_FAST, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_LOG),
        testing::Values(MIOPEN_SOFTMAX_MODE_INSTANCE, MIOPEN_SOFTMAX_MODE_CHANNEL),
        testing::Values(Scales{1.0f, 0.0f}, Scales{0.5f, 0.5f})));
