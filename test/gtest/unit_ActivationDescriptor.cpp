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

#include <algorithm>
#include <future>

#include <miopen/activ.hpp>
#include <miopen/miopen.h>
#include <miopen/handle.hpp>

#include "activ_common.hpp"
#include "gtest_common.hpp"

#include "../network_data.hpp"
#include "../random.hpp"
#include "../tensor_holder.hpp"
#include "../tensor_util.hpp"
#include "../verify.hpp"

#define UNIT_ACTIVATION_DESCRIPTOR_DEBUG 0

namespace {

std::set<std::vector<int>> GetTestCases(bool full)
{
    if(full)
        return get_inputs(0);
    else
        return {{1, 16, 8, 8}};
}

auto GetAllActivationModes()
{
    return std::vector<miopenActivationMode_t>{
        miopenActivationPASTHRU,
        miopenActivationLOGISTIC,
        miopenActivationTANH,
        miopenActivationRELU,
        miopenActivationSOFTRELU,
        miopenActivationABS,
        miopenActivationPOWER,
        miopenActivationCLIPPEDRELU,
        miopenActivationLEAKYRELU,
        miopenActivationELU,
    };
}

Gpu GetSupportedDevices() { return Gpu::All; }

template <class T>
class UnitTestActivationDescriptor
    : public ::testing::TestWithParam<std::tuple<Gpu, miopenActivationMode_t, std::vector<int>>>
{
public:
    void RunTest()
    {
        miopenActivationMode_t mode;
        std::vector<int> input_dims;
        std::tie(std::ignore, mode, input_dims) = GetParam();

        const double alpha = 0.95;
        const double beta  = 2.3;
        const double gamma = 3.4;
        const auto desc    = miopen::ActivationDescriptor{mode, alpha, beta, gamma};

        auto x_tensor      = tensor<T>{input_dims};
        auto y_tensor_gpu  = tensor<T>{input_dims};
        auto y_tensor_cpu  = tensor<T>{input_dims};
        auto dx_tensor_gpu = tensor<T>{input_dims};
        auto dy_tensor     = tensor<T>{input_dims};
        auto dx_tensor_cpu = tensor<T>{input_dims};

        auto x_data_ready      = std::async(std::launch::async, [&]() { GenData_x(x_tensor); });
        auto y_data_gpu_ready  = std::async(std::launch::async, [&]() { ZeroData(y_tensor_gpu); });
        auto dx_data_gpu_ready = std::async(std::launch::async, [&]() { ZeroData(dx_tensor_gpu); });
        auto dy_data_ready     = std::async(std::launch::async, [&]() { GenData_dy(dy_tensor); });

        x_data_ready.wait();
        y_data_gpu_ready.wait();
        dx_data_gpu_ready.wait();
        dy_data_ready.wait();

        std::promise<void> fwd_gpu_ready;
        std::promise<void> fwd_cpu_ready;
        auto gpu_ready = std::async(std::launch::async, [&]() {
            RunGpu(desc, x_tensor, y_tensor_gpu, dx_tensor_gpu, dy_tensor, fwd_gpu_ready);
        });
        auto cpu_ready = std::async(std::launch::async, [&]() {
            RunCpu(desc, x_tensor, y_tensor_cpu, dx_tensor_cpu, dy_tensor, fwd_cpu_ready);
        });

        fwd_gpu_ready.get_future().wait();
        fwd_cpu_ready.get_future().wait();

        auto verify_fwd_ready = std::async(std::launch::async, [&]() {
            return VerifyDataAsync(y_tensor_gpu, y_tensor_cpu, true);
        });

        gpu_ready.wait();
        cpu_ready.wait();

        auto verify_bwd_ready = std::async(std::launch::async, [&]() {
            return VerifyDataAsync(dx_tensor_gpu, dx_tensor_cpu, false);
        });

        if(!verify_fwd_ready.get())
        {
#if UNIT_ACTIVATION_DESCRIPTOR_DEBUG
            DebugPrintTensorsForward(x_tensor, y_tensor_cpu, y_tensor_gpu);
#endif
            GTEST_FAIL();
        }

        if(!verify_bwd_ready.get())
        {
#if UNIT_ACTIVATION_DESCRIPTOR_DEBUG
            DebugPrintTensorsBackward(
                x_tensor, y_tensor_gpu, dy_tensor, dx_tensor_cpu, dx_tensor_gpu);
#endif
            GTEST_FAIL();
        }
    }

protected:
    static void
    DebugPrintTensorsForward(const tensor<T>& x, const tensor<T>& y_cpu, const tensor<T>& y_gpu)
    {
        print_tensor(x, "X_TENSOR");
        print_tensor(y_cpu, "Y_TENSOR_CPU");
        print_tensor(y_gpu, "Y_TENSOR_GPU");
    }

    static void DebugPrintTensorsBackward(const tensor<T>& x,
                                          const tensor<T>& y_gpu,
                                          const tensor<T>& dy,
                                          const tensor<T>& dx_cpu,
                                          const tensor<T>& dx_gpu)
    {
        print_tensor(x, "X_TENSOR");
        print_tensor(y_gpu, "Y_TENSOR_GPU");
        print_tensor(dy, "dY_TENSOR");
        print_tensor(dx_cpu, "dX_TENSOR_CPU");
        print_tensor(dx_gpu, "dX_TENSOR_GPU");
    }

    /// \note In the original test, support for non-packed tensors was implemented in a very strange
    /// way (by incrementing the stride of the last dimension), in this reincarnation, support for
    /// non-packed tensors is not yet implemented. This warning is provided to help the developer
    /// when adding this missing functionality.
    static void NonPackedTensorWarning()
    {
        std::cout << "WARNING: Non-packed tensor. Some modifications are needed to achieve optimal "
                     "performance."
                  << std::endl;
    }

    static void ZeroData(tensor<T>& tensor)
    {
        if(!tensor.desc.IsPacked())
            NonPackedTensorWarning();

        std::fill(tensor.begin(), tensor.end(), T());
    }

    template <class G>
    static void GenData(tensor<T>& tensor, G g)
    {
        if(!tensor.desc.IsPacked())
            NonPackedTensorWarning();

        std::for_each(tensor.begin(), tensor.end(), g);
    }

    static void GenData_x(tensor<T>& tensor)
    {
        auto gen_x_value = [](auto& v) {
            v = prng::gen_A_to_B(static_cast<T>(-2), static_cast<T>(2));
        };
        GenData(tensor, gen_x_value);
    }

    static void GenData_dy(tensor<T>& tensor)
    {
        auto gen_dy_value = [](auto& v) {
            v = prng::gen_A_to_B(static_cast<T>(-0.5), static_cast<T>(0.5));
        };
        GenData(tensor, gen_dy_value);
    }

    static void RunGpu(const miopen::ActivationDescriptor& desc,
                       const tensor<T>& x,
                       tensor<T>& y,
                       tensor<T>& dx,
                       const tensor<T>& dy,
                       std::promise<void>& forward_ready)
    {
        miopenStatus_t status;

        // Dummy values
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        auto handle = miopen::Handle{};

        auto x_dev = handle.Write(x.data);
        auto y_dev = handle.Write(y.data);

        status = desc.Forward(handle, &alpha, x.desc, x_dev.get(), &beta, y.desc, y_dev.get());
        if(status != miopenStatusSuccess)
            throw std::runtime_error("Forward failed");

        y.data = handle.Read<T>(y_dev, y.data.size());

        forward_ready.set_value();

        auto dx_dev = handle.Write(dx.data);
        auto dy_dev = handle.Write(dy.data);

        status = desc.Backward(handle,
                               &alpha,
                               y.desc,
                               y_dev.get(),
                               dy.desc,
                               dy_dev.get(),
                               x.desc,
                               x_dev.get(),
                               &beta,
                               dx.desc,
                               dx_dev.get());
        if(status != miopenStatusSuccess)
            throw std::runtime_error("Backward failed");

        dx.data = handle.Read<T>(dx_dev, dx.data.size());
    }

    static void RunCpu(const miopen::ActivationDescriptor& desc,
                       const tensor<T>& x,
                       tensor<T>& y,
                       tensor<T>& dx,
                       const tensor<T>& dy,
                       std::promise<void>& forward_ready)
    {
        miopen::tests::activ::CpuActivationForward(
            desc.GetMode(), desc.GetAlpha(), desc.GetBeta(), desc.GetGamma(), x, y);

        forward_ready.set_value();

        miopen::tests::activ::CpuActivationBackward(
            desc.GetMode(), desc.GetAlpha(), desc.GetBeta(), desc.GetGamma(), y, dy, x, dx);
    }

    static double GetThreshold()
    {
        double tolerance = 50.0;
        double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        return threshold;
    }

    static bool
    VerifyDataAsync(const tensor<T>& tensor_gpu, const tensor<T>& tensor_cpu, bool forward)
    {
        if(!(tensor_gpu.desc.IsPacked() && tensor_cpu.desc.IsPacked()))
            NonPackedTensorWarning();

        const auto& cpu_data        = tensor_cpu.data;
        const auto& gpu_data        = tensor_gpu.data;
        const std::string direction = forward ? "(forward)" : "(backward)";

        auto check_gpu_zero = [&]() {
            auto res = miopen::range_zero(gpu_data);
            if(res)
            {
                std::cout << "GPU data is all zeros " << direction << std::endl;
                return false;
            }
            return true;
        };
        auto check_cpu_finite = [&]() {
            auto res = miopen::find_idx(cpu_data, miopen::not_finite);
            if(res != -1)
            {
                std::cout << "Non finite number found in the CPU data " << direction << std::endl;
                return false;
            }
            return true;
        };
        auto check_gpu_finite = [&]() {
            auto res = miopen::find_idx(gpu_data, miopen::not_finite);
            if(res != -1)
            {
                std::cout << "Non finite number found in the GPU data " << direction << std::endl;
                return false;
            }
            return true;
        };
        auto check_range_distance = [&]() {
            if(miopen::range_distance(cpu_data) != miopen::range_distance(gpu_data))
            {
                std::cout << "Range distance mismatch " << direction << std::endl;
                return false;
            }
            return true;
        };
        auto check_error = [&]() {
            const auto error       = miopen::rms_range(cpu_data, gpu_data);
            const double threshold = GetThreshold();
            if(error >= threshold)
            {
                std::cout << "Error beyond tolerance, error=" << error
                          << ", threshold=" << threshold << " " << direction << std::endl;
                return false;
            }
#if 0
            std::cout << "error: " << error << " threshold: " << threshold << " " << direction << std::endl;
#endif
            return true;
        };

        if(!check_range_distance())
            return false;

        auto check1 = std::async(std::launch::async, check_gpu_zero);
        auto check2 = std::async(std::launch::async, check_cpu_finite);
        auto check3 = std::async(std::launch::async, check_gpu_finite);
        auto check4 = std::async(std::launch::async, check_error);

        check1.wait();
        check2.wait();
        check3.wait();
        check4.wait();

        return check1.get() && check2.get() && check3.get() && check4.get();
    }

    void SetUp() override
    {
        Gpu supported_devs;
        std::tie(supported_devs, std::ignore, std::ignore) = GetParam();

        if(!IsTestSupportedByDevice(supported_devs))
        {
            GTEST_SKIP();
        }
    }
};

} // namespace

using GPU_UnitTestActivationDescriptor_FP16 = UnitTestActivationDescriptor<half_float::half>;
using GPU_UnitTestActivationDescriptor_FP32 = UnitTestActivationDescriptor<float>;

TEST_P(GPU_UnitTestActivationDescriptor_FP16, Activation) { this->RunTest(); };

TEST_P(GPU_UnitTestActivationDescriptor_FP32, Activation) { this->RunTest(); };

// Smoke tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestActivationDescriptor_FP16,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::ValuesIn(GetAllActivationModes()),
                                          testing::ValuesIn(GetTestCases(false))));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_UnitTestActivationDescriptor_FP32,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::ValuesIn(GetAllActivationModes()),
                                          testing::ValuesIn(GetTestCases(false))));

// Full tests
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestActivationDescriptor_FP16,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::ValuesIn(GetAllActivationModes()),
                                          testing::ValuesIn(GetTestCases(true))));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_UnitTestActivationDescriptor_FP32,
                         testing::Combine(testing::Values(GetSupportedDevices()),
                                          testing::ValuesIn(GetAllActivationModes()),
                                          testing::ValuesIn(GetTestCases(true))));
