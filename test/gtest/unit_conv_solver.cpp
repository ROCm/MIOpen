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

#include "unit_conv_solver.hpp"

namespace {

struct DevDescription
{
    std::string_view name;
    unsigned cu_cnt; // CU for gfx9, WGP for gfx10, 11, ...
};

// Add additional methods here if needed
class MockHandle : public miopen::Handle
{
public:
    MockHandle(const DevDescription& dev_description) : miopen::Handle{}, dev_descr{dev_description}
    {
    }

    std::string GetDeviceName() const override { return std::string{dev_descr.name}; }
    std::size_t GetMaxComputeUnits() const override { return dev_descr.cu_cnt; }
    bool CooperativeLaunchSupported() const override { return false; }

private:
    DevDescription dev_descr;
};

// This is a simplified function, only one device is returned for the entire family.
const auto& GetAllKnownDevices()
{
    static_assert(Gpu::gfx115X == Gpu::gfxLast);

    // https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
    static std::map<Gpu, DevDescription> known_devs = {
        // clang-format off
        {Gpu::gfx900,  {"gfx900",  64}},
        {Gpu::gfx906,  {"gfx906",  60}},
        {Gpu::gfx908,  {"gfx908",  120}},
        {Gpu::gfx90A,  {"gfx90a",  104}},
        {Gpu::gfx94X,  {"gfx941",  304}},
        {Gpu::gfx103X, {"gfx1030", 40}},
        {Gpu::gfx110X, {"gfx1100", 48}},
        {Gpu::gfx115X, {"gfx1150", 8/*???*/}},
        // clang-format on
    };
    return known_devs;
}

bool IsDeviceSupported(Gpu supported_devs, Gpu dev)
{
    if((supported_devs & dev) != Gpu::None)
        return true;
    return false;
}

miopen::conv::ProblemDescription GetProblemDescription(miopenDataType_t datatype,
                                                       miopen::conv::Direction direction,
                                                       const ConvTestCaseBase& conv_config)
{
    const auto inp_desc  = miopen::TensorDescriptor{datatype, conv_config.GetInput()};
    const auto wei_desc  = miopen::TensorDescriptor{datatype, conv_config.GetWeights()};
    const auto conv_desc = conv_config.GetConv();
    const auto out_desc  = conv_desc.GetForwardOutputTensor(inp_desc, wei_desc, datatype);

    switch(direction)
    {
    case miopen::conv::Direction::Forward:
    case miopen::conv::Direction::BackwardData:
        return miopen::conv::ProblemDescription(inp_desc, wei_desc, out_desc, conv_desc, direction);
    case miopen::conv::Direction::BackwardWeights:
        return miopen::conv::ProblemDescription(out_desc, wei_desc, inp_desc, conv_desc, direction);
    default: throw std::runtime_error("unknown direction");
    }
}

} // namespace

void UnitTestConvSolverDevApplicabilityBase::RunTestImpl(
    const miopen::solver::conv::ConvSolverBase& solver,
    Gpu supported_devs,
    miopenDataType_t datatype,
    miopen::conv::Direction direction,
    const ConvTestCaseBase& conv_config)
{
    const auto problem = GetProblemDescription(datatype, direction, conv_config);

    const auto all_known_devs = GetAllKnownDevices();
    for(const auto& [dev, dev_descr] : all_known_devs)
    {
        const auto supported = IsDeviceSupported(supported_devs, dev);
        // std::cout << "Test " << dev_descr.name << " (supported: " << supported << ")" <<
        // std::endl;

        auto handle    = MockHandle{dev_descr};
        const auto ctx = [&] {
            auto tmp = miopen::ExecutionContext{&handle};
            problem.SetupFloats(tmp);
            return tmp;
        }();

        const auto is_applicable = solver.IsApplicable(ctx, problem);
        // std::cout << "IsApplicable: " << is_applicable << std::endl;
        if(is_applicable != supported)
        {
            GTEST_FAIL() << dev_descr.name << " is" << (is_applicable ? "" : " not")
                         << " applicable for " << solver.SolverDbId() << " but "
                         << (supported ? "" : "not ") << "marked as supported";
        }
    }
}
