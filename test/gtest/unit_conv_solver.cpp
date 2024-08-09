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

class MockHandle : public miopen::Handle
{
public:
    MockHandle(const DevDescription& dev_description) : miopen::Handle{}, dev_descr{dev_description}
    {
    }

    // Add additional methods here if needed
    std::string GetDeviceName() const override { return std::string{dev_descr.name}; }
    std::size_t GetMaxComputeUnits() const override { return dev_descr.cu_cnt; }
    bool CooperativeLaunchSupported() const override { return false; }

private:
    DevDescription dev_descr;
};

// This is a simplified function, only one device is returned for the entire family.
const auto& GetAllKnownDevices()
{
    static_assert(Gpu::gfx110X == Gpu::gfxLast);

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

template <class T>
std::string VecToStr(const std::vector<T>& vec)
{
    bool first = true;
    std::ostringstream ss;

    ss << "{";
    for(auto val : vec)
    {
        if(first)
            first = false;
        else
            ss << ",";
        ss << val;
    }
    ss << "}";

    return ss.str();
}

} // namespace

// ConvTestCase

ConvTestCase::ConvTestCase(const std::initializer_list<size_t>& x_,
                           const std::initializer_list<size_t>& w_,
                           const std::initializer_list<int>& pad_,
                           const std::initializer_list<int>& stride_,
                           const std::initializer_list<int>& dilation_,
                           miopenDataType_t type_)
    : ConvTestCase(x_, w_, pad_, stride_, dilation_, type_, type_, type_)
{
}

ConvTestCase::ConvTestCase(const std::initializer_list<size_t>& x_,
                           const std::initializer_list<size_t>& w_,
                           const std::initializer_list<int>& pad_,
                           const std::initializer_list<int>& stride_,
                           const std::initializer_list<int>& dilation_,
                           miopenDataType_t type_x_,
                           miopenDataType_t type_w_,
                           miopenDataType_t type_y_)
    : x(x_),
      w(w_),
      pad(pad_),
      stride(stride_),
      dilation(dilation_),
      type_x(type_x_),
      type_w(type_w_),
      type_y(type_y_)
{
    const auto num_dims        = pad.size();
    const auto num_tensor_dims = num_dims + 2;

    if(x.size() != num_tensor_dims || w.size() != num_tensor_dims || stride.size() != num_dims ||
       dilation.size() != num_dims || x[1] != w[1])
    {
        throw std::runtime_error("wrong test case format");
    }
}

const std::vector<size_t>& ConvTestCase::GetXDims() const { return x; }

const std::vector<size_t>& ConvTestCase::GetWDims() const { return w; }

miopenDataType_t ConvTestCase::GetXDataType() const { return type_x; }

miopenDataType_t ConvTestCase::GetWDataType() const { return type_w; }

miopenDataType_t ConvTestCase::GetYDataType() const { return type_y; }

miopen::ConvolutionDescriptor ConvTestCase::GetConv() const
{
    return miopen::ConvolutionDescriptor{pad, stride, dilation};
}

std::ostream& operator<<(std::ostream& os, const ConvTestCase& tc)
{
    return os << "(x:" << VecToStr(tc.x) << " w:" << VecToStr(tc.w) << " pad:" << VecToStr(tc.pad)
              << " stride:" << VecToStr(tc.stride) << " dilation:" << VecToStr(tc.dilation)
              << " type_x:" << tc.type_x << " type_w:" << tc.type_w << " type_y:" << tc.type_y
              << ")";
}

// This test is designed to detect the expansion of the solver's device applicability

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
