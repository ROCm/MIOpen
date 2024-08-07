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

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_ENFORCE_DEVICE)

namespace {

// This is a simplified function, only one device is returned for the entire family.
std::string_view GpuTypeToStr(Gpu dev)
{
    switch(dev)
    {
    case Gpu::gfx900: return "gfx900";
    case Gpu::gfx906: return "gfx906";
    case Gpu::gfx908: return "gfx908";
    case Gpu::gfx90A: return "gfx90a";
    case Gpu::gfx94X: return "gfx940";
    case Gpu::gfx103X: return "gfx1030";
    case Gpu::gfx110X: return "gfx1100";
    case Gpu::gfx115X: return "gfx1150";
    default: throw std::runtime_error("unknown device");
    }
}

std::vector<Gpu> GetAllKnownGpusAsVector()
{
    std::vector<Gpu> gpu_v;
    for(Gpu gpu = Gpu::gfxFirst; gpu <= Gpu::gfxLast; gpu <<= 1)
    {
        gpu_v.push_back(gpu);
    }
    return gpu_v;
}

bool IsGpuSupported(Gpu supported_gpus, Gpu gpu)
{
    if((supported_gpus & gpu) != Gpu::None)
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

miopen::Handle EnforceDevice(Gpu gpu)
{
    const auto gpu_str = GpuTypeToStr(gpu);
    env::update(MIOPEN_DEBUG_ENFORCE_DEVICE, gpu_str);

    auto handle = miopen::Handle{};

    const auto name = handle.GetDeviceName();
    if(name != gpu_str)
        throw std::runtime_error("env::update failure");

    return handle;
}

} // namespace

void UnitTestConvSolverDevApplicabilityBase::RunTestImpl(
    const miopen::solver::conv::ConvSolverBase& solver,
    Gpu supported_gpus,
    miopenDataType_t datatype,
    miopen::conv::Direction direction,
    const ConvTestCaseBase& conv_config)
{
    const auto problem = GetProblemDescription(datatype, direction, conv_config);

    const auto all_known_gpus = GetAllKnownGpusAsVector();
    for(auto gpu : all_known_gpus)
    {
        const auto supported = IsGpuSupported(supported_gpus, gpu);
        // std::cout << "Test " << GpuTypeToStr(gpu) << " (supported: " << supported << ")" <<
        // std::endl;

        auto handle    = EnforceDevice(gpu);
        const auto ctx = [&] {
            auto tmp = miopen::ExecutionContext{&handle};
            problem.SetupFloats(tmp);
            return tmp;
        }();

        const auto is_applicable = solver.IsApplicable(ctx, problem);
        // std::cout << "IsApplicable: " << is_applicable << std::endl;
        if(is_applicable != supported)
        {
            GTEST_FAIL() << GpuTypeToStr(gpu) << " is" << (is_applicable ? "" : " not")
                         << " applicable for " << solver.SolverDbId() << " but "
                         << (supported ? "" : "not ") << "marked as supported";
        }
    }
}
