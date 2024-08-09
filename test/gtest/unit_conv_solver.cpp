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

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>

#include "unit_conv_solver.hpp"

#include "get_handle.hpp"
#include "conv_common.hpp"
#include "conv_tensor_gen.hpp"
#include "tensor_holder.hpp"

#include "../workspace.hpp"

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

miopen::conv::ProblemDescription GetProblemDescription(miopen::conv::Direction direction,
                                                       const ConvTestCase& conv_config)
{
    const auto x_desc =
        miopen::TensorDescriptor{conv_config.GetXDataType(), conv_config.GetXDims()};
    const auto w_desc =
        miopen::TensorDescriptor{conv_config.GetWDataType(), conv_config.GetWDims()};
    const auto conv_desc = conv_config.GetConv();
    const auto y_desc =
        conv_desc.GetForwardOutputTensor(x_desc, w_desc, conv_config.GetYDataType());

    switch(direction)
    {
    case miopen::conv::Direction::Forward:
    case miopen::conv::Direction::BackwardData:
        return miopen::conv::ProblemDescription(x_desc, w_desc, y_desc, conv_desc, direction);
    case miopen::conv::Direction::BackwardWeights:
        return miopen::conv::ProblemDescription(y_desc, w_desc, x_desc, conv_desc, direction);
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

//************************************************************************************
// ConvTestCase
//************************************************************************************

ConvTestCase::ConvTestCase() : type_x(miopenFloat), type_w(miopenFloat), type_y(miopenFloat) {}

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

//************************************************************************************
// Unit test for convolution solver
//************************************************************************************

namespace {

//**********************************
// Fwd
//**********************************
template <typename T = float, typename Tref = float>
void RunSolverFwd(const miopen::solver::conv::ConvSolverBase& solv,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo,
                  bool use_cpu_ref)
{
    //**********************************
    // Prepare
    //**********************************

    auto input   = tensor<T>{conv_config.GetXDims()};
    auto weights = tensor<T>{conv_config.GetWDims()};
    input.generate(GenData<T>{});
    weights.generate(GenWeights<T>{});

    const auto conv_desc = conv_config.GetConv();

    const auto output_desc =
        conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});

    auto output = tensor<T>{output_desc.GetLengths()};
    std::fill(output.begin(), output.end(), T(0));

    auto&& handle = get_handle();
    auto in_dev   = handle.Write(input.data);
    auto wei_dev  = handle.Write(weights.data);
    auto out_dev  = handle.Write(output.data);

    //**********************************
    // Run solver
    //**********************************

    const auto tensors = miopen::ConvFwdTensors{
        input.desc, in_dev.get(), weights.desc, wei_dev.get(), output.desc, out_dev.get()};

    const auto problem = miopen::conv::ProblemDescription(
        input.desc, weights.desc, output.desc, conv_desc, miopen::conv::Direction::Forward);
    const auto ctx = [&] {
        auto tmp = miopen::ExecutionContext{&handle};
        problem.SetupFloats(tmp);
        return tmp;
    }();

    if(!solv.IsApplicable(ctx, problem))
    {
        // Do not put GTEST_SKIP here.
        // The usage of non-applicable config should be considered as a bug in the test.
        GTEST_FAIL();
    }

    Workspace wspace;
    if(solv.MayNeedWorkspace())
    {
        const auto cur_sol_ws = solv.GetWorkspaceSize(ctx, problem);
        wspace.resize(cur_sol_ws);
    }

    const auto invoke_params = miopen::conv::DataInvokeParams{
        tensors, wspace.ptr(), wspace.size(), conv_desc.attribute.gfx90aFp16alt.GetFwd()};

    // \todo add path for tunable solvers
    const auto& conv_solv = dynamic_cast<const miopen::solver::conv::ConvSolver&>(solv);

    const auto sol = conv_solv.GetSolution(ctx, problem);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();

    //**********************************
    // Verify
    //**********************************

    auto ref_out = tensor<Tref>{output.desc.GetLayout_t(), output.desc.GetLengths()};
    if(use_cpu_ref)
    {
        cpu_convolution_forward(conv_desc.GetSpatialDimension(),
                                input,
                                weights,
                                ref_out,
                                conv_desc.GetConvPads(),
                                conv_desc.GetConvStrides(),
                                conv_desc.GetConvDilations(),
                                conv_desc.GetGroupCount());
    }
    else
    {
        ref_out = ref_conv_fwd(input, weights, ref_out, conv_desc);
    }

    output.data = handle.Read<T>(out_dev, output.data.size());

    ASSERT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
    ASSERT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
    ASSERT_EQ(miopen::range_distance(ref_out), miopen::range_distance(output));

    const double tolerance = 5;
    double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
    auto error             = miopen::rms_range(ref_out, output);

    ASSERT_LT(miopen::find_idx(ref_out, miopen::not_finite), 0)
        << "Non finite number found in the CPU data";

    ASSERT_LT(error, threshold) << "Error beyond tolerance Error";
    // std::cout << "error: " << error << " threshold: " << threshold << std::endl;
}

//**********************************
// Bwd
//**********************************
template <typename T = float, typename Tref = float>
void RunSolverBwd(const miopen::solver::conv::ConvSolverBase& solv,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo,
                  bool use_cpu_ref)
{
    //**********************************
    // Prepare
    //**********************************

    auto input   = tensor<T>{conv_config.GetXDims()};
    auto weights = tensor<T>{conv_config.GetWDims()};
    weights.generate(GenWeights<T>{});

    const auto conv_desc = conv_config.GetConv();

    const auto output_desc =
        conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});

    auto output = tensor<T>{output_desc.GetLengths()};
    output.generate(GenData<T>{});

    std::fill(input.begin(), input.end(), T(0));

    auto&& handle = get_handle();
    auto in_dev   = handle.Write(input.data);
    auto wei_dev  = handle.Write(weights.data);
    auto out_dev  = handle.Write(output.data);

    //**********************************
    // Run solver
    //**********************************

    const auto tensors = miopen::ConvBwdTensors{
        output.desc, out_dev.get(), weights.desc, wei_dev.get(), input.desc, in_dev.get()};

    const auto problem = miopen::conv::ProblemDescription(
        input.desc, weights.desc, output.desc, conv_desc, miopen::conv::Direction::BackwardData);
    const auto ctx = [&] {
        auto tmp = miopen::ExecutionContext{&handle};
        problem.SetupFloats(tmp);
        return tmp;
    }();

    if(!solv.IsApplicable(ctx, problem))
    {
        // Do not put GTEST_SKIP here.
        // The usage of non-applicable config should be considered as a bug in the test.
        GTEST_FAIL();
    }

    Workspace wspace;
    if(solv.MayNeedWorkspace())
    {
        const auto cur_sol_ws = solv.GetWorkspaceSize(ctx, problem);
        wspace.resize(cur_sol_ws);
    }

    const auto invoke_params = miopen::conv::DataInvokeParams{
        tensors, wspace.ptr(), wspace.size(), conv_desc.attribute.gfx90aFp16alt.GetBwd()};

    // \todo add path for tunable solvers
    const auto& conv_solv = dynamic_cast<const miopen::solver::conv::ConvSolver&>(solv);

    const auto sol = conv_solv.GetSolution(ctx, problem);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();

    //**********************************
    // Verify
    //**********************************

    auto ref_in = tensor<Tref>{input.desc.GetLengths()};
    if(use_cpu_ref)
    {
        cpu_convolution_backward_data(conv_desc.GetSpatialDimension(),
                                      ref_in,
                                      weights,
                                      output,
                                      conv_desc.GetConvPads(),
                                      conv_desc.GetConvStrides(),
                                      conv_desc.GetConvDilations(),
                                      conv_desc.GetGroupCount());
    }
    else
    {
        ref_in = ref_conv_bwd(ref_in, weights, output, conv_desc);
    }

    input.data = handle.Read<T>(in_dev, input.data.size());

    ASSERT_FALSE(miopen::range_zero(ref_in)) << "Cpu data is all zeros";
    ASSERT_FALSE(miopen::range_zero(input)) << "Gpu data is all zeros";
    ASSERT_EQ(miopen::range_distance(ref_in), miopen::range_distance(input));

    const double tolerance = 5;
    double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
    auto error             = miopen::rms_range(ref_in, input);

    ASSERT_LT(miopen::find_idx(ref_in, miopen::not_finite), 0)
        << "Non finite number found in the CPU data";

    ASSERT_LT(error, threshold) << "Error beyond tolerance Error:";
    // std::cout << "error: " << error << " threshold: " << threshold << std::endl;
}

//**********************************
// Wrw
//**********************************
template <typename T = float, typename Tref = float>
void RunSolverWrw(const miopen::solver::conv::ConvSolverBase& solv,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo,
                  bool use_cpu_ref)
{
    //**********************************
    // Prepare
    //**********************************

    auto input   = tensor<T>{conv_config.GetXDims()};
    auto weights = tensor<T>{conv_config.GetWDims()};
    input.generate(GenData<T>{});

    const auto conv_desc = conv_config.GetConv();

    const auto output_desc =
        conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});

    auto output = tensor<T>{output_desc.GetLengths()};
    output.generate(GenData<T>{});

    std::fill(weights.begin(), weights.end(), T(0));

    auto&& handle = get_handle();
    auto in_dev   = handle.Write(input.data);
    auto wei_dev  = handle.Write(weights.data);
    auto out_dev  = handle.Write(output.data);

    //**********************************
    // Run solver
    //**********************************

    const auto tensors = miopen::ConvWrwTensors{
        output.desc, out_dev.get(), input.desc, in_dev.get(), weights.desc, wei_dev.get()};

    const auto problem = miopen::conv::ProblemDescription(
        output.desc, weights.desc, input.desc, conv_desc, miopen::conv::Direction::BackwardWeights);
    const auto ctx = [&] {
        auto tmp = miopen::ExecutionContext{&handle};
        problem.SetupFloats(tmp);
        return tmp;
    }();

    if(!solv.IsApplicable(ctx, problem))
    {
        // Do not put GTEST_SKIP here.
        // The usage of non-applicable config should be considered as a bug in the test.
        GTEST_FAIL();
    }

    Workspace wspace;
    if(solv.MayNeedWorkspace())
    {
        const auto cur_sol_ws = solv.GetWorkspaceSize(ctx, problem);
        wspace.resize(cur_sol_ws);
    }

    const auto invoke_params = miopen::conv::WrWInvokeParams{
        tensors, wspace.ptr(), wspace.size(), conv_desc.attribute.gfx90aFp16alt.GetWrW()};

    // \todo add path for tunable solvers
    const auto& conv_solv = dynamic_cast<const miopen::solver::conv::ConvSolver&>(solv);

    const auto sol = conv_solv.GetSolution(ctx, problem);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();

    //**********************************
    // Verify
    //**********************************

    auto ref_weights = tensor<Tref>{weights.desc.GetLengths()};
    if(use_cpu_ref)
    {
        cpu_convolution_backward_weight(conv_desc.GetSpatialDimension(),
                                        input,
                                        ref_weights,
                                        output,
                                        conv_desc.GetConvPads(),
                                        conv_desc.GetConvStrides(),
                                        conv_desc.GetConvDilations(),
                                        conv_desc.GetGroupCount());
    }
    else
    {
        ref_weights = ref_conv_wrw(input, ref_weights, output, conv_desc);
    }

    weights.data = handle.Read<T>(wei_dev, weights.data.size());

    ASSERT_FALSE(miopen::range_zero(ref_weights)) << "Cpu data is all zeros";
    ASSERT_FALSE(miopen::range_zero(weights)) << "Gpu data is all zeros";
    ASSERT_EQ(miopen::range_distance(ref_weights), miopen::range_distance(weights));

    const double tolerance = 5;
    double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
    auto error             = miopen::rms_range(ref_weights, weights);

    ASSERT_LT(miopen::find_idx(ref_weights, miopen::not_finite), 0)
        << "Non finite number found in the CPU data";

    ASSERT_LT(error, threshold) << "Error beyond tolerance Error:";
    // std::cout << "error: " << error << " threshold: " << threshold << std::endl;
}

template <typename T = float, typename Tref = float>
void RunSolver(const miopen::solver::conv::ConvSolverBase& solver,
               miopen::conv::Direction direction,
               const ConvTestCase& conv_config,
               miopenConvAlgorithm_t algo,
               bool use_cpu_ref = false)
{
    switch(direction)
    {
    case miopen::conv::Direction::Forward:
        RunSolverFwd<T, Tref>(solver, conv_config, algo, use_cpu_ref);
        return;
    case miopen::conv::Direction::BackwardData:
        RunSolverBwd<T, Tref>(solver, conv_config, algo, use_cpu_ref);
        return;
    case miopen::conv::Direction::BackwardWeights:
        RunSolverWrw<T, Tref>(solver, conv_config, algo, use_cpu_ref);
        return;
    default: throw std::runtime_error("unknown direction");
    }
}

void RunSolver(const miopen::solver::conv::ConvSolverBase& solver,
               miopen::conv::Direction direction,
               const ConvTestCase& conv_config,
               miopenConvAlgorithm_t algo)
{
    if(conv_config.GetXDataType() == conv_config.GetWDataType() &&
       conv_config.GetWDataType() == conv_config.GetYDataType())
    {
        switch(conv_config.GetXDataType())
        {
        case miopenHalf:
            RunSolver<half_float::half, half_float::half>(solver, direction, conv_config, algo);
            return;
        case miopenFloat: RunSolver<float, float>(solver, direction, conv_config, algo); return;
        default: throw std::runtime_error("handling of this data type is not yet implemented");
        }
    }

    throw std::runtime_error("handling of mixed data types is not yet implemented");
}

} // namespace

void UnitTestConvSolverBase::SetUpImpl(Gpu supported_devs)
{
    if(!IsTestSupportedByDevice(supported_devs))
    {
        GTEST_SKIP();
    }
}

void UnitTestConvSolverBase::RunTestImpl(const miopen::solver::conv::ConvSolverBase& solver,
                                         miopen::conv::Direction direction,
                                         const ConvTestCase& conv_config,
                                         miopenConvAlgorithm_t algo)
{
    RunSolver(solver, direction, conv_config, algo);
}

//************************************************************************************
// This test is designed to detect the expansion of the solver's device applicability
//************************************************************************************

void UnitTestConvSolverDevApplicabilityBase::RunTestImpl(
    const miopen::solver::conv::ConvSolverBase& solver,
    Gpu supported_devs,
    miopen::conv::Direction direction,
    const ConvTestCase& conv_config)
{
    const auto problem = GetProblemDescription(direction, conv_config);

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
