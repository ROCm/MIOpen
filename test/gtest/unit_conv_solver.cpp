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
    const auto trans_output_pads = std::vector<int>(pad.size(), 0);
    return miopen::ConvolutionDescriptor{pad, stride, dilation, trans_output_pads};
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

template <typename T>
double GetThreshold(miopenConvAlgorithm_t algo, miopen::conv::Direction direction)
{
    double tolerance = 1.0;

    if constexpr(std::is_same_v<T, half_float::half>)
    {
        if(algo == miopenConvolutionAlgoGEMM && direction != miopen::conv::Direction::Forward)
        {
            tolerance *= 2.0;
        }
    }

    double threshold = std::numeric_limits<T>::epsilon() * tolerance;
    return threshold;
}

template <typename T, typename Tref>
void VerifyData(const std::vector<T>& data,
                const std::vector<Tref>& ref_data,
                miopenConvAlgorithm_t algo,
                miopen::conv::Direction direction)
{
    ASSERT_FALSE(miopen::range_zero(ref_data)) << "Reference data is all zeros";
    if constexpr(!std::is_integral_v<T>)
    {
        ASSERT_LT(miopen::find_idx(ref_data, miopen::not_finite), 0)
            << "Non finite number found in the reference data";
    }

    ASSERT_FALSE(miopen::range_zero(data)) << "Gpu data is all zeros";
    if constexpr(!std::is_integral_v<T>)
    {
        ASSERT_LT(miopen::find_idx(data, miopen::not_finite), 0)
            << "Non finite number found in the Gpu data";
    }

    ASSERT_EQ(miopen::range_distance(ref_data), miopen::range_distance(data));

    if constexpr(std::is_integral_v<T>)
    {
        const auto error = miopen::max_diff_v2(ref_data, data);
        static_assert(std::is_integral_v<decltype(error)>);
        ASSERT_EQ(error, 0) << "Error beyond tolerance";
    }
    else
    {
        const auto error       = miopen::rms_range(ref_data, data);
        const double threshold = GetThreshold<T>(algo, direction);
        ASSERT_LT(error, threshold) << "Error beyond tolerance";
        // std::cout << "error: " << error << " threshold: " << threshold << std::endl;
    }
}

//**********************************
// Fwd
//**********************************
template <typename Tin, typename Twei, typename Tout, typename Tref>
void RunSolverFwd(const miopen::solver::conv::ConvSolverBase& solv,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo,
                  bool use_cpu_ref)
{
    //**********************************
    // Prepare
    //**********************************

    auto input   = tensor<Tin>{conv_config.GetXDims()};
    auto weights = tensor<Twei>{conv_config.GetWDims()};

    const auto conv_desc = conv_config.GetConv();

    const auto output_desc =
        conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<Tout>{});

    auto output = tensor<Tout>{output_desc.GetLengths()};

    input.generate(GenConvData<Tin, Tout>{conv_config.GetWDims()});
    weights.generate(GenConvData<Twei, Tout>{conv_config.GetWDims()});
    std::fill(output.begin(), output.end(), Tout());

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

    auto ref_out = tensor<Tref>{output.desc.GetLengths()};
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

    output.data = handle.Read<Tout>(out_dev, output.data.size());

    VerifyData(output.data, ref_out.data, algo, miopen::conv::Direction::Forward);
}

template <typename T, typename Tref>
void RunSolverFwd(const miopen::solver::conv::ConvSolverBase& solv,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo,
                  bool use_cpu_ref)
{
    RunSolverFwd<T, T, T, Tref>(solv, conv_config, algo, use_cpu_ref);
}

//**********************************
// Bwd
//**********************************
template <typename Tin, typename Twei, typename Tout, typename Tref>
void RunSolverBwd(const miopen::solver::conv::ConvSolverBase& solv,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo,
                  bool use_cpu_ref)
{
    //**********************************
    // Prepare
    //**********************************

    auto input   = tensor<Tin>{conv_config.GetXDims()};
    auto weights = tensor<Twei>{conv_config.GetWDims()};

    const auto conv_desc = conv_config.GetConv();

    const auto output_desc =
        conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<Tout>{});

    auto output = tensor<Tout>{output_desc.GetLengths()};

    output.generate(GenConvData<Tout, Tin>{conv_config.GetWDims()});
    weights.generate(GenConvData<Twei, Tin>{conv_config.GetWDims()});
    std::fill(input.begin(), input.end(), Tin());

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
        output.desc, weights.desc, input.desc, conv_desc, miopen::conv::Direction::BackwardData);
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

    input.data = handle.Read<Tin>(in_dev, input.data.size());

    VerifyData(input.data, ref_in.data, algo, miopen::conv::Direction::BackwardData);
}

template <typename T, typename Tref>
void RunSolverBwd(const miopen::solver::conv::ConvSolverBase& solv,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo,
                  bool use_cpu_ref)
{
    RunSolverBwd<T, T, T, Tref>(solv, conv_config, algo, use_cpu_ref);
}

//**********************************
// Wrw
//**********************************
template <typename Tin, typename Twei, typename Tout, typename Tref>
void RunSolverWrw(const miopen::solver::conv::ConvSolverBase& solv,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo,
                  bool use_cpu_ref)
{
    //**********************************
    // Prepare
    //**********************************

    auto input   = tensor<Tin>{conv_config.GetXDims()};
    auto weights = tensor<Twei>{conv_config.GetWDims()};

    const auto conv_desc = conv_config.GetConv();

    const auto output_desc =
        conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<Tout>{});

    auto output = tensor<Tout>{output_desc.GetLengths()};

    input.generate(GenConvData<Tin, Twei>{output_desc.GetLengths()});
    output.generate(GenConvData<Tout, Twei>{output_desc.GetLengths()});
    std::fill(weights.begin(), weights.end(), Twei());

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

    weights.data = handle.Read<Twei>(wei_dev, weights.data.size());

    VerifyData(weights.data, ref_weights.data, algo, miopen::conv::Direction::BackwardWeights);
}

template <typename T, typename Tref>
void RunSolverWrw(const miopen::solver::conv::ConvSolverBase& solv,
                  const ConvTestCase& conv_config,
                  miopenConvAlgorithm_t algo,
                  bool use_cpu_ref)
{
    RunSolverWrw<T, T, T, Tref>(solv, conv_config, algo, use_cpu_ref);
}

template <typename T, typename Tref>
void RunSolver(const miopen::solver::conv::ConvSolverBase& solver,
               miopen::conv::Direction direction,
               const ConvTestCase& conv_config,
               miopenConvAlgorithm_t algo,
               bool use_cpu_ref)
{
    // clang-format off
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
    default:
        throw std::runtime_error("unknown direction");
    }
    // clang-format on
}

void RunSolver(const miopen::solver::conv::ConvSolverBase& solver,
               miopen::conv::Direction direction,
               const ConvTestCase& conv_config,
               miopenConvAlgorithm_t algo,
               bool use_cpu_ref = false)
{
    if(conv_config.GetXDataType() == conv_config.GetWDataType() &&
       conv_config.GetWDataType() == conv_config.GetYDataType())
    {
        // clang-format off
        switch(conv_config.GetXDataType())
        {
        case miopenHalf:
            RunSolver<half_float::half, half_float::half>(solver, direction, conv_config, algo, use_cpu_ref);
            return;
        case miopenFloat:
            RunSolver<float, float>(solver, direction, conv_config, algo, use_cpu_ref);
            return;
        case miopenBFloat16:
            RunSolver<bfloat16, bfloat16>(solver, direction, conv_config, algo, use_cpu_ref);
            return;
        default:
            throw std::runtime_error("handling of this data type is not yet implemented");
        }
        // clang-format on
    }
    else if(direction == miopen::conv::Direction::Forward &&
            conv_config.GetXDataType() == miopenInt8 && conv_config.GetWDataType() == miopenInt8 &&
            conv_config.GetYDataType() == miopenInt32)
    {
        RunSolverFwd<int8_t, int8_t, int32_t, int32_t>(solver, conv_config, algo, use_cpu_ref);
        return;
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
        // std::cout << "Test " << dev_descr << " (supported: " << supported << ")" << std::endl;

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
            GTEST_FAIL() << dev_descr << " is" << (is_applicable ? "" : " not")
                         << " applicable for " << solver.SolverDbId() << " but "
                         << (supported ? "" : "not ") << "marked as supported";
        }
    }
}
