// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "unit_conv_solver.hpp"

// Specialized version for 6 tests: ConvHipImplicitGemmGroup tests for 2D/3D  Fwd/Bwd/Wrw
// I don't use existing class because it's difficult to pass different layouts to be combined with
// other data

namespace miopen {
namespace unit_tests {

struct GroupXdlopsNumericData
{
    std::vector<size_t> x;
    std::vector<size_t> w;
    std::vector<int> pad;
    std::vector<int> stride;
    std::vector<int> dilation;
    unsigned int group_count;

    bool deterministic = false;
};

template <miopenDataType_t datatype>
ConvTestCase GetConvTestForGroupXdlops(miopenTensorLayout_t layout,
                                       GroupXdlopsNumericData&& conv_numeric_data)
{
    ConvTestCase conv_test_case = {{datatype, layout, std::move(conv_numeric_data.x)},
                                   {datatype, layout, std::move(conv_numeric_data.w)},
                                   datatype,
                                   {std::move(conv_numeric_data.pad),
                                    std::move(conv_numeric_data.stride),
                                    std::move(conv_numeric_data.dilation),
                                    std::move(conv_numeric_data.group_count),
                                    conv_numeric_data.deterministic}};

    return conv_test_case;
}

template <miopen::conv::Direction direction, miopenDataType_t datatype>
class UnitTestConvSolverGroupXDlops
    : public UnitTestConvSolverBase,
      public ::testing::TestWithParam<
          std::tuple<UnitTestConvSolverParams, miopenTensorLayout_t, GroupXdlopsNumericData>>
{
public:
    void RunTest(const miopen::solver::conv::ConvSolverInterface& solver)
    {
        UnitTestConvSolverParams params;
        miopenTensorLayout_t layout;
        GroupXdlopsNumericData conv_numeric_data;
        std::tie(params, layout, conv_numeric_data) = GetParam();

        ConvTestCase conv_config =
            GetConvTestForGroupXdlops<datatype>(layout, std::move(conv_numeric_data));

        this->RunTestImpl(
            solver, params, direction, conv_config, miopenConvolutionAlgoImplicitGEMM);
    }

protected:
    void SetUp() override
    {
        UnitTestConvSolverParams params;
        std::tie(params, std::ignore, std::ignore) = GetParam();
        this->SetUpImpl(params);
    }
};

} // namespace unit_tests
} // namespace miopen
