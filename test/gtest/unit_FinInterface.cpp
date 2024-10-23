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

#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <gtest/gtest.h>

#include <miopen/fin/fin_interface.hpp>

namespace {

struct TestParams
{
    friend std::ostream& operator<<(std::ostream& os, const TestParams& tp)
    {
        os << "none";
        return os;
    }
};

struct SolverInfo
{
    SolverInfo() = default;
    SolverInfo(uint64_t id_, bool dynamic_, bool tunable_) : id(id_), dynamic(dynamic_), tunable(tunable_) {}

    friend std::ostream& operator<<(std::ostream& os, const SolverInfo& info)
    {
        os << "(";
        os << "id:" << info.id;
        os << ", dynamic:" << info.dynamic;
        os << ", tunable:" << info.tunable;
        os << ")";
        return os;
    }

    uint64_t id;
    bool dynamic;
    bool tunable;
};

struct ConvSolverInfo : SolverInfo
{
    using SolverInfo::SolverInfo;
    ConvSolverInfo(uint64_t id_, bool dynamic_, bool tunable_, std::string algo_) : SolverInfo(id_, dynamic_, tunable_), algo(std::move(algo_)) {}

    friend std::ostream& operator<<(std::ostream& os, const ConvSolverInfo& info)
    {
        os << "(";
        os << static_cast<const SolverInfo&>(info);
        os << ", algo:" << info.algo;
        os << ")";
        return os;
    }

    std::string algo;
};

using BatchNormSolverInfo = SolverInfo;

template <class Info>
struct TestCase
{
    friend std::ostream& operator<<(std::ostream& os, const TestCase& tc)
    {
        os << "(";
        os << "name:" << tc.name;
        os << ", info:" << tc.info;
        os << ")";
        return os;
    }

    std::string name;
    Info info;
};

using ConvTestCase = TestCase<ConvSolverInfo>;
using BatchNormTestCase = TestCase<BatchNormSolverInfo>;

const auto& GetTestParams()
{
    static const auto params = TestParams{};
    return params;
}

const auto& GetConvSolversInfo()
{
    static const std::unordered_map<std::string, ConvSolverInfo> solver_info = {
        // clang-format off
        {"ConvAsm3x3U",                                         {1,     false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvAsm1x1U",                                         {2,     false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvAsm1x1UV2",                                       {3,     false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvAsm5x10u2v2f1",                                   {5,     false,  false,  "miopenConvolutionFwdAlgoDirect"}},
        {"ConvAsm5x10u2v2b1",                                   {6,     false,  false,  "miopenConvolutionFwdAlgoDirect"}},
        {"ConvAsm7x7c3h224w224k64u2v2p3q3f1",                   {7,     false,  false,  "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclDirectFwd11x11",                               {8,     false,  false,  "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclDirectFwdGen",                                 {9,     false,  false,  "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclDirectFwd",                                    {11,    false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclDirectFwd1x1",                                 {13,    false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvBinWinograd3x3U",                                 {14,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvBinWinogradRxS",                                  {15,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvAsmBwdWrW3x3",                                    {16,    false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvAsmBwdWrW1x1",                                    {17,    false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclBwdWrW2<1>",                                   {18,    false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclBwdWrW2<2>",                                   {19,    false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclBwdWrW2<4>",                                   {20,    false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclBwdWrW2<8>",                                   {21,    false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclBwdWrW2<16>",                                  {22,    false,  true,   "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclBwdWrW2NonTunable",                            {23,    false,  false,  "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclBwdWrW53",                                     {24,    false,  false,  "miopenConvolutionFwdAlgoDirect"}},
        {"ConvOclBwdWrW1x1",                                    {25,    false,  false,  "miopenConvolutionFwdAlgoDirect"}},
        {"ConvHipImplicitGemmV4R1Fwd",                          {26,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmV4R1WrW",                          {31,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"fft",                                                 {34,    false,  false,  "miopenConvolutionFwdAlgoFFT"}},
        {"ConvWinograd3x3MultipassWrW<3-4>",                    {35,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvBinWinogradRxSf3x2",                              {37,    true,   true,   "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<3-5>",                    {38,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<3-6>",                    {39,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<3-2>",                    {40,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<3-3>",                    {41,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<7-2>",                    {42,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<7-3>",                    {43,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<7-2-1-1>",                {44,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<7-3-1-1>",                {45,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<1-1-7-2>",                {46,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<1-1-7-3>",                {47,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<5-3>",                    {48,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvWinograd3x3MultipassWrW<5-4>",                    {49,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvBinWinogradRxSf2x3",                              {53,    true,   true,   "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvHipImplicitGemmV4R4Fwd",                          {54,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmBwdDataV1R1",                      {55,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmBwdDataV4R1",                      {56,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmBwdDataV1R1Xdlops",                {57,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmBwdDataV4R1Xdlops",                {60,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmV4R4WrW",                          {61,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvAsmImplicitGemmV4R1DynamicFwd",                   {62,    true,   false,  "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvAsmImplicitGemmV4R1DynamicFwd_1x1",               {63,    true,   false,  "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmForwardV4R4Xdlops",                {64,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvAsmImplicitGemmV4R1DynamicBwd",                   {65,    true,   false,  "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvAsmImplicitGemmV4R1DynamicWrw",                   {66,    true,   false,  "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvMPBidirectWinograd<2-3>",                         {67,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvMPBidirectWinograd<3-3>",                         {68,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvMPBidirectWinograd<4-3>",                         {69,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvMPBidirectWinograd<5-3>",                         {70,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvMPBidirectWinograd<6-3>",                         {71,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvAsmImplicitGemmGTCDynamicWrwXdlops",              {72,    true,   false,  "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmWrwV4R4Xdlops",                    {73,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvAsmImplicitGemmGTCDynamicFwdXdlops",              {74,    true,   false,  "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvMPBidirectWinograd_xdlops<2-3>",                  {75,    false,  true,   "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvMPBidirectWinograd_xdlops<3-3>",                  {76,    false,  true,   "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvMPBidirectWinograd_xdlops<4-3>",                  {77,    false,  true,   "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvMPBidirectWinograd_xdlops<5-3>",                  {78,    false,  true,   "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvMPBidirectWinograd_xdlops<6-3>",                  {79,    false,  true,   "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvHipImplicitGemmForwardV4R5Xdlops",                {80,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm",    {81,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvAsmImplicitGemmGTCDynamicBwdXdlops",              {82,    true,   false,  "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm",        {83,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvBinWinogradRxSf2x3g1",                            {84,    true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvDirectNaiveConvFwd",                              {85,    true,   false,  "miopenConvolutionFwdAlgoDirect"}},
        {"ConvDirectNaiveConvBwd",                              {86,    true,   false,  "miopenConvolutionFwdAlgoDirect"}},
        {"ConvDirectNaiveConvWrw",                              {87,    true,   false,  "miopenConvolutionFwdAlgoDirect"}},
        {"GemmFwd1x1_0_1",                                      {88,    true,   false,  "miopenConvolutionFwdAlgoGEMM"}},
        {"GemmFwd1x1_0_1_int8",                                 {89,    true,   false,  "miopenConvolutionFwdAlgoGEMM"}},
        {"GemmFwd1x1_0_2",                                      {90,    true,   false,  "miopenConvolutionFwdAlgoGEMM"}},
        {"GemmFwdRest",                                         {91,    true,   false,  "miopenConvolutionFwdAlgoGEMM"}},
        {"GemmBwd1x1_stride2",                                  {95,    true,   false,  "miopenConvolutionFwdAlgoGEMM"}},
        {"GemmBwd1x1_stride1",                                  {96,    true,   false,  "miopenConvolutionFwdAlgoGEMM"}},
        {"GemmBwdRest",                                         {97,    true,   false,  "miopenConvolutionFwdAlgoGEMM"}},
        {"ConvMlirIgemmFwd",                                    {98,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvMlirIgemmBwd",                                    {99,    false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvMlirIgemmWrW",                                    {100,   false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"GemmWrw1x1_stride1",                                  {101,   true,   false,  "miopenConvolutionFwdAlgoGEMM"}},
        {"GemmWrwUniversal",                                    {102,   true,   false,  "miopenConvolutionFwdAlgoGEMM"}},
        {"ConvMlirIgemmFwdXdlops",                              {103,   false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvMlirIgemmBwdXdlops",                              {104,   false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvMlirIgemmWrWXdlops",                              {105,   false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC",          {107,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC",          {108,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC",          {110,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvCkIgemmFwdV6r1DlopsNchw",                         {114,   false,  true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC",          {127,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmFwdXdlops",                        {128,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmBwdXdlops",                        {129,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmGroupFwdXdlops",                   {137,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemm3DGroupFwdXdlops",                 {138,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvWinoFuryRxS<2-3>",                                {139,   true,   false,  "miopenConvolutionFwdAlgoWinograd"}},
        {"ConvHipImplicitGemm3DGroupWrwXdlops",                 {140,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemm3DGroupBwdXdlops",                 {141,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmF16F8F16FwdXdlops",                {149,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmF16F8F16BwdXdlops",                {150,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmF16F8F16WrwXdlops",                {151,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmGroupBwdXdlops",                   {155,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmGroupWrwXdlops",                   {156,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        // clang-format on
    };

    return solver_info;
}

const auto& GetBatchNormSolversInfo()
{
    static const std::unordered_map<std::string, BatchNormSolverInfo> solver_info = {
        // clang-format off
        {"BnFwdTrainingSpatialSingle",      {113,   false,  false}},
        {"BnFwdTrainingSpatialMultiple",    {115,   false,  false}},
        {"BnFwdTrainingPerActivation",      {116,   false,  false}},
        {"BnBwdTrainingSpatialSingle",      {117,   false,  false}},
        {"BnBwdTrainingSpatialMultiple",    {118,   false,  false}},
        {"BnBwdTrainingPerActivation",      {119,   false,  false}},
        {"BnFwdInference",                  {120,   true,   false}},
        {"BnCKFwdInference",                {142,   true,   false}},
        {"BnCKBwdBackward",                 {143,   true,   false}},
        {"BnCKFwdTraining",                 {144,   true,   false}},
        // clang-format on
    };

    return solver_info;
}

const auto& GetConvTestCases()
{
    static const auto test_cases = [] {
        std::vector<ConvTestCase> test_cases;
        const auto& sinfo = GetConvSolversInfo();
        for(const auto& s : sinfo)
            test_cases.emplace_back(ConvTestCase{s.first, s.second});
        return test_cases;
    }();
    return test_cases;
}

const auto& GetBatchNormTestCases()
{
    static const auto test_cases = [] {
        std::vector<BatchNormTestCase> test_cases;
        const auto& sinfo = GetBatchNormSolversInfo();
        for(const auto& s : sinfo)
            test_cases.emplace_back(BatchNormTestCase{s.first, s.second});
        return test_cases;
    }();
    return test_cases;
}

template <class Solver, class Info>
void CheckSolverInfo(const Solver& solver, const Info& info)
{
    ASSERT_EQ(solver.GetId(), info.id);
    ASSERT_EQ(solver.IsDynamic(), info.dynamic);
    ASSERT_EQ(solver.IsTunable(), info.tunable);
}

void CheckConvSolverInfo(const miopen::fin::ConvSolver& solver, const ConvSolverInfo& info)
{
    ASSERT_NO_FATAL_FAILURE(CheckSolverInfo(solver, info));
    ASSERT_EQ(solver.GetAlgo(miopen::conv::Direction::Forward), info.algo);
}

template <class Solver, class TestCase>
void CheckSolver(const Solver& solver, const TestCase& test_case)
{
    ASSERT_EQ(solver.GetName(), test_case.name);
    ASSERT_EQ(solver.IsValid(), true);
    ASSERT_NO_FATAL_FAILURE(CheckSolverInfo(solver, test_case.info));
}

void CheckConvSolver(const miopen::fin::ConvSolver& solver, const ConvTestCase& test_case)
{
    ASSERT_EQ(solver.GetName(), test_case.name);
    ASSERT_EQ(solver.IsValid(), true);
    ASSERT_NO_FATAL_FAILURE(CheckConvSolverInfo(solver, test_case.info));
}

class TestGetAllConvSolvers : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
        const auto& solvers = miopen::fin::FinInterface::GetAllConvSolvers();
        const auto& solvers_info = GetConvSolversInfo();

        ASSERT_EQ(solvers.size(), solvers_info.size());
        for(const auto& solver : solvers)
        {
            const auto& name = solver.GetName();
            const auto& solver_info = solvers_info.find(name);
            if(solver_info == solvers_info.end())
            {
                const std::string error = name + " not found";
                GTEST_FAIL() << error;
            }
            ASSERT_NO_FATAL_FAILURE(CheckConvSolverInfo(solver, solver_info->second));
        }
    }
};

class TestGetConvSolver : public ::testing::TestWithParam<std::tuple<TestParams, ConvTestCase>>
{
public:
    void RunTest()
    {
        ConvTestCase test_case;
        std::tie(std::ignore, test_case) = GetParam();
        const auto solver = miopen::fin::FinInterface::GetConvSolver(test_case.name);
        CheckConvSolver(solver, test_case);
    }
};

class TestGetAllBatchNormSolvers : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
        const auto& solvers = miopen::fin::FinInterface::GetAllBatchNormSolvers();
        const auto& solvers_info = GetBatchNormSolversInfo();

        ASSERT_EQ(solvers.size(), solvers_info.size());
        for(const auto& solver : solvers)
        {
            const auto& name = solver.GetName();
            const auto& solver_info = solvers_info.find(name);
            if(solver_info == solvers_info.end())
            {
                const std::string error = name + " not found";
                GTEST_FAIL() << error;
            }
            ASSERT_NO_FATAL_FAILURE(CheckSolverInfo(solver, solver_info->second));
        }
    }
};

class TestGetBatchNormSolver : public ::testing::TestWithParam<std::tuple<TestParams, BatchNormTestCase>>
{
public:
    void RunTest()
    {
        BatchNormTestCase test_case;
        std::tie(std::ignore, test_case) = GetParam();
        const auto solver = miopen::fin::FinInterface::GetBatchNormSolver(test_case.name);
        CheckSolver(solver, test_case);
    }
};

} // namespace

using CPU_FinInterfaceTestGetAllConvSolvers_NONE  = TestGetAllConvSolvers;
using CPU_FinInterfaceTestGetConvSolver_NONE  = TestGetConvSolver;
using CPU_FinInterfaceTestGetAllBatchNormSolvers_NONE  = TestGetAllBatchNormSolvers;
using CPU_FinInterfaceTestGetBatchNormSolver_NONE  = TestGetBatchNormSolver;

TEST_P(CPU_FinInterfaceTestGetAllConvSolvers_NONE, FinInterface) { this->RunTest(); };
TEST_P(CPU_FinInterfaceTestGetConvSolver_NONE, FinInterface) { this->RunTest(); };
TEST_P(CPU_FinInterfaceTestGetAllBatchNormSolvers_NONE, FinInterface) { this->RunTest(); };
TEST_P(CPU_FinInterfaceTestGetBatchNormSolver_NONE, FinInterface) { this->RunTest(); };

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_FinInterfaceTestGetAllConvSolvers_NONE,
                         testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_FinInterfaceTestGetConvSolver_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::ValuesIn(GetConvTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_FinInterfaceTestGetAllBatchNormSolvers_NONE,
                         testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_FinInterfaceTestGetBatchNormSolver_NONE,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::ValuesIn(GetBatchNormTestCases())));
