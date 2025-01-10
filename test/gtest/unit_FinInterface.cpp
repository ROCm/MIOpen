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

// batchnorm::ProblemDescription
#include <miopen/batchnorm/problem_description.hpp>
#if MIOPEN_ENABLE_FIN_INTERFACE
#include <miopen/fin/fin_interface.hpp>
#endif

#include "get_handle.hpp"
#include "unit_conv_solver.hpp"

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
    [[maybe_unused]] SolverInfo() = default;

    SolverInfo(uint64_t id_, bool dynamic_, bool tunable_)
        : id(id_), dynamic(dynamic_), tunable(tunable_)
    {
    }

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
    ConvSolverInfo(uint64_t id_, bool dynamic_, bool tunable_, std::string algo_)
        : SolverInfo(id_, dynamic_, tunable_), algo(std::move(algo_))
    {
    }

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

struct SolverConfig
{
    SolverConfig() : empty(true) {}
    SolverConfig(bool empty_) : empty(empty_) {}

    bool empty;
};

struct ConvSolverConfig : SolverConfig, private miopen::unit_tests::ConvTestCase
{
    ConvSolverConfig(miopen::conv::Direction direction_,
                     miopen::unit_tests::TensorDescriptorParams&& x,
                     miopen::unit_tests::TensorDescriptorParams&& w,
                     miopenDataType_t type_y,
                     miopen::unit_tests::ConvolutionDescriptorParams&& conv)
        : SolverConfig(false),
          miopen::unit_tests::ConvTestCase(std::move(x), std::move(w), type_y, std::move(conv)),
          direction(direction_)
    {
    }

    [[maybe_unused]] auto GetProblemDescription() const { return GetProblemDescription(direction); }

    friend std::ostream& operator<<(std::ostream& os, const ConvSolverConfig& config)
    {
        os << "(";
        if(config.empty)
        {
            os << "empty";
        }
        else
        {
            os << "direction:" << static_cast<int>(config.direction);
            os << ", " << static_cast<const miopen::unit_tests::ConvTestCase&>(config);
        }
        os << ")";
        return os;
    }

private:
    miopen::conv::Direction direction;

    using miopen::unit_tests::ConvTestCase::GetProblemDescription;
    using SolverConfig::SolverConfig;
};

struct BatchNormSolverConfig : SolverConfig
{
    BatchNormSolverConfig(int dummy) : SolverConfig(false) { std::ignore = dummy; }

    [[maybe_unused]] auto GetProblemDescription() const
    {
        return miopen::batchnorm::ProblemDescription{{}, {}, {}, {}, {}, {}, {}, {}};
    }

    friend std::ostream& operator<<(std::ostream& os, const BatchNormSolverConfig& config)
    {
        os << "(";
        if(config.empty)
            os << "empty";
        else
            os << "none";
        os << ")";
        return os;
    }

    using SolverConfig::SolverConfig;
};

template <class Info, class SolverConfig>
struct TestCase
{
    friend std::ostream& operator<<(std::ostream& os, const TestCase& tc)
    {
        os << "(";
        os << "name:" << tc.name;
        os << ", info:" << tc.info;
        os << ", config:" << tc.config;
        os << ")";
        return os;
    }

    std::string name;
    Info info;
    SolverConfig config;
};

using ConvTestCase      = TestCase<ConvSolverInfo, ConvSolverConfig>;
using BatchNormTestCase = TestCase<BatchNormSolverInfo, BatchNormSolverConfig>;

const auto& GetTestParams()
{
    static const auto params = TestParams{};
    return params;
}

template <class SolverInfo>
const auto& GetSolversInfo();

template <>
const auto& GetSolversInfo<ConvSolverInfo>()
{
    /// \anchor fin_interface_solver_info_coverage
    // This is the initial list of solvers for testing the interface. At the time of its creation,
    // it included all the available solvers. This was necessary to verify that all solvers were
    // correctly added to the solver registry. There is no need to keep it up to date by adding new
    // solvers as all new solvers will be added to the registry according to the existing template,
    // and it won't improve test coverage (and will only waste extra time).
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
        {"ConvHipImplicitGemmGroupBwdXdlops",                   {155,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        {"ConvHipImplicitGemmGroupWrwXdlops",                   {156,   true,   true,   "miopenConvolutionFwdAlgoImplicitGEMM"}},
        // clang-format on
    };

    return solver_info;
}

template <>
const auto& GetSolversInfo<BatchNormSolverInfo>()
{
    /// \ref fin_interface_solver_info_coverage
    static const std::unordered_map<std::string, BatchNormSolverInfo> solver_info = {
        // clang-format off
        {"BnFwdTrainingSpatialSingle",      {113,   false,  false}},
        {"BnFwdTrainingSpatialMultiple",    {115,   false,  false}},
        {"BnFwdTrainingPerActivation",      {116,   false,  false}},
        {"BnBwdTrainingSpatialSingle",      {117,   false,  false}},
        {"BnBwdTrainingSpatialMultiple",    {118,   false,  false}},
        {"BnBwdTrainingPerActivation",      {119,   false,  false}},
        {"BnFwdInference",                  {120,   true,   false}},
        {"BnCKFwdInference",                {142,   true,   true}},
        {"BnCKBwdBackward",                 {143,   true,   true}},
        {"BnCKFwdTraining",                 {144,   true,   true}},
        // clang-format on
    };

    return solver_info;
}

template <class SolverConfig>
const auto& GetSolverConfigs();

template <>
const auto& GetSolverConfigs<ConvSolverConfig>()
{
    /// \anchor fin_interface_solver_config_coverage
    // This list should include solvers that allow testing the core functionality, such as tunable
    // and non-tunable solvers. There's no need to add all solvers here, as it won't improve test
    // coverage (and will only waste extra time).
    static const std::unordered_map<std::string, ConvSolverConfig> configs = {
        // clang-format off
        // Non-tunable solvers
        {"ConvDirectNaiveConvFwd", {miopen::conv::Direction::Forward,           {miopenFloat, {1, 16, 14, 14}}, {miopenFloat, {48, 16, 5, 5}}, miopenFloat, {{2, 2}, {1, 1}, {1, 1}}}},
        {"ConvDirectNaiveConvBwd", {miopen::conv::Direction::BackwardData,      {miopenFloat, {1, 16, 14, 14}}, {miopenFloat, {48, 16, 5, 5}}, miopenFloat, {{2, 2}, {1, 1}, {1, 1}}}},
        {"ConvDirectNaiveConvWrw", {miopen::conv::Direction::BackwardWeights,   {miopenFloat, {1, 16, 14, 14}}, {miopenFloat, {48, 16, 5, 5}}, miopenFloat, {{2, 2}, {1, 1}, {1, 1}}}},
        // Tunable solvers
        {"ConvBinWinogradRxSf3x2", {miopen::conv::Direction::Forward,           {miopenFloat, {1, 20, 20, 20}}, {miopenFloat, {20, 20, 3, 3}}, miopenFloat, {{1, 1}, {1, 1}, {1, 1}}}},
        {"ConvBinWinogradRxSf2x3", {miopen::conv::Direction::BackwardWeights,   {miopenFloat, {1, 20, 20, 20}}, {miopenFloat, {20, 20, 3, 3}}, miopenFloat, {{1, 1}, {1, 1}, {1, 1}}}},
        // clang-format on
    };

    return configs;
}

template <>
const auto& GetSolverConfigs<BatchNormSolverConfig>()
{
    /// \ref fin_interface_solver_config_coverage
    static const std::unordered_map<std::string, BatchNormSolverConfig> configs = {
        // clang-format off
        /// \todo add configs
        {"DummySolver", {42}},
        // clang-format on
    };

    return configs;
}

template <class SolverInfo>
const auto& GetSolverNames()
{
    static const auto names = [] {
        std::vector<std::string> names;
        const auto& sinfo = GetSolversInfo<SolverInfo>();
        names.reserve(sinfo.size());
        for(const auto& s : sinfo)
            names.push_back(s.first);
        return names;
    }();
    return names;
}

template <class TestCase>
const auto& GetTestCases()
{
    static const auto test_cases = [] {
        std::vector<TestCase> test_cases;
        const auto& sinfo   = GetSolversInfo<decltype(std::declval<TestCase>().info)>();
        const auto& configs = GetSolverConfigs<decltype(std::declval<TestCase>().config)>();
        test_cases.reserve(sinfo.size());
        for(const auto& s : sinfo)
        {
            const auto& config = configs.find(s.first);
            if(config == configs.end())
                test_cases.emplace_back(TestCase{s.first, s.second, {}});
            else
                test_cases.emplace_back(TestCase{s.first, s.second, config->second});
        }
        return test_cases;
    }();
    return test_cases;
}

#if MIOPEN_ENABLE_FIN_INTERFACE

// Context
template <class Problem>
auto GetContext(miopen::Handle* handle, const Problem& problem);

template <>
auto GetContext(miopen::Handle* handle, const miopen::conv::ProblemDescription& problem)
{
    auto tmp = miopen::ExecutionContext{handle};
    problem.SetupFloats(tmp);
    return tmp;
}

template <>
auto GetContext(miopen::Handle* handle, const miopen::batchnorm::ProblemDescription&)
{
    auto tmp = miopen::ExecutionContext{handle};
    return tmp;
}

// Checks
template <class Solver, class SolverInfo>
void CheckSolverInfo(const Solver& solver, const SolverInfo& info)
{
    ASSERT_EQ(solver.GetId(), info.id);
    ASSERT_EQ(solver.IsDynamic(), info.dynamic);
    ASSERT_EQ(solver.IsTunable(), info.tunable);
    if constexpr(std::is_same_v<Solver, miopen::fin_interface::ConvSolver>)
        ASSERT_EQ(solver.GetAlgo(miopen::conv::Direction::Forward), info.algo);
}

template <class Solver, class SolverConfig>
void CheckSolverConfig(const Solver& solver, const SolverConfig& config)
{
    auto&& handle      = get_handle();
    const auto problem = config.GetProblemDescription();
    const auto ctx     = GetContext(&handle, problem);
    auto db            = miopen::GetDb(ctx);

    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    std::ignore = solver.GetWorkspaceSize(ctx, problem);

    /// \todo test FindSolution()

    const auto solutions = solver.GetAllSolutions(ctx, problem);
    ASSERT_GT(solutions.size(), 0);

    const auto pcfg_params = solver.GetPerfCfgParams(ctx, problem, db);
    ASSERT_NE(pcfg_params.empty(), solver.IsTunable());

    ASSERT_EQ(solver.TestPerfCfgParams(ctx, problem, pcfg_params), solver.IsTunable());
}

template <class Solver, class TestCase>
void CheckSolver(const Solver& solver, const TestCase& test_case)
{
    ASSERT_EQ(solver.GetName(), test_case.name);
    ASSERT_EQ(solver.IsValid(), true);
    ASSERT_NO_FATAL_FAILURE(CheckSolverInfo(solver, test_case.info));
    if(!test_case.config.empty)
        ASSERT_NO_FATAL_FAILURE(CheckSolverConfig(solver, test_case.config));
}

// GetAll*Solvers()
template <class TestCase>
const auto& InterfaceGetAllSolvers();

template <>
const auto& InterfaceGetAllSolvers<ConvTestCase>()
{
    return miopen::fin_interface::GetAllConvSolvers();
}

template <>
const auto& InterfaceGetAllSolvers<BatchNormTestCase>()
{
    return miopen::fin_interface::GetAllBatchNormSolvers();
}

// Get*Solvers(names)
template <class TestCase>
const auto InterfaceGetSolvers(const std::vector<std::string>& names);

template <>
const auto InterfaceGetSolvers<ConvTestCase>(const std::vector<std::string>& names)
{
    return miopen::fin_interface::GetConvSolvers(names);
}

template <>
const auto InterfaceGetSolvers<BatchNormTestCase>(const std::vector<std::string>& names)
{
    return miopen::fin_interface::GetBatchNormSolvers(names);
}

// Get*Solver(name)
template <class TestCase>
auto InterfaceGetSolver(const std::string& name);

template <>
auto InterfaceGetSolver<ConvTestCase>(const std::string& name)
{
    return miopen::fin_interface::GetConvSolver(name);
}

template <>
auto InterfaceGetSolver<BatchNormTestCase>(const std::string& name)
{
    return miopen::fin_interface::GetBatchNormSolver(name);
}

#endif // MIOPEN_ENABLE_FIN_INTERFACE

// Tests
template <class TestCase>
class TestGetAllSolvers : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
#if MIOPEN_ENABLE_FIN_INTERFACE
        const auto& solvers      = InterfaceGetAllSolvers<TestCase>();
        const auto& solvers_info = GetSolversInfo<decltype(std::declval<TestCase>().info)>();

        std::size_t num_checked_solvers = 0;
        for(const auto& solver : solvers)
        {
            const auto& name        = solver.GetName();
            const auto& solver_info = solvers_info.find(name);
            if(solver_info == solvers_info.end())
                continue;
            ASSERT_NO_FATAL_FAILURE(CheckSolverInfo(solver, solver_info->second));
            num_checked_solvers++;
        }
        ASSERT_EQ(num_checked_solvers, solvers_info.size());
#endif // MIOPEN_ENABLE_FIN_INTERFACE
    }

protected:
    void SetUp() override
    {
#if !MIOPEN_ENABLE_FIN_INTERFACE
        GTEST_SKIP();
#endif // !MIOPEN_ENABLE_FIN_INTERFACE
    }
};

template <class TestCase>
class TestGetSolvers : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
#if MIOPEN_ENABLE_FIN_INTERFACE
        const auto& solvers_info = GetSolversInfo<decltype(std::declval<TestCase>().info)>();
        const auto& names        = GetSolverNames<decltype(std::declval<TestCase>().info)>();
        const auto solvers       = InterfaceGetSolvers<TestCase>(names);

        ASSERT_EQ(solvers.size(), names.size());
        for(const auto& solver : solvers)
        {
            const auto& name        = solver.GetName();
            const auto& solver_info = solvers_info.find(name);
            if(solver_info == solvers_info.end())
            {
                const std::string error = name + " not found";
                GTEST_FAIL() << error;
            }
            ASSERT_NO_FATAL_FAILURE(CheckSolver(solver, TestCase{name, solver_info->second, {}}));
        }
#endif // MIOPEN_ENABLE_FIN_INTERFACE
    }

protected:
    void SetUp() override
    {
#if !MIOPEN_ENABLE_FIN_INTERFACE
        GTEST_SKIP();
#endif // !MIOPEN_ENABLE_FIN_INTERFACE
    }
};

template <class TestCase>
class TestGetSolver : public ::testing::TestWithParam<std::tuple<TestParams, TestCase>>
{
public:
    void RunTest()
    {
#if MIOPEN_ENABLE_FIN_INTERFACE
        TestCase test_case;
        std::tie(std::ignore, test_case) = this->GetParam();
        const auto solver                = InterfaceGetSolver<TestCase>(test_case.name);
        CheckSolver(solver, test_case);
#endif // MIOPEN_ENABLE_FIN_INTERFACE
    }

protected:
    void SetUp() override
    {
#if !MIOPEN_ENABLE_FIN_INTERFACE
        GTEST_SKIP();
#endif // !MIOPEN_ENABLE_FIN_INTERFACE
    }
};

} // namespace

// Convolution
using CPU_FinInterfaceTestGetAllConvSolvers_NONE = TestGetAllSolvers<ConvTestCase>;
using CPU_FinInterfaceTestGetConvSolvers_NONE    = TestGetSolvers<ConvTestCase>;
using GPU_FinInterfaceTestGetConvSolver_FP32     = TestGetSolver<ConvTestCase>;

TEST_P(CPU_FinInterfaceTestGetAllConvSolvers_NONE, FinInterface) { this->RunTest(); };
TEST_P(CPU_FinInterfaceTestGetConvSolvers_NONE, FinInterface) { this->RunTest(); };
TEST_P(GPU_FinInterfaceTestGetConvSolver_FP32, FinInterface) { this->RunTest(); };

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_FinInterfaceTestGetAllConvSolvers_NONE,
                         testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_FinInterfaceTestGetConvSolvers_NONE,
                         testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_FinInterfaceTestGetConvSolver_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::ValuesIn(GetTestCases<ConvTestCase>())));

// Batch normalization
using CPU_FinInterfaceTestGetAllBatchNormSolvers_NONE = TestGetAllSolvers<BatchNormTestCase>;
using CPU_FinInterfaceTestGetBatchNormSolvers_NONE    = TestGetSolvers<BatchNormTestCase>;
using GPU_FinInterfaceTestGetBatchNormSolver_FP32     = TestGetSolver<BatchNormTestCase>;

TEST_P(CPU_FinInterfaceTestGetAllBatchNormSolvers_NONE, FinInterface) { this->RunTest(); };
TEST_P(CPU_FinInterfaceTestGetBatchNormSolvers_NONE, FinInterface) { this->RunTest(); };
TEST_P(GPU_FinInterfaceTestGetBatchNormSolver_FP32, FinInterface) { this->RunTest(); };

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_FinInterfaceTestGetAllBatchNormSolvers_NONE,
                         testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_FinInterfaceTestGetBatchNormSolvers_NONE,
                         testing::Values(GetTestParams()));

// clang-format off
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_FinInterfaceTestGetBatchNormSolver_FP32,
                         testing::Combine(testing::Values(GetTestParams()),
                                          testing::ValuesIn(GetTestCases<BatchNormTestCase>())));
// clang-format on
