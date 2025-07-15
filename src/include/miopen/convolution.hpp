/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_CONVOLUTION_HPP_
#define GUARD_MIOPEN_CONVOLUTION_HPP_

#include <miopen/common.hpp>
#include <miopen/env.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/kernel.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/names.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/invoker.hpp>
#include <miopen/conv/tensors.hpp>

#include <nlohmann/json_fwd.hpp>

#include <boost/any.hpp>

#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <random>

MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP8_ROUNDING_SEED)

namespace miopen {

namespace conv {
struct ProblemDescription;
} // namespace conv

namespace solver {
struct ConvSolution;
} // namespace solver

struct ExecutionContext;
struct Handle;
struct TensorDescriptor;

struct MIOPEN_INTERNALS_EXPORT ConvolutionAttribute
{
    class Gfx90aFp16alt
    {
        int value = -1;
        friend struct ConvolutionAttribute; // For direct r/w.

        inline int Get() const
        {
            return env::value_or(MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL, value);
        }

    public:
        inline bool GetFwd() const { return Get() == 1; } // false is the default.
        inline bool GetBwd() const { return Get() != 0; } // true is the default.
        inline bool GetWrW() const { return Get() != 0; } // true is the default.

        friend void to_json(nlohmann::json& json, const Gfx90aFp16alt& attribute);
        friend void from_json(const nlohmann::json& json, Gfx90aFp16alt& attribute);
    } gfx90aFp16alt;

    struct FP8RoundingMode
    {
        inline uint32_t InitSeed()
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
            return distribution(gen);
        }
        miopenF8RoundingMode_t rounding_mode = miopenF8RoundingModeStochastic;
        uint32_t seed                        = InitSeed();
        friend struct ConvolutionAttribute;

        inline miopenF8RoundingMode_t Get() const
        {
            return env::value_or(MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP8_ROUNDING_MODE, rounding_mode);
        }

        inline uint32_t GetSeed() const
        {
            // assert(rounding_mode == miopenF8RoundingModeStochastic);
            return env::value_or(MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP8_ROUNDING_SEED, seed);
        }

        inline void SetSeed(const uint32_t s) { seed = s; }
    } fp8rounding_mode;

    class Deterministic
    {
        int value = 0;
        friend struct ConvolutionAttribute;

    public:
        inline int Get() const
        {
            return env::value_or(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC, value);
        }
        operator bool() const
        {
            const auto tmp_val = this->Get(); // Make sure we read the env var
            return tmp_val == 1;
        }
    } deterministic;

    /// Tri-state attribute values:
    /// * -1: Default (attribute-specific).
    /// * 0: Disabled/Yes.
    /// * 1: Enabled/No.
    void Set(miopenConvolutionAttrib_t attr, int value);
    int Get(miopenConvolutionAttrib_t attr) const;

    friend void to_json(nlohmann::json& json, const ConvolutionAttribute& conv);
    friend void from_json(const nlohmann::json& json, ConvolutionAttribute& conv);
};

struct Solution;

std::vector<Solution> FindConvolution(const ExecutionContext& ctx,
                                      const conv::ProblemDescription& problem,
                                      const AnyInvokeParams& invoke_ctx,
                                      int requestAlgoCount,
                                      bool force_attach_binary);

enum class FallbackPath
{
    None = 0,
    WTI,
    AI,
    Default_ = None
};

struct MIOPEN_INTERNALS_EXPORT ConvolutionDescriptor : miopenConvolutionDescriptor
{
    ConvolutionDescriptor(std::size_t spatial_dim,
                          miopenConvolutionMode_t c_mode,
                          miopenPaddingMode_t p_mode,
                          const std::vector<int>& p_pads              = {0, 0},
                          const std::vector<int>& p_strides           = {1, 1},
                          const std::vector<int>& p_dilations         = {1, 1},
                          const std::vector<int>& p_trans_output_pads = {0, 0},
                          int p_group_count                           = 1,
                          float p_lowp_quant                          = float(1));

    ConvolutionDescriptor(const std::vector<int>& p_pads              = {0, 0},
                          const std::vector<int>& p_strides           = {1, 1},
                          const std::vector<int>& p_dilations         = {1, 1},
                          const std::vector<int>& p_trans_output_pads = {0, 0},
                          int p_group_count                           = 1,
                          float p_lowp_quant                          = float(1));

    std::size_t GetSpatialDimension() const;

    const std::vector<int>& GetConvPads() const;

    const std::vector<int>& GetConvStrides() const;

    const std::vector<int>& GetConvDilations() const;

    const std::vector<int>& GetTransposeConvPads() const;

    int GetGroupCount() const;

    TensorDescriptor GetForwardOutputTensorWithLayout(const TensorDescriptor& xDesc,
                                                      const TensorDescriptor& wDesc,
                                                      const std::string& yLayout,
                                                      miopenDataType_t yType = miopenFloat) const;

    TensorDescriptor GetForwardOutputTensor(const TensorDescriptor& xDesc,
                                            const TensorDescriptor& wDesc,
                                            miopenDataType_t yType = miopenFloat) const;

    bool IsWinograd3x3SupportedAndFast(const miopen::ExecutionContext& ctx,
                                       const conv::ProblemDescription& problem) const;

    std::size_t GetWorkSpaceSize(ExecutionContext ctx,
                                 const conv::ProblemDescription& problem) const;

    void FindConvFwdAlgorithm(const Handle& handle,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& wDesc,
                              ConstData_t w,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              int requestAlgoCount,
                              int* returnedAlgoCount,
                              miopenConvAlgoPerf_t* perfResults,
                              Data_t workSpace,
                              std::size_t workSpaceSize,
                              bool exhaustiveSearch) const;

    void ConvolutionForward(const Handle& handle,
                            const void* alpha,
                            const TensorDescriptor& xDesc,
                            ConstData_t x,
                            const TensorDescriptor& wDesc,
                            ConstData_t w,
                            miopenConvFwdAlgorithm_t algo,
                            const void* beta,
                            const TensorDescriptor& yDesc,
                            Data_t y,
                            Data_t workSpace,
                            std::size_t workSpaceSize) const;

    std::size_t GetSolutionCount(const ExecutionContext& ctx,
                                 const conv::ProblemDescription& problem) const;

    std::vector<miopenConvSolution_t>
    GetSolutions(const ExecutionContext& ctx,
                 const conv::ProblemDescription& problem,
                 size_t maxSolutionCount,
                 FallbackPath* fallbackPathTaken,
                 const AnyInvokeParams* invokeParams = nullptr) const;

    void CompileSolution(const ExecutionContext& ctx,
                         const conv::ProblemDescription& problem,
                         solver::Id solver_id) const;

    std::size_t GetForwardSolutionWorkspaceSize(const Handle& handle,
                                                const TensorDescriptor& wDesc,
                                                const TensorDescriptor& xDesc,
                                                const TensorDescriptor& yDesc,
                                                solver::Id solver_id) const;

    void ConvolutionForwardImmediate(const Handle& handle,
                                     const TensorDescriptor& wDesc,
                                     ConstData_t w,
                                     const TensorDescriptor& xDesc,
                                     ConstData_t x,
                                     const TensorDescriptor& yDesc,
                                     Data_t y,
                                     Data_t workSpace,
                                     std::size_t workSpaceSize,
                                     solver::Id solver_id) const;

    void FindConvBwdDataAlgorithm(const Handle& handle,
                                  const TensorDescriptor& dyDesc,
                                  ConstData_t dy,
                                  const TensorDescriptor& wDesc,
                                  ConstData_t w,
                                  const TensorDescriptor& dxDesc,
                                  Data_t dx,
                                  int requestAlgoCount,
                                  int* returnedAlgoCount,
                                  miopenConvAlgoPerf_t* perfResults,
                                  Data_t workSpace,
                                  std::size_t workSpaceSize,
                                  bool exhaustiveSearch) const;

    void ConvolutionBackwardData(const Handle& handle,
                                 const void* alpha,
                                 const TensorDescriptor& dyDesc,
                                 ConstData_t dy,
                                 const TensorDescriptor& wDesc,
                                 ConstData_t w,
                                 miopenConvBwdDataAlgorithm_t algo,
                                 const void* beta,
                                 const TensorDescriptor& dxDesc,
                                 Data_t dx,
                                 Data_t workSpace,
                                 std::size_t workSpaceSize) const;

    std::size_t GetBackwardSolutionWorkspaceSize(const Handle& handle,
                                                 const TensorDescriptor& dyDesc,
                                                 const TensorDescriptor& wDesc,
                                                 const TensorDescriptor& dxDesc,
                                                 solver::Id solver_id) const;

    void ConvolutionBackwardImmediate(const Handle& handle,
                                      const TensorDescriptor& dyDesc,
                                      ConstData_t dy,
                                      const TensorDescriptor& wDesc,
                                      ConstData_t w,
                                      const TensorDescriptor& dxDesc,
                                      Data_t dx,
                                      Data_t workSpace,
                                      std::size_t workSpaceSize,
                                      solver::Id solver_id) const;

    std::size_t GetWrwSolutionWorkspaceSize(const Handle& handle,
                                            const TensorDescriptor& dyDesc,
                                            const TensorDescriptor& xDesc,
                                            const TensorDescriptor& dwDesc,
                                            solver::Id solver_id) const;

    void ConvolutionWrwImmediate(const Handle& handle,
                                 const TensorDescriptor& dyDesc,
                                 ConstData_t dy,
                                 const TensorDescriptor& xDesc,
                                 ConstData_t x,
                                 const TensorDescriptor& dwDesc,
                                 Data_t dw,
                                 Data_t workSpace,
                                 std::size_t workSpaceSize,
                                 solver::Id solver_id) const;

    void FindConvBwdWeightsAlgorithm(const Handle& handle,
                                     const TensorDescriptor& dyDesc,
                                     ConstData_t dy,
                                     const TensorDescriptor& xDesc,
                                     ConstData_t x,
                                     const TensorDescriptor& dwDesc,
                                     Data_t dw,
                                     int requestAlgoCount,
                                     int* returnedAlgoCount,
                                     miopenConvAlgoPerf_t* perfResults,
                                     Data_t workSpace,
                                     std::size_t workSpaceSize,
                                     bool exhaustiveSearch) const;

    void ConvolutionBackwardWeights(const Handle& handle,
                                    const void* alpha,
                                    const TensorDescriptor& dyDesc,
                                    ConstData_t dy,
                                    const TensorDescriptor& xDesc,
                                    ConstData_t x,
                                    miopenConvBwdWeightsAlgorithm_t algo,
                                    const void* beta,
                                    const TensorDescriptor& dwDesc,
                                    Data_t dw,
                                    Data_t workSpace,
                                    std::size_t workSpaceSize) const;

    std::size_t spatialDim;
    miopenConvolutionMode_t mode;
    miopenPaddingMode_t paddingMode;
    std::vector<int> pads;
    std::vector<int> strides;
    std::vector<int> dilations;
    std::vector<int> trans_output_pads;
    int group_count;
    float lowp_quant; // quantization factor for low precision
    FindMode findMode;
    ConvolutionAttribute attribute;

    std::vector<miopenConvSolution_t>
    GetSolutionsFallback(const ExecutionContext& ctx,
                         const conv::ProblemDescription& problem,
                         size_t maxSolutionCount,
                         FallbackPath* fallbackPathTaken     = nullptr,
                         const AnyInvokeParams* invokeParams = nullptr) const;

    std::size_t GetSolutionCountFallback(const ExecutionContext& ctx,
                                         const conv::ProblemDescription& problem) const;

    friend void to_json(nlohmann::json& json, const ConvolutionDescriptor& conv);
    friend void from_json(const nlohmann::json& json, ConvolutionDescriptor& conv);

private:
    void ValidateTensors(const ConvTensors& conv_tensors) const;
};

MIOPEN_INTERNALS_EXPORT void ConvolutionBackwardBias(const Handle& handle,
                                                     const void* alpha,
                                                     const TensorDescriptor& dyDesc,
                                                     ConstData_t dy,
                                                     const void* beta,
                                                     const TensorDescriptor& dbDesc,
                                                     Data_t db);

MIOPEN_INTERNALS_EXPORT Invoker LoadOrPrepareInvoker(const ExecutionContext& ctx,
                                                     const conv::ProblemDescription& problem,
                                                     solver::Id solver_id);

MIOPEN_INTERNALS_EXPORT std::ostream& operator<<(std::ostream& stream,
                                                 const ConvolutionDescriptor& c);

MIOPEN_INTERNALS_EXPORT void DumpTensorToFileFromDevice(const miopen::Handle& handle,
                                                        const miopen::TensorDescriptor& tDesc,
                                                        ConstData_t dData,
                                                        const fs::path& filename);

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenConvolutionDescriptor, miopen::ConvolutionDescriptor);

#endif // GUARD_MIOPEN_CONVOLUTION_HPP_
