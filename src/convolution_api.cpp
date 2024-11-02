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
#include <miopen/miopen.h>
#include <miopen/miopen_internal.h>

#include <miopen/convolution.hpp>
#include <miopen/errors.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/driver_arguments.hpp>
#include <miopen/config.hpp>

#include <algorithm>
#include <optional>

using ExecutionContext   = miopen::ExecutionContext;
using Direction          = miopen::conv::Direction;
using ProblemDescription = miopen::conv::ProblemDescription;

// TODO: Make miopenConvAlgoPerf_t loggable
inline std::ostream& operator<<(std::ostream& os, miopenConvAlgoPerf_t /*unused*/) { return os; }

static inline auto MakeFwdCtxAndProblem(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t xDesc,
                                        const miopenTensorDescriptor_t wDesc,
                                        const miopenConvolutionDescriptor_t convDesc,
                                        const miopenTensorDescriptor_t yDesc)
{
    const auto& conv = miopen::deref(convDesc);
    const auto direction =
        (conv.mode != miopenTranspose) ? Direction::Forward : Direction::BackwardData;

    /// \anchor transpose_convolutions_x_y_swapping
    /// In transpose mode we exchange x with y. From the other hand, when Backward*
    /// ProblemDescription instances are constructed, x and y shall be swapped as well.
    /// As transpose mode swaps Forward with Backward AND x with y, the order of
    /// ctor arguments remains the same.
    auto problem = ProblemDescription{
        miopen::deref(xDesc), miopen::deref(wDesc), miopen::deref(yDesc), conv, direction};

    auto ctx = ExecutionContext{&miopen::deref(handle)};
    problem.SetupFloats(ctx);
    return std::make_tuple(std::move(ctx), std::move(problem));
}

static inline auto MakeBwdCtxAndProblem(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t dyDesc,
                                        const miopenTensorDescriptor_t wDesc,
                                        const miopenConvolutionDescriptor_t convDesc,
                                        const miopenTensorDescriptor_t dxDesc)
{
    const auto& conv = miopen::deref(convDesc);
    const auto direction =
        (conv.mode != miopenTranspose) ? Direction::BackwardData : Direction::Forward;

    /// \ref transpose_convolutions_x_y_swapping
    auto problem = ProblemDescription{
        miopen::deref(dyDesc), miopen::deref(wDesc), miopen::deref(dxDesc), conv, direction};

    auto ctx = ExecutionContext{&miopen::deref(handle)};
    problem.SetupFloats(ctx);
    return std::make_tuple(std::move(ctx), std::move(problem));
}

static inline auto MakeWrWCtxAndProblem(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t dyDesc,
                                        const miopenTensorDescriptor_t xDesc,
                                        const miopenConvolutionDescriptor_t convDesc,
                                        const miopenTensorDescriptor_t dwDesc)
{
    const auto direction = Direction::BackwardWeights;
    const auto& conv     = miopen::deref(convDesc);

    auto problem = (conv.mode == miopenTranspose) ? ProblemDescription{miopen::deref(xDesc),
                                                                       miopen::deref(dwDesc),
                                                                       miopen::deref(dyDesc),
                                                                       conv,
                                                                       direction}
                                                  : ProblemDescription{miopen::deref(dyDesc),
                                                                       miopen::deref(dwDesc),
                                                                       miopen::deref(xDesc),
                                                                       conv,
                                                                       direction};

    auto ctx = ExecutionContext{&miopen::deref(handle)};
    problem.SetupFloats(ctx);
    return std::make_tuple(std::move(ctx), std::move(problem));
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t* convDesc)
{
    MIOPEN_LOG_FUNCTION(convDesc);
    return miopen::try_([&] {
        auto& desc = miopen::deref(convDesc);
        desc       = new miopen::ConvolutionDescriptor();
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenInitConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                          miopenConvolutionMode_t c_mode,
                                                          int pad_h,
                                                          int pad_w,
                                                          int stride_h,
                                                          int stride_w,
                                                          int dilation_h,
                                                          int dilation_w)
{
    MIOPEN_LOG_FUNCTION(convDesc, c_mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    return miopen::try_([&] {
        miopen::deref(convDesc) = miopen::ConvolutionDescriptor(2,
                                                                c_mode,
                                                                miopenPaddingDefault,
                                                                {pad_h, pad_w},
                                                                {stride_h, stride_w},
                                                                {dilation_h, dilation_w});
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenInitConvolutionNdDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                            int spatialDim,
                                                            const int* padA,
                                                            const int* strideA,
                                                            const int* dilationA,
                                                            miopenConvolutionMode_t c_mode)
{
    const auto pads      = std::vector<int>(padA, padA + spatialDim);
    const auto strides   = std::vector<int>(strideA, strideA + spatialDim);
    const auto dilations = std::vector<int>(dilationA, dilationA + spatialDim);
    MIOPEN_LOG_FUNCTION(convDesc, spatialDim, pads, strides, dilations, c_mode);
    return miopen::try_([&] {
        miopen::deref(convDesc) = miopen::ConvolutionDescriptor(spatialDim,
                                                                c_mode,
                                                                miopenPaddingDefault,
                                                                pads,
                                                                strides,
                                                                dilations,
                                                                std::vector<int>(spatialDim, 0),
                                                                1,
                                                                1.0);
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenGetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                         int* groupCount)
{
    MIOPEN_LOG_FUNCTION(convDesc);
    return miopen::try_([&] { miopen::deref(groupCount) = miopen::deref(convDesc).group_count; });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenSetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                         int groupCount)
{
    MIOPEN_LOG_FUNCTION(convDesc, groupCount);
    return miopen::try_([&] { miopen::deref(convDesc).group_count = groupCount; });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenSetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc,
                                                       miopenConvolutionFindMode_t findMode)
{
    MIOPEN_LOG_FUNCTION(convDesc, findMode);
    return miopen::try_([&] {
        miopen::deref(convDesc).findMode.Set(static_cast<miopen::FindMode::Values>(findMode));
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenGetConvolutionFindMode(const miopenConvolutionDescriptor_t convDesc,
                                                       miopenConvolutionFindMode_t* findMode)
{
    MIOPEN_LOG_FUNCTION(convDesc, findMode);
    return miopen::try_([&] {
        miopen::deref(findMode) =
            static_cast<miopenConvolutionFindMode_t>(miopen::deref(convDesc).findMode.Get());
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionABBackwardWeightsGetWorkSpaceSize(const miopenAlphaBetaCase_t alpha_beta_case,
                                                   const miopenTensorDescriptor_t inputTensorDesc,
                                                   const miopenTensorDescriptor_t outputTensorDesc,
                                                   const miopenConvolutionDescriptor_t convDesc,
                                                   size_t* buffer_size)
{
    MIOPEN_LOG_FUNCTION(alpha_beta_case, outputTensorDesc);
    return miopen::try_([&] {
        miopenDataType_t data_type = miopen::deref(outputTensorDesc).GetType();
        size_t in_spatial_dims     = miopen::deref(inputTensorDesc).GetNumDims();

        assert(in_spatial_dims == miopen::deref(outputTensorDesc).GetNumDims());

        int G    = miopen::deref(convDesc).GetGroupCount();
        size_t C = std::get<1>(
            miopen::GetNCDHW(in_spatial_dims, miopen::deref(inputTensorDesc).GetLengths()));
        size_t K = std::get<1>(
            miopen::GetNCDHW(in_spatial_dims, miopen::deref(outputTensorDesc).GetLengths()));

        auto CKWrwRequireWorkspace = [&](size_t G,
                                         size_t C,
                                         size_t K,
                                         miopenDataType_t data_type,
                                         miopenAlphaBetaCase_t alpha_beta_case) {
            auto is_odd        = [](int num) { return num % 2 != 0; };
            size_t C_per_group = C / G;
            size_t K_per_group = K / G;

            return (alpha_beta_case == BILINEAR || alpha_beta_case == SCALE) ||
                   ((data_type == miopenHalf || data_type == miopenBFloat16) &&
                    (is_odd(C_per_group) || is_odd(K_per_group)));
        };

        size_t output_tensor_size = miopen::deref(outputTensorDesc).GetElementSize();
        size_t byte_size          = 0;
        if(CKWrwRequireWorkspace(G, C, K, data_type, alpha_beta_case))
        {
            switch(data_type)
            {
            case miopenInt32:
            case miopenFloat:
            case miopenHalf:
            case miopenBFloat16:
            case miopenInt8:
            case miopenFloat8:
            case miopenBFloat8: byte_size = 4; break;
            case miopenDouble:
            case miopenInt64: byte_size = 8; break;
            }
            *buffer_size = byte_size * output_tensor_size;
        }
        else
        {
            *buffer_size = 0;
        }

        MIOPEN_LOG_FUNCTION(
            alpha_beta_case, data_type, C, K, output_tensor_size, byte_size, *buffer_size);
    });
}

// Hidden C++ functions for MIGraphX.
MIOPEN_EXPORT extern "C" miopenStatus_t
miopenHiddenSetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc, int findMode)
{
    return miopen::try_([&] {
        miopen::deref(convDesc).findMode.Set(static_cast<miopen::FindMode::Values>(findMode));
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenHiddenGetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc,
                                                             int* findMode)
{
    return miopen::try_([&] {
        miopen::deref(findMode) = static_cast<int>(miopen::deref(convDesc).findMode.Get());
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenSetTransposeConvOutputPadding(miopenConvolutionDescriptor_t convDesc, int adj_h, int adj_w)
{
    MIOPEN_LOG_FUNCTION(convDesc, adj_h, adj_w);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).GetSpatialDimension() != 2)
        {
            MIOPEN_THROW("this API only deals with 2-D convolution");
        }

        miopen::deref(convDesc).trans_output_pads[0] = adj_h;
        miopen::deref(convDesc).trans_output_pads[1] = adj_w;
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenSetTransposeConvNdOutputPadding(
    miopenConvolutionDescriptor_t convDesc, int spatialDim, const int* adjA)
{
    if(miopen::IsLoggingFunctionCalls())
    {
        const miopen::logger::CArray<int, int> adj(adjA, spatialDim);
        MIOPEN_LOG_FUNCTION(convDesc, spatialDim, adj.values);
    }
    return miopen::try_([&] {
        if(spatialDim != miopen::deref(convDesc).GetSpatialDimension())
        {
            MIOPEN_THROW("spatialDim not consistent with convolution descriptor");
        }

        std::copy_n(adjA, spatialDim, miopen::deref(convDesc).trans_output_pads.begin());
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenGetConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                         miopenConvolutionMode_t* c_mode,
                                                         int* pad_h,
                                                         int* pad_w,
                                                         int* stride_h,
                                                         int* stride_w,
                                                         int* dilation_h,
                                                         int* dilation_w)
{
    MIOPEN_LOG_FUNCTION(convDesc);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).GetSpatialDimension() != 2)
        {
            MIOPEN_THROW("this API only deals with 2-D convolution");
        }

        miopen::deref(c_mode)     = miopen::deref(convDesc).mode;
        miopen::deref(pad_h)      = miopen::deref(convDesc).GetConvPads()[0];
        miopen::deref(pad_w)      = miopen::deref(convDesc).GetConvPads()[1];
        miopen::deref(stride_h)   = miopen::deref(convDesc).GetConvStrides()[0];
        miopen::deref(stride_w)   = miopen::deref(convDesc).GetConvStrides()[1];
        miopen::deref(dilation_h) = miopen::deref(convDesc).GetConvDilations()[0];
        miopen::deref(dilation_w) = miopen::deref(convDesc).GetConvDilations()[1];
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenGetConvolutionNdDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                           int requestedSpatialDim,
                                                           int* spatialDim,
                                                           int* padA,
                                                           int* strideA,
                                                           int* dilationA,
                                                           miopenConvolutionMode_t* c_mode)
{
    MIOPEN_LOG_FUNCTION(convDesc, requestedSpatialDim);
    return miopen::try_([&] {
        int spatial_dim = miopen::deref(convDesc).GetSpatialDimension();
        if(spatial_dim < requestedSpatialDim)
        {
            MIOPEN_THROW("requestedSpatialDim is larger than actual spatial dimension");
        }
        if(spatialDim != nullptr)
        {
            miopen::deref(spatialDim) = spatial_dim;
        }
        std::copy_n(miopen::deref(convDesc).GetConvPads().begin(), requestedSpatialDim, padA);
        std::copy_n(miopen::deref(convDesc).GetConvStrides().begin(), requestedSpatialDim, strideA);
        std::copy_n(
            miopen::deref(convDesc).GetConvDilations().begin(), requestedSpatialDim, dilationA);
        if(c_mode != nullptr)
        {
            miopen::deref(c_mode) = miopen::deref(convDesc).mode;
        }
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenGetConvolutionSpatialDim(miopenConvolutionDescriptor_t convDesc,
                                                         int* spatialDim)
{
    MIOPEN_LOG_FUNCTION(convDesc);
    return miopen::try_(
        [&] { miopen::deref(spatialDim) = miopen::deref(convDesc).GetSpatialDimension(); });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenGetConvolutionForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
                                     const miopenTensorDescriptor_t inputTensorDesc,
                                     const miopenTensorDescriptor_t filterDesc,
                                     int* n,
                                     int* c,
                                     int* h,
                                     int* w)
{
    MIOPEN_LOG_FUNCTION(convDesc, inputTensorDesc, filterDesc);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).GetSpatialDimension() != 2)
        {
            MIOPEN_THROW("this API only deals with 2-D convolution");
        }

        miopen::tie_deref(n, c, h, w) = miopen::tien<4>(
            miopen::deref(convDesc)
                .GetForwardOutputTensor(miopen::deref(inputTensorDesc), miopen::deref(filterDesc))
                .GetLengths());
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenGetConvolutionNdForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
                                       const miopenTensorDescriptor_t inputTensorDesc,
                                       const miopenTensorDescriptor_t filterDesc,
                                       int* nDim,
                                       int* outputTensorDimA)
{
    MIOPEN_LOG_FUNCTION(convDesc, inputTensorDesc, filterDesc);
    return miopen::try_([&] {
        auto out_desc = miopen::deref(convDesc).GetForwardOutputTensor(
            miopen::deref(inputTensorDesc), miopen::deref(filterDesc));

        miopen::deref(nDim) = out_desc.GetNumDims();

        for(unsigned i = 0; i < out_desc.GetNumDims(); ++i)
        {
            outputTensorDimA[i] = out_desc.GetLengths()[i];
        }
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc)
{
    MIOPEN_LOG_FUNCTION(convDesc);
    return miopen::try_([&] { miopen_destroy_object(convDesc); });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionForwardGetWorkSpaceSize(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeFwdCtxAndProblem(handle, xDesc, wDesc, convDesc, yDesc);
        *workSpaceSize         = miopen::deref(convDesc).GetWorkSpaceSize(ctx, problem);
    });
}

namespace miopen {
namespace debug {

MIOPEN_INTERNALS_EXPORT
void LogCmdConvolution(const miopen::TensorDescriptor& x,
                       const miopen::TensorDescriptor& w,
                       const miopen::ConvolutionDescriptor& conv,
                       const miopen::TensorDescriptor& y,
                       miopenProblemDirection_t dir,
                       std::optional<uint64_t> solver_id)
{
    if(miopen::IsLoggingCmd())
    {
        const std::string& str = ConvArgsForMIOpenDriver(x, w, conv, y, dir, solver_id);
        MIOPEN_LOG_DRIVER_CMD(str);
    }
}

MIOPEN_INTERNALS_EXPORT
void LogCmdFindConvolution(const miopen::TensorDescriptor& x,
                           const miopen::TensorDescriptor& w,
                           const miopen::ConvolutionDescriptor& conv,
                           const miopen::TensorDescriptor& y,
                           miopenProblemDirection_t dir,
                           std::optional<uint64_t> solver_id)
{
    if(miopen::IsLoggingCmd())
    {
        const std::string& str = ConvArgsForMIOpenDriver(x, w, conv, y, dir, solver_id);
        MIOPEN_LOG_DRIVER_CMD(str);
    }
}

MIOPEN_INTERNALS_EXPORT
void LogCmdConvolution(const miopenTensorDescriptor_t& xDesc,
                       const miopenTensorDescriptor_t& wDesc,
                       const miopenConvolutionDescriptor_t& convDesc,
                       const miopenTensorDescriptor_t& yDesc,
                       ConvDirection conv_dir,
                       bool is_immediate)
{
    if(miopen::IsLoggingCmd())
    {
        const auto dir              = CmdArgToDirection(conv_dir);
        const auto& [x, w, conv, y] = miopen::tie_deref(xDesc, wDesc, convDesc, yDesc);
        const auto solver_id        = is_immediate ? std::optional(0) : std::nullopt;
        LogCmdConvolution(x, w, conv, y, dir, solver_id);
    }
}

MIOPEN_INTERNALS_EXPORT
void LogCmdFindConvolution(const miopenTensorDescriptor_t& xDesc,
                           const miopenTensorDescriptor_t& wDesc,
                           const miopenConvolutionDescriptor_t& convDesc,
                           const miopenTensorDescriptor_t& yDesc,
                           ConvDirection conv_dir,
                           bool is_immediate)
{
    if(miopen::IsLoggingCmd())
    {
        const auto dir              = CmdArgToDirection(conv_dir);
        const auto& [x, w, conv, y] = miopen::tie_deref(xDesc, wDesc, convDesc, yDesc);
        const auto solver_id        = is_immediate ? std::optional(0) : std::nullopt;
        LogCmdFindConvolution(x, w, conv, y, dir, solver_id);
    }
}

} // namespace debug
} // namespace miopen

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenFindConvolutionForwardAlgorithm(miopenHandle_t handle,
                                      const miopenTensorDescriptor_t xDesc,
                                      const void* x,
                                      const miopenTensorDescriptor_t wDesc,
                                      const void* w,
                                      const miopenConvolutionDescriptor_t convDesc,
                                      const miopenTensorDescriptor_t yDesc,
                                      void* y,
                                      const int requestAlgoCount,
                                      int* returnedAlgoCount,
                                      miopenConvAlgoPerf_t* perfResults,
                                      void* workSpace,
                                      size_t workSpaceSize,
                                      bool exhaustiveSearch)
{

    MIOPEN_LOG_FUNCTION(handle,
                        xDesc,
                        x,
                        wDesc,
                        w,
                        convDesc,
                        yDesc,
                        y,
                        requestAlgoCount,
                        returnedAlgoCount,
                        perfResults,
                        workSpace,
                        workSpaceSize,
                        exhaustiveSearch);

    miopen::debug::LogCmdFindConvolution(
        xDesc, wDesc, convDesc, yDesc, miopen::debug::ConvDirection::Fwd, false);

    if(miopen::deref(convDesc).mode == miopenTranspose)
    {
        return miopen::try_([&] {
            miopen::deref(convDesc).FindConvBwdDataAlgorithm(miopen::deref(handle),
                                                             miopen::deref(xDesc),
                                                             DataCast(x),
                                                             miopen::deref(wDesc),
                                                             DataCast(w),
                                                             miopen::deref(yDesc),
                                                             DataCast(y),
                                                             requestAlgoCount,
                                                             returnedAlgoCount,
                                                             perfResults,
                                                             DataCast(workSpace),
                                                             workSpaceSize,
                                                             exhaustiveSearch);

            for(int i = 0; i < *returnedAlgoCount; ++i)
            {
                // It is guaranteed that enum values are equal, see conv_algo_name.cpp
                perfResults[i].fwd_algo =
                    static_cast<miopenConvFwdAlgorithm_t>(perfResults[i].bwd_data_algo);
            }
        });
    }

    return miopen::try_([&] {
        miopen::deref(convDesc).FindConvFwdAlgorithm(miopen::deref(handle),
                                                     miopen::deref(xDesc),
                                                     DataCast(x),
                                                     miopen::deref(wDesc),
                                                     DataCast(w),
                                                     miopen::deref(yDesc),
                                                     DataCast(y),
                                                     requestAlgoCount,
                                                     returnedAlgoCount,
                                                     perfResults,
                                                     DataCast(workSpace),
                                                     workSpaceSize,
                                                     exhaustiveSearch);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionForward(miopenHandle_t handle,
                         const void* alpha,
                         const miopenTensorDescriptor_t xDesc,
                         const void* x,
                         const miopenTensorDescriptor_t wDesc,
                         const void* w,
                         const miopenConvolutionDescriptor_t convDesc,
                         miopenConvFwdAlgorithm_t algo,
                         const void* beta,
                         const miopenTensorDescriptor_t yDesc,
                         void* y,
                         void* workSpace,
                         size_t workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle,
                        alpha,
                        xDesc,
                        x,
                        wDesc,
                        w,
                        convDesc,
                        algo,
                        beta,
                        yDesc,
                        y,
                        workSpace,
                        workSpaceSize);
    miopen::debug::LogCmdConvolution(
        xDesc, wDesc, convDesc, yDesc, miopen::debug::ConvDirection::Fwd, false);

    if(miopen::deref(convDesc).mode == miopenTranspose)
    {
        return miopen::try_([&] {
            // It is guaranteed that enum values are equal, see conv_algo_name.cpp
            const auto algo_trans = static_cast<miopenConvBwdDataAlgorithm_t>(algo);
            miopen::deref(convDesc).ConvolutionBackwardData(miopen::deref(handle),
                                                            alpha,
                                                            miopen::deref(xDesc),
                                                            DataCast(x),
                                                            miopen::deref(wDesc),
                                                            DataCast(w),
                                                            algo_trans,
                                                            beta,
                                                            miopen::deref(yDesc),
                                                            DataCast(y),
                                                            DataCast(workSpace),
                                                            workSpaceSize);
        });
    }

    return miopen::try_([&] {
        miopen::deref(convDesc).ConvolutionForward(miopen::deref(handle),
                                                   alpha,
                                                   miopen::deref(xDesc),
                                                   DataCast(x),
                                                   miopen::deref(wDesc),
                                                   DataCast(w),
                                                   algo,
                                                   beta,
                                                   miopen::deref(yDesc),
                                                   DataCast(y),
                                                   DataCast(workSpace),
                                                   workSpaceSize);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionForwardBias(miopenHandle_t handle,
                             const void* alpha,
                             const miopenTensorDescriptor_t bDesc,
                             const void* b,
                             const void* beta,
                             const miopenTensorDescriptor_t yDesc,
                             void* y)
{

    MIOPEN_LOG_FUNCTION(handle, alpha, bDesc, b, beta, yDesc, y);

    // bfloat16 not supported for bias operation
    if(miopen::deref(yDesc).GetType() == miopenBFloat16 ||
       miopen::deref(bDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }

    return miopen::try_([&] {
        return OpTensor(miopen::deref(handle),
                        miopenTensorOpAdd,
                        alpha,
                        miopen::deref(yDesc),
                        DataCast(y),
                        alpha,
                        miopen::deref(bDesc),
                        DataCast(b),
                        beta,
                        miopen::deref(yDesc),
                        DataCast(y));
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionCount(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* solutionCount)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeFwdCtxAndProblem(handle, xDesc, wDesc, convDesc, yDesc);

        *solutionCount = miopen::deref(convDesc).GetSolutionCount(ctx, problem);
    });
}

static inline void ReturnSolutions(const std::vector<miopenConvSolution_t>& solutions,
                                   size_t* solution_count_ret,
                                   miopenConvSolution_t* solutions_ret)
{
    if(solution_count_ret != nullptr)
        *solution_count_ret = solutions.size();
    if(solutions_ret != nullptr)
    {
        for(auto i = 0; i < solutions.size(); ++i)
            solutions_ret[i] = solutions[i];
    }
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionForwardGetSolution(miopenHandle_t handle,
                                    const miopenTensorDescriptor_t wDesc,
                                    const miopenTensorDescriptor_t xDesc,
                                    const miopenConvolutionDescriptor_t convDesc,
                                    const miopenTensorDescriptor_t yDesc,
                                    const size_t maxSolutionCount,
                                    size_t* solutionCount,
                                    miopenConvSolution_t* solutions)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc, maxSolutionCount);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeFwdCtxAndProblem(handle, xDesc, wDesc, convDesc, yDesc);

        const auto found =
            miopen::deref(convDesc).GetSolutions(ctx, problem, maxSolutionCount, nullptr);

        assert(found.size() <= maxSolutionCount);
        ReturnSolutions(found, solutionCount, solutions);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t wDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 const uint64_t solution_id,
                                                 size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc, solution_id);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
        {
            *workSpaceSize =
                miopen::deref(convDesc).GetBackwardSolutionWorkspaceSize(miopen::deref(handle),
                                                                         miopen::deref(xDesc),
                                                                         miopen::deref(wDesc),
                                                                         miopen::deref(yDesc),
                                                                         solution_id);
        }
        else
        {
            *workSpaceSize =
                miopen::deref(convDesc).GetForwardSolutionWorkspaceSize(miopen::deref(handle),
                                                                        miopen::deref(wDesc),
                                                                        miopen::deref(xDesc),
                                                                        miopen::deref(yDesc),
                                                                        solution_id);
        }
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionForwardCompileSolution(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t wDesc,
                                        const miopenTensorDescriptor_t xDesc,
                                        const miopenConvolutionDescriptor_t convDesc,
                                        const miopenTensorDescriptor_t yDesc,
                                        const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc, solution_id);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeFwdCtxAndProblem(handle, xDesc, wDesc, convDesc, yDesc);
        miopen::deref(convDesc).CompileSolution(ctx, problem, solution_id);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionForwardImmediate(miopenHandle_t handle,
                                  const miopenTensorDescriptor_t wDesc,
                                  const void* w,
                                  const miopenTensorDescriptor_t xDesc,
                                  const void* x,
                                  const miopenConvolutionDescriptor_t convDesc,
                                  const miopenTensorDescriptor_t yDesc,
                                  void* y,
                                  void* workSpace,
                                  size_t workSpaceSize,
                                  const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(
        handle, wDesc, w, xDesc, x, convDesc, yDesc, y, workSpace, workSpaceSize, solution_id);
    miopen::debug::LogCmdConvolution(
        xDesc, wDesc, convDesc, yDesc, miopen::debug::ConvDirection::Fwd, true);

    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
        {
            miopen::deref(convDesc).ConvolutionBackwardImmediate(miopen::deref(handle),
                                                                 miopen::deref(xDesc),
                                                                 DataCast(x),
                                                                 miopen::deref(wDesc),
                                                                 DataCast(w),
                                                                 miopen::deref(yDesc),
                                                                 DataCast(y),
                                                                 DataCast(workSpace),
                                                                 workSpaceSize,
                                                                 solution_id);
        }
        else
        {
            miopen::deref(convDesc).ConvolutionForwardImmediate(miopen::deref(handle),
                                                                miopen::deref(wDesc),
                                                                DataCast(w),
                                                                miopen::deref(xDesc),
                                                                DataCast(x),
                                                                miopen::deref(yDesc),
                                                                DataCast(y),
                                                                DataCast(workSpace),
                                                                workSpaceSize,
                                                                solution_id);
        }
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionCount(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* solutionCount)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeBwdCtxAndProblem(handle, dyDesc, wDesc, convDesc, dxDesc);

        *solutionCount = miopen::deref(convDesc).GetSolutionCount(ctx, problem);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolution(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t dyDesc,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t dxDesc,
                                         const size_t maxSolutionCount,
                                         size_t* solutionCount,
                                         miopenConvSolution_t* solutions)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, maxSolutionCount);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeBwdCtxAndProblem(handle, dyDesc, wDesc, convDesc, dxDesc);

        const auto found =
            miopen::deref(convDesc).GetSolutions(ctx, problem, maxSolutionCount, nullptr);

        assert(found.size() <= maxSolutionCount);
        ReturnSolutions(found, solutionCount, solutions);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionWorkspaceSize(miopenHandle_t handle,
                                                      const miopenTensorDescriptor_t dyDesc,
                                                      const miopenTensorDescriptor_t wDesc,
                                                      const miopenConvolutionDescriptor_t convDesc,
                                                      const miopenTensorDescriptor_t dxDesc,
                                                      const uint64_t solution_id,
                                                      size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, solution_id);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
        {
            *workSpaceSize =
                miopen::deref(convDesc).GetForwardSolutionWorkspaceSize(miopen::deref(handle),
                                                                        miopen::deref(wDesc),
                                                                        miopen::deref(dyDesc),
                                                                        miopen::deref(dxDesc),
                                                                        solution_id);
        }
        else
        {
            *workSpaceSize =
                miopen::deref(convDesc).GetBackwardSolutionWorkspaceSize(miopen::deref(handle),
                                                                         miopen::deref(dyDesc),
                                                                         miopen::deref(wDesc),
                                                                         miopen::deref(dxDesc),
                                                                         solution_id);
        }
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardDataCompileSolution(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t dyDesc,
                                             const miopenTensorDescriptor_t wDesc,
                                             const miopenConvolutionDescriptor_t convDesc,
                                             const miopenTensorDescriptor_t dxDesc,
                                             const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, solution_id);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeBwdCtxAndProblem(handle, dyDesc, wDesc, convDesc, dxDesc);
        miopen::deref(convDesc).CompileSolution(ctx, problem, solution_id);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardDataImmediate(miopenHandle_t handle,
                                       const miopenTensorDescriptor_t dyDesc,
                                       const void* dy,
                                       const miopenTensorDescriptor_t wDesc,
                                       const void* w,
                                       const miopenConvolutionDescriptor_t convDesc,
                                       const miopenTensorDescriptor_t dxDesc,
                                       void* dx,
                                       void* workSpace,
                                       size_t workSpaceSize,
                                       const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(
        handle, dyDesc, wDesc, convDesc, dxDesc, workSpace, workSpaceSize, solution_id);
    miopen::debug::LogCmdConvolution(
        dxDesc, wDesc, convDesc, dyDesc, miopen::debug::ConvDirection::Bwd, true);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
        {
            miopen::deref(convDesc).ConvolutionForwardImmediate(miopen::deref(handle),
                                                                miopen::deref(wDesc),
                                                                DataCast(w),
                                                                miopen::deref(dyDesc),
                                                                DataCast(dy),
                                                                miopen::deref(dxDesc),
                                                                DataCast(dx),
                                                                DataCast(workSpace),
                                                                workSpaceSize,
                                                                solution_id);
        }
        else
        {
            miopen::deref(convDesc).ConvolutionBackwardImmediate(miopen::deref(handle),
                                                                 miopen::deref(dyDesc),
                                                                 DataCast(dy),
                                                                 miopen::deref(wDesc),
                                                                 DataCast(w),
                                                                 miopen::deref(dxDesc),
                                                                 DataCast(dx),
                                                                 DataCast(workSpace),
                                                                 workSpaceSize,
                                                                 solution_id);
        }
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolutionCount(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* solutionCount)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeWrWCtxAndProblem(handle, dyDesc, xDesc, convDesc, dwDesc);

        *solutionCount = miopen::deref(convDesc).GetSolutionCount(ctx, problem);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolution(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t dyDesc,
                                            const miopenTensorDescriptor_t xDesc,
                                            const miopenConvolutionDescriptor_t convDesc,
                                            const miopenTensorDescriptor_t dwDesc,
                                            const size_t maxSolutionCount,
                                            size_t* solutionCount,
                                            miopenConvSolution_t* solutions)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, maxSolutionCount);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeWrWCtxAndProblem(handle, dyDesc, xDesc, convDesc, dwDesc);

        const auto found =
            miopen::deref(convDesc).GetSolutions(ctx, problem, maxSolutionCount, nullptr);

        assert(found.size() <= maxSolutionCount);
        ReturnSolutions(found, solutionCount, solutions);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc,
    const uint64_t solution_id,
    size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, solution_id);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
        {
            *workSpaceSize =
                miopen::deref(convDesc).GetWrwSolutionWorkspaceSize(miopen::deref(handle),
                                                                    miopen::deref(xDesc),
                                                                    miopen::deref(dyDesc),
                                                                    miopen::deref(dwDesc),
                                                                    solution_id);
        }
        else
        {
            *workSpaceSize =
                miopen::deref(convDesc).GetWrwSolutionWorkspaceSize(miopen::deref(handle),
                                                                    miopen::deref(dyDesc),
                                                                    miopen::deref(xDesc),
                                                                    miopen::deref(dwDesc),
                                                                    solution_id);
        }
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsCompileSolution(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const miopenTensorDescriptor_t xDesc,
                                                const miopenConvolutionDescriptor_t convDesc,
                                                const miopenTensorDescriptor_t dwDesc,
                                                const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, solution_id);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeWrWCtxAndProblem(handle, dyDesc, xDesc, convDesc, dwDesc);
        miopen::deref(convDesc).CompileSolution(ctx, problem, solution_id);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsImmediate(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t dyDesc,
                                          const void* dy,
                                          const miopenTensorDescriptor_t xDesc,
                                          const void* x,
                                          const miopenConvolutionDescriptor_t convDesc,
                                          const miopenTensorDescriptor_t dwDesc,
                                          void* dw,
                                          void* workSpace,
                                          size_t workSpaceSize,
                                          const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(
        handle, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, workSpace, workSpaceSize, solution_id);
    miopen::debug::LogCmdConvolution(
        xDesc, dwDesc, convDesc, dyDesc, miopen::debug::ConvDirection::WrW, true);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
        {
            miopen::deref(convDesc).ConvolutionWrwImmediate(miopen::deref(handle),
                                                            miopen::deref(xDesc),
                                                            DataCast(x),
                                                            miopen::deref(dyDesc),
                                                            DataCast(dy),
                                                            miopen::deref(dwDesc),
                                                            DataCast(dw),
                                                            DataCast(workSpace),
                                                            workSpaceSize,
                                                            solution_id);
        }
        else
        {
            miopen::deref(convDesc).ConvolutionWrwImmediate(miopen::deref(handle),
                                                            miopen::deref(dyDesc),
                                                            DataCast(dy),
                                                            miopen::deref(xDesc),
                                                            DataCast(x),
                                                            miopen::deref(dwDesc),
                                                            DataCast(dw),
                                                            DataCast(workSpace),
                                                            workSpaceSize,
                                                            solution_id);
        }
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenFindConvolutionBackwardDataAlgorithm(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t dyDesc,
                                           const void* dy,
                                           const miopenTensorDescriptor_t wDesc,
                                           const void* w,
                                           const miopenConvolutionDescriptor_t convDesc,
                                           const miopenTensorDescriptor_t dxDesc,
                                           void* dx,
                                           const int requestAlgoCount,
                                           int* returnedAlgoCount,
                                           miopenConvAlgoPerf_t* perfResults,
                                           void* workSpace,
                                           size_t workSpaceSize,
                                           bool exhaustiveSearch)
{

    MIOPEN_LOG_FUNCTION(handle,
                        dyDesc,
                        dy,
                        wDesc,
                        w,
                        convDesc,
                        dxDesc,
                        dx,
                        requestAlgoCount,
                        returnedAlgoCount,
                        perfResults,
                        workSpace,
                        workSpaceSize,
                        exhaustiveSearch);

    miopen::debug::LogCmdFindConvolution(
        dxDesc, wDesc, convDesc, dyDesc, miopen::debug::ConvDirection::Bwd, false);

    if(miopen::deref(convDesc).mode == miopenTranspose)
    {
        return miopen::try_([&] {
            miopen::deref(convDesc).FindConvFwdAlgorithm(miopen::deref(handle),
                                                         miopen::deref(dyDesc),
                                                         DataCast(dy),
                                                         miopen::deref(wDesc),
                                                         DataCast(w),
                                                         miopen::deref(dxDesc),
                                                         DataCast(dx),
                                                         requestAlgoCount,
                                                         returnedAlgoCount,
                                                         perfResults,
                                                         DataCast(workSpace),
                                                         workSpaceSize,
                                                         exhaustiveSearch);

            for(int i = 0; i < *returnedAlgoCount; ++i)
            {
                // It is guaranteed that enum values are equal, see conv_algo_name.cpp
                perfResults[i].bwd_data_algo =
                    static_cast<miopenConvBwdDataAlgorithm_t>(perfResults[i].fwd_algo);
            }
        });
    }

    return miopen::try_([&] {
        miopen::deref(convDesc).FindConvBwdDataAlgorithm(miopen::deref(handle),
                                                         miopen::deref(dyDesc),
                                                         DataCast(dy),
                                                         miopen::deref(wDesc),
                                                         DataCast(w),
                                                         miopen::deref(dxDesc),
                                                         DataCast(dx),
                                                         requestAlgoCount,
                                                         returnedAlgoCount,
                                                         perfResults,
                                                         DataCast(workSpace),
                                                         workSpaceSize,
                                                         exhaustiveSearch);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardData(miopenHandle_t handle,
                              const void* alpha,
                              const miopenTensorDescriptor_t dyDesc,
                              const void* dy,
                              const miopenTensorDescriptor_t wDesc,
                              const void* w,
                              const miopenConvolutionDescriptor_t convDesc,
                              miopenConvBwdDataAlgorithm_t algo,
                              const void* beta,
                              const miopenTensorDescriptor_t dxDesc,
                              void* dx,
                              void* workSpace,
                              size_t workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle,
                        alpha,
                        dyDesc,
                        dy,
                        wDesc,
                        w,
                        convDesc,
                        algo,
                        beta,
                        dxDesc,
                        dx,
                        workSpace,
                        workSpaceSize);
    miopen::debug::LogCmdConvolution(
        dxDesc, wDesc, convDesc, dyDesc, miopen::debug::ConvDirection::Bwd, false);

    if(miopen::deref(convDesc).mode == miopenTranspose)
    {
        return miopen::try_([&] {
            // It is guaranteed that enum values are equal, see conv_algo_name.cpp
            const auto algo_trans = static_cast<miopenConvFwdAlgorithm_t>(algo);
            miopen::deref(convDesc).ConvolutionForward(miopen::deref(handle),
                                                       alpha,
                                                       miopen::deref(dyDesc),
                                                       DataCast(dy),
                                                       miopen::deref(wDesc),
                                                       DataCast(w),
                                                       algo_trans,
                                                       beta,
                                                       miopen::deref(dxDesc),
                                                       DataCast(dx),
                                                       DataCast(workSpace),
                                                       workSpaceSize);
        });
    }

    return miopen::try_([&] {
        miopen::deref(convDesc).ConvolutionBackwardData(miopen::deref(handle),
                                                        alpha,
                                                        miopen::deref(dyDesc),
                                                        DataCast(dy),
                                                        miopen::deref(wDesc),
                                                        DataCast(w),
                                                        algo,
                                                        beta,
                                                        miopen::deref(dxDesc),
                                                        DataCast(dx),
                                                        DataCast(workSpace),
                                                        workSpaceSize);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetWorkSpaceSize(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeBwdCtxAndProblem(handle, dyDesc, wDesc, convDesc, dxDesc);
        *workSpaceSize         = miopen::deref(convDesc).GetWorkSpaceSize(ctx, problem);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetWorkSpaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc);
    return miopen::try_([&] {
        auto ctx               = ExecutionContext{};
        auto problem           = ProblemDescription{};
        std::tie(ctx, problem) = MakeWrWCtxAndProblem(handle, dyDesc, xDesc, convDesc, dwDesc);
        *workSpaceSize         = miopen::deref(convDesc).GetWorkSpaceSize(ctx, problem);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenFindConvolutionBackwardWeightsAlgorithm(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const void* dy,
                                              const miopenTensorDescriptor_t xDesc,
                                              const void* x,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dwDesc,
                                              void* dw,
                                              const int requestAlgoCount,
                                              int* returnedAlgoCount,
                                              miopenConvAlgoPerf_t* perfResults,
                                              void* workSpace,
                                              size_t workSpaceSize,
                                              bool exhaustiveSearch)
{

    MIOPEN_LOG_FUNCTION(handle,
                        dyDesc,
                        dy,
                        xDesc,
                        x,
                        convDesc,
                        dwDesc,
                        dw,
                        requestAlgoCount,
                        returnedAlgoCount,
                        perfResults,
                        workSpace,
                        workSpaceSize,
                        exhaustiveSearch);
    miopen::debug::LogCmdFindConvolution(
        xDesc, dwDesc, convDesc, dyDesc, miopen::debug::ConvDirection::WrW, false);

    return miopen::try_([&] {
        const auto trans = (miopen::deref(convDesc).mode == miopenTranspose);
        miopen::deref(convDesc).FindConvBwdWeightsAlgorithm(
            miopen::deref(handle),
            trans ? miopen::deref(xDesc) : miopen::deref(dyDesc),
            trans ? DataCast(x) : DataCast(dy),
            trans ? miopen::deref(dyDesc) : miopen::deref(xDesc),
            trans ? DataCast(dy) : DataCast(x),
            miopen::deref(dwDesc),
            DataCast(dw),
            requestAlgoCount,
            returnedAlgoCount,
            perfResults,
            DataCast(workSpace),
            workSpaceSize,
            exhaustiveSearch);
    });
}

MIOPEN_EXPORT extern "C" miopenStatus_t
miopenConvolutionBackwardWeights(miopenHandle_t handle,
                                 const void* alpha,
                                 const miopenTensorDescriptor_t dyDesc,
                                 const void* dy,
                                 const miopenTensorDescriptor_t xDesc,
                                 const void* x,
                                 const miopenConvolutionDescriptor_t convDesc,
                                 miopenConvBwdWeightsAlgorithm_t algo,
                                 const void* beta,
                                 const miopenTensorDescriptor_t dwDesc,
                                 void* dw,
                                 void* workSpace,
                                 size_t workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle,
                        alpha,
                        dyDesc,
                        dy,
                        xDesc,
                        x,
                        convDesc,
                        algo,
                        beta,
                        dwDesc,
                        dw,
                        workSpace,
                        workSpaceSize);
    miopen::debug::LogCmdConvolution(
        xDesc, dwDesc, convDesc, dyDesc, miopen::debug::ConvDirection::WrW, false);

    return miopen::try_([&] {
        const auto trans = (miopen::deref(convDesc).mode == miopenTranspose);
        miopen::deref(convDesc).ConvolutionBackwardWeights(
            miopen::deref(handle),
            alpha,
            trans ? miopen::deref(xDesc) : miopen::deref(dyDesc),
            trans ? DataCast(x) : DataCast(dy),
            trans ? miopen::deref(dyDesc) : miopen::deref(xDesc),
            trans ? DataCast(dy) : DataCast(x),
            algo,
            beta,
            miopen::deref(dwDesc),
            DataCast(dw),
            DataCast(workSpace),
            workSpaceSize);
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenConvolutionBackwardBias(miopenHandle_t handle,
                                                        const void* alpha,
                                                        const miopenTensorDescriptor_t dyDesc,
                                                        const void* dy,
                                                        const void* beta,
                                                        const miopenTensorDescriptor_t dbDesc,
                                                        void* db)
{
    MIOPEN_LOG_FUNCTION(handle, alpha, dyDesc, dy, beta, dbDesc, db);
    // bfloat16 not supported for bias operation
    if(miopen::deref(dyDesc).GetType() == miopenBFloat16 ||
       miopen::deref(dbDesc).GetType() == miopenBFloat16)
    {
        return miopenStatusNotImplemented;
    }

    return miopen::try_([&] {
        ConvolutionBackwardBias(miopen::deref(handle),
                                alpha,
                                miopen::deref(dyDesc),
                                DataCast(dy),
                                beta,
                                miopen::deref(dbDesc),
                                DataCast(db));
    });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenSetConvolutionAttribute(miopenConvolutionDescriptor_t convDesc,
                                                        const miopenConvolutionAttrib_t attr,
                                                        const int value)
{
    MIOPEN_LOG_FUNCTION(convDesc, attr, value);
    return miopen::try_([&] { miopen::deref(convDesc).attribute.Set(attr, value); });
}

MIOPEN_EXPORT
extern "C" miopenStatus_t miopenGetConvolutionAttribute(miopenConvolutionDescriptor_t convDesc,
                                                        const miopenConvolutionAttrib_t attr,
                                                        int* const value)
{
    MIOPEN_LOG_FUNCTION(convDesc, attr);
    return miopen::try_(
        [&] { miopen::deref(value) = miopen::deref(convDesc).attribute.Get(attr); });
}
