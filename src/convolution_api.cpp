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
#include <miopen/find_controls.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <algorithm>

// TODO: Make miopenConvAlgoPerf_t loggable
inline std::ostream& operator<<(std::ostream& os, miopenConvAlgoPerf_t) { return os; }

extern "C" miopenStatus_t miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t* convDesc)
{
    MIOPEN_LOG_FUNCTION(convDesc);
    return miopen::try_([&] { miopen::deref(convDesc) = new miopen::ConvolutionDescriptor(); });
}

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

extern "C" miopenStatus_t miopenInitConvolutionNdDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                            int spatialDim,
                                                            int* padA,
                                                            int* strideA,
                                                            int* dilationA,
                                                            miopenConvolutionMode_t c_mode)
{
    auto pads      = std::vector<int>(padA, padA + spatialDim);
    auto strides   = std::vector<int>(strideA, strideA + spatialDim);
    auto dilations = std::vector<int>(dilationA, dilationA + spatialDim);
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

extern "C" miopenStatus_t miopenSetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                         int groupCount)
{
    MIOPEN_LOG_FUNCTION(convDesc, groupCount);
    return miopen::try_([&] { miopen::deref(convDesc).group_count = groupCount; });
}

extern "C" miopenStatus_t miopenSetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc,
                                                       miopenConvolutionFindMode_t findMode)
{
    MIOPEN_LOG_FUNCTION(convDesc, findMode);
    return miopen::try_([&] {
        miopen::deref(convDesc).findMode.Set(static_cast<miopen::FindMode::Values>(findMode));
    });
}

extern "C" miopenStatus_t miopenGetConvolutionFindMode(const miopenConvolutionDescriptor_t convDesc,
                                                       miopenConvolutionFindMode_t* findMode)
{
    MIOPEN_LOG_FUNCTION(convDesc, findMode);
    return miopen::try_([&] {
        miopen::deref(findMode) =
            static_cast<miopenConvolutionFindMode_t>(miopen::deref(convDesc).findMode.Get());
    });
}

// Hidden C++ functions for MIGraphX.
extern "C" miopenStatus_t miopenHiddenSetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc,
                                                             int findMode)
{
    return miopen::try_([&] {
        miopen::deref(convDesc).findMode.Set(static_cast<miopen::FindMode::Values>(findMode));
    });
}
extern "C" miopenStatus_t miopenHiddenGetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc,
                                                             int* findMode)
{
    return miopen::try_([&] {
        miopen::deref(findMode) = static_cast<int>(miopen::deref(convDesc).findMode.Get());
    });
}

extern "C" miopenStatus_t
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

extern "C" miopenStatus_t miopenSetTransposeConvNdOutputPadding(
    miopenConvolutionDescriptor_t convDesc, int spatialDim, int* adjA)
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

extern "C" miopenStatus_t miopenGetConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                         miopenConvolutionMode_t* c_mode,
                                                         int* pad_h,
                                                         int* pad_w,
                                                         int* stride_h,
                                                         int* stride_w,
                                                         int* dilation_h,
                                                         int* dilation_w)
{
    MIOPEN_LOG_FUNCTION(convDesc, c_mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
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

extern "C" miopenStatus_t miopenGetConvolutionNdDescriptor(miopenConvolutionDescriptor_t convDesc,
                                                           int requestedSpatialDim,
                                                           int* spatialDim,
                                                           int* padA,
                                                           int* strideA,
                                                           int* dilationA,
                                                           miopenConvolutionMode_t* c_mode)
{
    MIOPEN_LOG_FUNCTION(
        convDesc, requestedSpatialDim, spatialDim, padA, strideA, dilationA, c_mode);
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

extern "C" miopenStatus_t
miopenGetConvolutionForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
                                     const miopenTensorDescriptor_t inputTensorDesc,
                                     const miopenTensorDescriptor_t filterDesc,
                                     int* n,
                                     int* c,
                                     int* h,
                                     int* w)
{
    MIOPEN_LOG_FUNCTION(convDesc, inputTensorDesc, filterDesc, n, c, h, w);
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

extern "C" miopenStatus_t
miopenGetConvolutionNdForwardOutputDim(miopenConvolutionDescriptor_t convDesc,
                                       const miopenTensorDescriptor_t inputTensorDesc,
                                       const miopenTensorDescriptor_t filterDesc,
                                       int* nDim,
                                       int* outputTensorDimA)
{
    MIOPEN_LOG_FUNCTION(convDesc, inputTensorDesc, filterDesc, nDim, outputTensorDimA);
    return miopen::try_([&] {
        auto out_desc = miopen::deref(convDesc).GetForwardOutputTensor(
            miopen::deref(inputTensorDesc), miopen::deref(filterDesc));

        miopen::deref(nDim) = out_desc.GetSize();

        for(int i = 0; i < out_desc.GetSize(); ++i)
        {
            outputTensorDimA[i] = out_desc.GetLengths()[i];
        }
    });
}

extern "C" miopenStatus_t miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc)
{
    MIOPEN_LOG_FUNCTION(convDesc);
    return miopen::try_([&] { miopen_destroy_object(convDesc); });
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetWorkSpaceSize(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle, wDesc, yDesc, convDesc, workSpaceSize);
    miopen::try_([&] {
        miopen::deref(workSpaceSize) =
            miopen::deref(convDesc).mode == miopenTranspose
                ? miopen::deref(convDesc).BackwardDataGetWorkSpaceSize(miopen::deref(handle),
                                                                       miopen::deref(wDesc),
                                                                       miopen::deref(xDesc),
                                                                       miopen::deref(yDesc))
                : miopen::deref(convDesc).ForwardGetWorkSpaceSize(miopen::deref(handle),
                                                                  miopen::deref(wDesc),
                                                                  miopen::deref(xDesc),
                                                                  miopen::deref(yDesc));
    });

    return (miopenStatusSuccess);
}

enum class ConvDirection
{
    Fwd = 1,
    Bwd = 2,
    WrW = 4
};

static void LogCmdConvolution(const miopenTensorDescriptor_t xDesc,
                              const miopenTensorDescriptor_t wDesc,
                              const miopenConvolutionDescriptor_t convDesc,
                              const miopenTensorDescriptor_t yDesc,
                              const ConvDirection conv_dir,
                              const bool is_immediate)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(miopen::deref(xDesc).GetType() == miopenHalf)
        {
            ss << "convfp16";
        }
        else if(miopen::deref(xDesc).GetType() == miopenBFloat16)
        {
            ss << "convbfp16";
        }
        else if(miopen::deref(xDesc).GetType() == miopenInt8 ||
                miopen::deref(xDesc).GetType() == miopenInt8x4)
        {
            ss << "convint8";
        }
        else
        {
            ss << "conv";
        }
        if(miopen::deref(convDesc).GetSpatialDimension() == 2)
        {
            ss << " -n " << miopen::deref(xDesc).GetLengths()[0] // clang-format off
                << " -c " << miopen::deref(xDesc).GetLengths()[1]
                << " -H " << miopen::deref(xDesc).GetLengths()[2]
                << " -W " << miopen::deref(xDesc).GetLengths()[3]
                << " -k " << miopen::deref(wDesc).GetLengths()[0]
                << " -y " << miopen::deref(wDesc).GetLengths()[2]
                << " -x " << miopen::deref(wDesc).GetLengths()[3]
                << " -p " << miopen::deref(convDesc).GetConvPads()[0]
                << " -q " << miopen::deref(convDesc).GetConvPads()[1]
                << " -u " << miopen::deref(convDesc).GetConvStrides()[0]
                << " -v " << miopen::deref(convDesc).GetConvStrides()[1]
                << " -l " << miopen::deref(convDesc).GetConvDilations()[0]
                << " -j " << miopen::deref(convDesc).GetConvDilations()[1]; // clang-format on
            std::string x_layout = miopen::deref(xDesc).GetLayout("NCHW");
            std::string w_layout = miopen::deref(wDesc).GetLayout("NCHW");
            std::string y_layout = miopen::deref(yDesc).GetLayout("NCHW");
            if(x_layout != "NCHW")
                ss << " --in_layout " << x_layout;
            if(w_layout != "NCHW")
                ss << " --fil_layout " << w_layout;
            if(y_layout != "NCHW")
                ss << " --out_layout " << y_layout;
        }
        else if(miopen::deref(convDesc).GetSpatialDimension() == 3)
        {
            ss << " -n " << miopen::deref(xDesc).GetLengths()[0] // clang-format off
                << " -c " << miopen::deref(xDesc).GetLengths()[1]
                << " --in_d " << miopen::deref(xDesc).GetLengths()[2]
                << " -H " << miopen::deref(xDesc).GetLengths()[3]
                << " -W " << miopen::deref(xDesc).GetLengths()[4]
                << " -k " << miopen::deref(wDesc).GetLengths()[0]
                << " --fil_d " << miopen::deref(wDesc).GetLengths()[2]
                << " -y " << miopen::deref(wDesc).GetLengths()[3]
                << " -x " << miopen::deref(wDesc).GetLengths()[4]
                << " --pad_d " << miopen::deref(convDesc).GetConvPads()[0]
                << " -p " << miopen::deref(convDesc).GetConvPads()[1]
                << " -q " << miopen::deref(convDesc).GetConvPads()[2]
                << " --conv_stride_d " << miopen::deref(convDesc).GetConvStrides()[0]
                << " -u " << miopen::deref(convDesc).GetConvStrides()[1]
                << " -v " << miopen::deref(convDesc).GetConvStrides()[2]
                << " --dilation_d " << miopen::deref(convDesc).GetConvDilations()[0]
                << " -l " << miopen::deref(convDesc).GetConvDilations()[1]
                << " -j " << miopen::deref(convDesc).GetConvDilations()[2]
                << " --spatial_dim 3"; // clang-format on
            std::string x_layout = miopen::deref(xDesc).GetLayout("NCDHW");
            std::string w_layout = miopen::deref(wDesc).GetLayout("NCDHW");
            std::string y_layout = miopen::deref(yDesc).GetLayout("NCDHW");
            if(x_layout != "NCDHW")
                ss << " --in_layout " << x_layout;
            if(w_layout != "NCDHW")
                ss << " --fil_layout " << w_layout;
            if(y_layout != "NCDHW")
                ss << " --out_layout " << y_layout;
        }
        ss << " -m " << (miopen::deref(convDesc).mode == 1 ? "trans" : "conv") // clang-format off
            << " -g " << miopen::deref(convDesc).group_count
            << " -F " << std::to_string(static_cast<int>(conv_dir))
            << " -t 1"; // clang-format on
        if(miopen::deref(xDesc).GetType() == miopenInt8x4)
            ss << " -Z 1";
        if(is_immediate)
            ss << " -S 0";
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
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

    /// workaround for previous trans conv logic
    if(miopen::deref(convDesc).mode == miopenTranspose)
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

extern "C" miopenStatus_t miopenConvolutionForward(miopenHandle_t handle,
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
    LogCmdConvolution(xDesc, wDesc, convDesc, yDesc, ConvDirection::Fwd, false);

    /// workaround for previous trans conv logic
    if(miopen::deref(convDesc).mode == miopenTranspose)
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

extern "C" miopenStatus_t miopenConvolutionForwardBias(miopenHandle_t handle,
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

extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionCount(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenTensorDescriptor_t xDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t yDesc,
                                         size_t* solutionCount)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            *solutionCount = miopen::deref(convDesc).GetBackwardSolutionCount(miopen::deref(handle),
                                                                              miopen::deref(xDesc),
                                                                              miopen::deref(wDesc),
                                                                              miopen::deref(yDesc));
        else
            *solutionCount = miopen::deref(convDesc).GetForwardSolutionCount(miopen::deref(handle),
                                                                             miopen::deref(wDesc),
                                                                             miopen::deref(xDesc),
                                                                             miopen::deref(yDesc));
    });
}

extern "C" miopenStatus_t
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
        if(miopen::deref(convDesc).mode == miopenTranspose)
            miopen::deref(convDesc).GetBackwardSolutions(miopen::deref(handle),
                                                         miopen::deref(xDesc),
                                                         miopen::deref(wDesc),
                                                         miopen::deref(yDesc),
                                                         maxSolutionCount,
                                                         solutionCount,
                                                         solutions,
                                                         nullptr);
        else
            miopen::deref(convDesc).GetForwardSolutions(miopen::deref(handle),
                                                        miopen::deref(wDesc),
                                                        miopen::deref(xDesc),
                                                        miopen::deref(yDesc),
                                                        maxSolutionCount,
                                                        solutionCount,
                                                        solutions,
                                                        nullptr);
    });
}

extern "C" miopenStatus_t
miopenConvolutionForwardGetSolutionWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t wDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t yDesc,
                                                 const uint64_t solution_id,
                                                 size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc, solution_id, workSpaceSize);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            *workSpaceSize =
                miopen::deref(convDesc).GetBackwardSolutionWorkspaceSize(miopen::deref(handle),
                                                                         miopen::deref(xDesc),
                                                                         miopen::deref(wDesc),
                                                                         miopen::deref(yDesc),
                                                                         solution_id);
        else
            *workSpaceSize =
                miopen::deref(convDesc).GetForwardSolutionWorkspaceSize(miopen::deref(handle),
                                                                        miopen::deref(wDesc),
                                                                        miopen::deref(xDesc),
                                                                        miopen::deref(yDesc),
                                                                        solution_id);
    });
}

extern "C" miopenStatus_t
miopenConvolutionForwardCompileSolution(miopenHandle_t handle,
                                        const miopenTensorDescriptor_t wDesc,
                                        const miopenTensorDescriptor_t xDesc,
                                        const miopenConvolutionDescriptor_t convDesc,
                                        const miopenTensorDescriptor_t yDesc,
                                        const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(handle, wDesc, xDesc, convDesc, yDesc, solution_id);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            miopen::deref(convDesc).CompileBackwardSolution(miopen::deref(handle),
                                                            miopen::deref(xDesc),
                                                            miopen::deref(wDesc),
                                                            miopen::deref(yDesc),
                                                            solution_id);
        else
            miopen::deref(convDesc).CompileForwardSolution(miopen::deref(handle),
                                                           miopen::deref(wDesc),
                                                           miopen::deref(xDesc),
                                                           miopen::deref(yDesc),
                                                           solution_id);
    });
}

extern "C" miopenStatus_t
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
    LogCmdConvolution(xDesc, wDesc, convDesc, yDesc, ConvDirection::Fwd, true);

    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
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
        else
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
    });
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionCount(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* solutionCount)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            *solutionCount = miopen::deref(convDesc).GetForwardSolutionCount(miopen::deref(handle),
                                                                             miopen::deref(wDesc),
                                                                             miopen::deref(dyDesc),
                                                                             miopen::deref(dxDesc));
        else
            *solutionCount =
                miopen::deref(convDesc).GetBackwardSolutionCount(miopen::deref(handle),
                                                                 miopen::deref(dyDesc),
                                                                 miopen::deref(wDesc),
                                                                 miopen::deref(dxDesc));
    });
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolution(miopenHandle_t handle,
                                         const miopenTensorDescriptor_t dyDesc,
                                         const miopenTensorDescriptor_t wDesc,
                                         const miopenConvolutionDescriptor_t convDesc,
                                         const miopenTensorDescriptor_t dxDesc,
                                         const size_t maxSolutionCount,
                                         size_t* solutionCount,
                                         miopenConvSolution_t* solutions)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, maxSolutionCount, solutionCount);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            miopen::deref(convDesc).GetForwardSolutions(miopen::deref(handle),
                                                        miopen::deref(wDesc),
                                                        miopen::deref(dyDesc),
                                                        miopen::deref(dxDesc),
                                                        maxSolutionCount,
                                                        solutionCount,
                                                        solutions,
                                                        nullptr);

        else
            miopen::deref(convDesc).GetBackwardSolutions(miopen::deref(handle),
                                                         miopen::deref(dyDesc),
                                                         miopen::deref(wDesc),
                                                         miopen::deref(dxDesc),
                                                         maxSolutionCount,
                                                         solutionCount,
                                                         solutions,
                                                         nullptr);
    });
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetSolutionWorkspaceSize(miopenHandle_t handle,
                                                      const miopenTensorDescriptor_t dyDesc,
                                                      const miopenTensorDescriptor_t wDesc,
                                                      const miopenConvolutionDescriptor_t convDesc,
                                                      const miopenTensorDescriptor_t dxDesc,
                                                      const uint64_t solution_id,
                                                      size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, solution_id, workSpaceSize);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            *workSpaceSize =
                miopen::deref(convDesc).GetForwardSolutionWorkspaceSize(miopen::deref(handle),
                                                                        miopen::deref(wDesc),
                                                                        miopen::deref(dyDesc),
                                                                        miopen::deref(dxDesc),
                                                                        solution_id);
        else
            *workSpaceSize =
                miopen::deref(convDesc).GetBackwardSolutionWorkspaceSize(miopen::deref(handle),
                                                                         miopen::deref(dyDesc),
                                                                         miopen::deref(wDesc),
                                                                         miopen::deref(dxDesc),
                                                                         solution_id);
    });
}

extern "C" miopenStatus_t
miopenConvolutionBackwardDataCompileSolution(miopenHandle_t handle,
                                             const miopenTensorDescriptor_t dyDesc,
                                             const miopenTensorDescriptor_t wDesc,
                                             const miopenConvolutionDescriptor_t convDesc,
                                             const miopenTensorDescriptor_t dxDesc,
                                             const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, solution_id);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            miopen::deref(convDesc).CompileForwardSolution(miopen::deref(handle),
                                                           miopen::deref(wDesc),
                                                           miopen::deref(dyDesc),
                                                           miopen::deref(dxDesc),
                                                           solution_id);
        else
            miopen::deref(convDesc).CompileBackwardSolution(miopen::deref(handle),
                                                            miopen::deref(dyDesc),
                                                            miopen::deref(wDesc),
                                                            miopen::deref(dxDesc),
                                                            solution_id);
    });
}

extern "C" miopenStatus_t
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
    LogCmdConvolution(dxDesc, wDesc, convDesc, dyDesc, ConvDirection::Bwd, true);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
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
        else
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
    });
}
extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolutionCount(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* solutionCount)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, solutionCount);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            *solutionCount = miopen::deref(convDesc).GetWrwSolutionCount(miopen::deref(handle),
                                                                         miopen::deref(xDesc),
                                                                         miopen::deref(dyDesc),
                                                                         miopen::deref(dwDesc));
        else
            *solutionCount = miopen::deref(convDesc).GetWrwSolutionCount(miopen::deref(handle),
                                                                         miopen::deref(dyDesc),
                                                                         miopen::deref(xDesc),
                                                                         miopen::deref(dwDesc));
    });
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetSolution(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t dyDesc,
                                            const miopenTensorDescriptor_t xDesc,
                                            const miopenConvolutionDescriptor_t convDesc,
                                            const miopenTensorDescriptor_t dwDesc,
                                            const size_t maxSolutionCount,
                                            size_t* solutionCount,
                                            miopenConvSolution_t* solutions)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, maxSolutionCount, solutionCount);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            miopen::deref(convDesc).GetWrwSolutions(miopen::deref(handle),
                                                    miopen::deref(xDesc),
                                                    miopen::deref(dyDesc),
                                                    miopen::deref(dwDesc),
                                                    maxSolutionCount,
                                                    solutionCount,
                                                    solutions,
                                                    nullptr);
        else
            miopen::deref(convDesc).GetWrwSolutions(miopen::deref(handle),
                                                    miopen::deref(dyDesc),
                                                    miopen::deref(xDesc),
                                                    miopen::deref(dwDesc),
                                                    maxSolutionCount,
                                                    solutionCount,
                                                    solutions,
                                                    nullptr);
    });
}

extern "C" miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc,
    const uint64_t solution_id,
    size_t* workSpaceSize)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, solution_id, workSpaceSize);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            *workSpaceSize =
                miopen::deref(convDesc).GetWrwSolutionWorkspaceSize(miopen::deref(handle),
                                                                    miopen::deref(xDesc),
                                                                    miopen::deref(dyDesc),
                                                                    miopen::deref(dwDesc),
                                                                    solution_id);
        else
            *workSpaceSize =
                miopen::deref(convDesc).GetWrwSolutionWorkspaceSize(miopen::deref(handle),
                                                                    miopen::deref(dyDesc),
                                                                    miopen::deref(xDesc),
                                                                    miopen::deref(dwDesc),
                                                                    solution_id);
    });
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsCompileSolution(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const miopenTensorDescriptor_t xDesc,
                                                const miopenConvolutionDescriptor_t convDesc,
                                                const miopenTensorDescriptor_t dwDesc,
                                                const uint64_t solution_id)
{
    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, solution_id);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
            miopen::deref(convDesc).CompileWrwSolution(miopen::deref(handle),
                                                       miopen::deref(xDesc),
                                                       miopen::deref(dyDesc),
                                                       miopen::deref(dwDesc),
                                                       solution_id);
        else
            miopen::deref(convDesc).CompileWrwSolution(miopen::deref(handle),
                                                       miopen::deref(dyDesc),
                                                       miopen::deref(xDesc),
                                                       miopen::deref(dwDesc),
                                                       solution_id);
    });
}

extern "C" miopenStatus_t
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
    LogCmdConvolution(xDesc, dwDesc, convDesc, dyDesc, ConvDirection::WrW, true);
    return miopen::try_([&] {
        if(miopen::deref(convDesc).mode == miopenTranspose)
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
        else
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
    });
}

extern "C" miopenStatus_t
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

    /// workaround for previous trans conv logic
    if(miopen::deref(convDesc).mode == miopenTranspose)
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

extern "C" miopenStatus_t
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
    LogCmdConvolution(dxDesc, wDesc, convDesc, dyDesc, ConvDirection::Bwd, false);

    /// workaround for previous trans conv logic
    if(miopen::deref(convDesc).mode == miopenTranspose)
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

extern "C" miopenStatus_t
miopenConvolutionBackwardDataGetWorkSpaceSize(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t dyDesc,
                                              const miopenTensorDescriptor_t wDesc,
                                              const miopenConvolutionDescriptor_t convDesc,
                                              const miopenTensorDescriptor_t dxDesc,
                                              size_t* workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle, dyDesc, wDesc, convDesc, dxDesc, workSpaceSize);
    return miopen::try_([&] {
        miopen::deref(workSpaceSize) =
            miopen::deref(convDesc).mode == miopenTranspose
                ? miopen::deref(convDesc).ForwardGetWorkSpaceSize(miopen::deref(handle),
                                                                  miopen::deref(wDesc),
                                                                  miopen::deref(dyDesc),
                                                                  miopen::deref(dxDesc))
                : miopen::deref(convDesc).BackwardDataGetWorkSpaceSize(miopen::deref(handle),
                                                                       miopen::deref(wDesc),
                                                                       miopen::deref(dyDesc),
                                                                       miopen::deref(dxDesc));
    });
}

extern "C" miopenStatus_t
miopenConvolutionBackwardWeightsGetWorkSpaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t dyDesc,
                                                 const miopenTensorDescriptor_t xDesc,
                                                 const miopenConvolutionDescriptor_t convDesc,
                                                 const miopenTensorDescriptor_t dwDesc,
                                                 size_t* workSpaceSize)
{

    MIOPEN_LOG_FUNCTION(handle, dyDesc, xDesc, convDesc, dwDesc, workSpaceSize);
    return miopen::try_([&] {
        miopen::deref(workSpaceSize) = miopen::deref(convDesc).BackwardWeightsGetWorkSpaceSize(
            miopen::deref(handle),
            miopen::deref(convDesc).mode == miopenTranspose ? miopen::deref(xDesc)
                                                            : miopen::deref(dyDesc),
            miopen::deref(convDesc).mode == miopenTranspose ? miopen::deref(dyDesc)
                                                            : miopen::deref(xDesc),
            miopen::deref(dwDesc));
    });
}

extern "C" miopenStatus_t
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
    LogCmdConvolution(xDesc, dwDesc, convDesc, dyDesc, ConvDirection::WrW, false);

    return miopen::try_([&] {
        miopen::deref(convDesc).FindConvBwdWeightsAlgorithm(
            miopen::deref(handle),
            /// workaround for previous trans conv logic
            miopen::deref(convDesc).mode == miopenTranspose ? miopen::deref(xDesc)
                                                            : miopen::deref(dyDesc),
            miopen::deref(convDesc).mode == miopenTranspose ? DataCast(x) : DataCast(dy),
            miopen::deref(convDesc).mode == miopenTranspose ? miopen::deref(dyDesc)
                                                            : miopen::deref(xDesc),
            miopen::deref(convDesc).mode == miopenTranspose ? DataCast(dy) : DataCast(x),
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

extern "C" miopenStatus_t
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
    LogCmdConvolution(xDesc, dwDesc, convDesc, dyDesc, ConvDirection::WrW, false);

    return miopen::try_([&] {
        miopen::deref(convDesc).ConvolutionBackwardWeights(
            miopen::deref(handle),
            alpha,
            /// workaround for previous trans conv logic
            miopen::deref(convDesc).mode == miopenTranspose ? miopen::deref(xDesc)
                                                            : miopen::deref(dyDesc),
            miopen::deref(convDesc).mode == miopenTranspose ? DataCast(x) : DataCast(dy),
            miopen::deref(convDesc).mode == miopenTranspose ? miopen::deref(dyDesc)
                                                            : miopen::deref(xDesc),
            miopen::deref(convDesc).mode == miopenTranspose ? DataCast(dy) : DataCast(x),
            algo,
            beta,
            miopen::deref(dwDesc),
            DataCast(dw),
            DataCast(workSpace),
            workSpaceSize);
    });
}

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
