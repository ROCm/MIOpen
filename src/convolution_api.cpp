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
#include <miopen/convolution.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

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
        miopen::deref(convDesc) = miopen::ConvolutionDescriptor(c_mode,
                                                                miopenPaddingDefault,
                                                                {pad_h, pad_w},
                                                                {stride_h, stride_w},
                                                                {dilation_h, dilation_w});
    });
}

extern "C" miopenStatus_t miopenSetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                                                         int groupCount)
{

    MIOPEN_LOG_FUNCTION(convDesc, groupCount);
    return miopen::try_([&] { miopen::deref(convDesc).group_count = groupCount; });
}

extern "C" miopenStatus_t
miopenSetTransposeConvOutputPadding(miopenConvolutionDescriptor_t convDesc, int adj_h, int adj_w)
{

    MIOPEN_LOG_FUNCTION(convDesc, adj_h, adj_w);
    return miopen::try_([&] {
        miopen::deref(convDesc).trans_output_pads[0] = adj_h;
        miopen::deref(convDesc).trans_output_pads[1] = adj_w;
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
        miopen::deref(c_mode)     = miopen::deref(convDesc).mode;
        miopen::deref(pad_h)      = miopen::deref(convDesc).GetConvPads()[0];
        miopen::deref(pad_w)      = miopen::deref(convDesc).GetConvPads()[1];
        miopen::deref(stride_h)   = miopen::deref(convDesc).GetConvStrides()[0];
        miopen::deref(stride_w)   = miopen::deref(convDesc).GetConvStrides()[1];
        miopen::deref(dilation_h) = miopen::deref(convDesc).GetConvDilations()[0];
        miopen::deref(dilation_w) = miopen::deref(convDesc).GetConvDilations()[1];
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
        miopen::tie_deref(n, c, h, w) = miopen::deref(convDesc).GetForwardOutputDim(
            miopen::deref(inputTensorDesc), miopen::deref(filterDesc));
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

    MIOPEN_LOG_FUNCTION(wDesc, yDesc, convDesc, workSpaceSize);
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

    MIOPEN_LOG_FUNCTION(xDesc,
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
            std::vector<miopenConvAlgoPerf_t> perfResults_trans(requestAlgoCount);
            miopen::deref(convDesc).FindConvBwdDataAlgorithm(miopen::deref(handle),
                                                             miopen::deref(xDesc),
                                                             DataCast(x),
                                                             miopen::deref(wDesc),
                                                             DataCast(w),
                                                             miopen::deref(yDesc),
                                                             DataCast(y),
                                                             requestAlgoCount,
                                                             returnedAlgoCount,
                                                             perfResults_trans.data(),
                                                             DataCast(workSpace),
                                                             workSpaceSize,
                                                             exhaustiveSearch);

            int i = 0;
            for(auto perfResults_temp : perfResults_trans)
            {
                perfResults_temp.fwd_algo =
                    perfResults_temp.bwd_data_algo == miopenConvBwdDataAlgorithm_t(1)
                        ? miopenConvFwdAlgorithm_t(1)
                        : (perfResults_temp.bwd_data_algo == miopenConvBwdDataAlgorithm_t(3)
                               ? miopenConvFwdAlgorithm_t(3)
                               : (perfResults_temp.bwd_data_algo == miopenConvBwdDataAlgorithm_t(0)
                                      ? miopenConvFwdAlgorithm_t(0)
                                      : (perfResults_temp.bwd_data_algo ==
                                                 miopenConvBwdDataAlgorithm_t(2)
                                             ? miopenConvFwdAlgorithm_t(2)
                                             : miopenConvFwdAlgorithm_t(4))));
                *(perfResults + (i++)) = perfResults_temp;
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

    MIOPEN_LOG_FUNCTION(
        alpha, xDesc, x, wDesc, w, convDesc, algo, beta, yDesc, y, workSpace, workSpaceSize);

    if(miopen::IsLoggingCmd())
    {
        if(miopen::deref(xDesc).GetType() == miopenHalf)
        {
            std::cerr << MIOPEN_DRIVER_CMD("convfp16");
        }
        else if(miopen::deref(xDesc).GetType() == miopenInt8)
        {
            std::cerr << MIOPEN_DRIVER_CMD("convint8");
        }
        else
        {
            std::cerr << MIOPEN_DRIVER_CMD("conv");
        }
        std::cerr << " -n " << miopen::deref(xDesc).GetLengths()[0] << " -c "
                  << miopen::deref(xDesc).GetLengths()[1] << " -H "
                  << miopen::deref(xDesc).GetLengths()[2] << " -W "
                  << miopen::deref(xDesc).GetLengths()[3]

                  << " -k " << miopen::deref(wDesc).GetLengths()[0] << " -y "
                  << miopen::deref(wDesc).GetLengths()[2] << " -x "
                  << miopen::deref(wDesc).GetLengths()[3]

                  << " -p " << miopen::deref(convDesc).GetConvPads()[0] << " -q "
                  << miopen::deref(convDesc).GetConvPads()[1] << " -u "
                  << miopen::deref(convDesc).GetConvStrides()[0] << " -v "
                  << miopen::deref(convDesc).GetConvStrides()[1] << " -l "
                  << miopen::deref(convDesc).GetConvDilations()[0] << " -j "
                  << miopen::deref(convDesc).GetConvDilations()[1] << " -m "
                  << (miopen::deref(convDesc).mode == 1 ? "trans" : "conv") << " -g "
                  << miopen::deref(convDesc).group_count << " -t "
                  << "1"
                  << "\n";
    }

    /// workaround for previous trans conv logic
    if(miopen::deref(convDesc).mode == miopenTranspose)
        return miopen::try_([&] {
            auto algo_trans = algo == miopenConvFwdAlgorithm_t(1)
                                  ? miopenConvBwdDataAlgorithm_t(1)
                                  : (algo == miopenConvFwdAlgorithm_t(3)
                                         ? miopenConvBwdDataAlgorithm_t(3)
                                         : (algo == miopenConvFwdAlgorithm_t(0)
                                                ? miopenConvBwdDataAlgorithm_t(0)
                                                : (algo == miopenConvFwdAlgorithm_t(2)
                                                       ? miopenConvBwdDataAlgorithm_t(2)
                                                       : miopenConvBwdDataAlgorithm_t(4))));

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

    MIOPEN_LOG_FUNCTION(alpha, bDesc, b, beta, yDesc, y);
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
miopenFindConvolutionBackwardDataAlgorithm(miopenHandle_t handle,
                                           const miopenTensorDescriptor_t dyDesc,
                                           const void* dy,
                                           const miopenTensorDescriptor_t wDesc,
                                           const void* w,
                                           const miopenConvolutionDescriptor_t convDesc,
                                           const miopenTensorDescriptor_t dxDesc,
                                           const void* dx,
                                           const int requestAlgoCount,
                                           int* returnedAlgoCount,
                                           miopenConvAlgoPerf_t* perfResults,
                                           void* workSpace,
                                           size_t workSpaceSize,
                                           bool exhaustiveSearch)
{

    MIOPEN_LOG_FUNCTION(dyDesc,
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
            std::vector<miopenConvAlgoPerf_t> perfResults_trans(requestAlgoCount);
            miopen::deref(convDesc).FindConvFwdAlgorithm(miopen::deref(handle),
                                                         miopen::deref(dyDesc),
                                                         DataCast(dy),
                                                         miopen::deref(wDesc),
                                                         DataCast(w),
                                                         miopen::deref(dxDesc),
                                                         DataCast(dx),
                                                         requestAlgoCount,
                                                         returnedAlgoCount,
                                                         perfResults_trans.data(),
                                                         DataCast(workSpace),
                                                         workSpaceSize,
                                                         exhaustiveSearch);

            int i = 0;
            for(auto perfResults_temp : perfResults_trans)
            {
                perfResults_temp.bwd_data_algo =
                    perfResults_temp.fwd_algo == miopenConvFwdAlgorithm_t(1)
                        ? miopenConvBwdDataAlgorithm_t(1)
                        : (perfResults_temp.fwd_algo == miopenConvFwdAlgorithm_t(3)
                               ? miopenConvBwdDataAlgorithm_t(3)
                               : (perfResults_temp.fwd_algo == miopenConvFwdAlgorithm_t(0)
                                      ? miopenConvBwdDataAlgorithm_t(0)
                                      : (perfResults_temp.fwd_algo == miopenConvFwdAlgorithm_t(2)
                                             ? miopenConvBwdDataAlgorithm_t(2)
                                             : miopenConvBwdDataAlgorithm_t(4))));
                *(perfResults + (i++)) = perfResults_temp;
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

    MIOPEN_LOG_FUNCTION(
        alpha, dyDesc, dy, wDesc, w, convDesc, algo, beta, dxDesc, dx, workSpace, workSpaceSize);

    if(miopen::IsLoggingCmd())
    {
        if(miopen::deref(dyDesc).GetType() == miopenHalf)
        {
            std::cerr << MIOPEN_DRIVER_CMD("convfp16");
        }
        else
        {
            std::cerr << MIOPEN_DRIVER_CMD("conv");
        }
        std::cerr << " -n " << miopen::deref(dxDesc).GetLengths()[0] << " -c "
                  << miopen::deref(dxDesc).GetLengths()[1] << " -H "
                  << miopen::deref(dxDesc).GetLengths()[2] << " -W "
                  << miopen::deref(dxDesc).GetLengths()[3]

                  << " -k " << miopen::deref(wDesc).GetLengths()[0] << " -y "
                  << miopen::deref(wDesc).GetLengths()[2] << " -x "
                  << miopen::deref(wDesc).GetLengths()[3]

                  << " -p " << miopen::deref(convDesc).GetConvPads()[0] << " -q "
                  << miopen::deref(convDesc).GetConvPads()[1] << " -u "
                  << miopen::deref(convDesc).GetConvStrides()[0] << " -v "
                  << miopen::deref(convDesc).GetConvStrides()[1] << " -l "
                  << miopen::deref(convDesc).GetConvDilations()[0] << " -j "
                  << miopen::deref(convDesc).GetConvDilations()[1] << " -m "
                  << (miopen::deref(convDesc).mode == 1 ? "trans" : "conv") << " -g "
                  << miopen::deref(convDesc).group_count << " -t "
                  << "1"
                  << "\n";
    }

    /// workaround for previous trans conv logic
    if(miopen::deref(convDesc).mode == miopenTranspose)
        return miopen::try_([&] {
            auto algo_trans = algo == miopenConvBwdDataAlgorithm_t(1)
                                  ? miopenConvFwdAlgorithm_t(1)
                                  : (algo == miopenConvBwdDataAlgorithm_t(3)
                                         ? miopenConvFwdAlgorithm_t(3)
                                         : (algo == miopenConvBwdDataAlgorithm_t(0)
                                                ? miopenConvFwdAlgorithm_t(0)
                                                : (algo == miopenConvBwdDataAlgorithm_t(2)
                                                       ? miopenConvFwdAlgorithm_t(2)
                                                       : miopenConvFwdAlgorithm_t(4))));

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

    MIOPEN_LOG_FUNCTION(dyDesc, wDesc, convDesc, dxDesc, workSpaceSize);
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

    MIOPEN_LOG_FUNCTION(dyDesc, xDesc, convDesc, dwDesc, workSpaceSize);
    return miopen::try_([&] {
        miopen::deref(workSpaceSize) =
            miopen::deref(convDesc).ConvolutionBackwardWeightsGetWorkSpaceSize(
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

    MIOPEN_LOG_FUNCTION(dyDesc,
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

    if(miopen::IsLoggingCmd())
    {
        if(miopen::deref(xDesc).GetType() == miopenHalf)
        {
            std::cerr << MIOPEN_DRIVER_CMD("convfp16");
        }
        else
        {
            std::cerr << MIOPEN_DRIVER_CMD("conv");
        }
        std::cerr << " -n " << miopen::deref(xDesc).GetLengths()[0] << " -c "
                  << miopen::deref(xDesc).GetLengths()[1] << " -H "
                  << miopen::deref(xDesc).GetLengths()[2] << " -W "
                  << miopen::deref(xDesc).GetLengths()[3]

                  << " -k " << miopen::deref(dwDesc).GetLengths()[0] << " -y "
                  << miopen::deref(dwDesc).GetLengths()[2] << " -x "
                  << miopen::deref(dwDesc).GetLengths()[3]

                  << " -p " << miopen::deref(convDesc).GetConvPads()[0] << " -q "
                  << miopen::deref(convDesc).GetConvPads()[1] << " -u "
                  << miopen::deref(convDesc).GetConvStrides()[0] << " -v "
                  << miopen::deref(convDesc).GetConvStrides()[1] << " -l "
                  << miopen::deref(convDesc).GetConvDilations()[0] << " -j "
                  << miopen::deref(convDesc).GetConvDilations()[1] << " -m "
                  << (miopen::deref(convDesc).mode == 1 ? "trans" : "conv") << " -g "
                  << miopen::deref(convDesc).group_count << " -t "
                  << "1"
                  << "\n";
    }

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

    MIOPEN_LOG_FUNCTION(
        alpha, dyDesc, dy, xDesc, x, convDesc, algo, beta, dwDesc, dw, workSpace, workSpaceSize);
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
    MIOPEN_LOG_FUNCTION(alpha, dyDesc, dy, beta, dbDesc, db);
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
