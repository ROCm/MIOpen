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

#include <miopen/rnn.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <vector>
// TODO: Make miopenConvAlgoPerf_t loggable
// inline std::ostream& operator<<(std::ostream& os, miopenConvAlgoPerf_t) { return os; }

extern "C" miopenStatus_t miopenCreateRNNDescriptor(miopenRNNDescriptor_t* rnnDesc)
{
    MIOPEN_LOG_FUNCTION(rnnDesc);
    return miopen::try_([&] { miopen::deref(rnnDesc) = new miopen::RNNDescriptor(); });
}

extern "C" miopenStatus_t miopenDestroyRNNDescriptor(miopenRNNDescriptor_t rnnDesc)
{
    MIOPEN_LOG_FUNCTION(rnnDesc);
    return miopen::try_([&] { delete rnnDesc; });
}

extern "C" miopenStatus_t miopenGetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                 miopenRNNMode_t* rnnMode,
                                                 miopenRNNAlgo_t* algoMode,
                                                 miopenRNNInputMode_t* inputMode,
                                                 miopenRNNDirectionMode_t* dirMode,
                                                 miopenRNNBiasMode_t* biasMode,
                                                 int* hiddenSize,
                                                 int* layer)
{

    MIOPEN_LOG_FUNCTION(
        rnnDesc, rnnMode, algoMode, inputMode, dirMode, biasMode, hiddenSize, layer);
    return miopen::try_([&] {
        if(rnnMode != nullptr)
        {
            miopen::deref(rnnMode) = miopen::deref(rnnDesc).rnnMode;
        }
        if(algoMode != nullptr)
        {
            miopen::deref(algoMode) = miopen::deref(rnnDesc).algoMode;
        }
        if(inputMode != nullptr)
        {
            miopen::deref(inputMode) = miopen::deref(rnnDesc).inputMode;
        }
        if(layer != nullptr)
        {
            miopen::deref(layer) = miopen::deref(rnnDesc).nLayers;
        }
        if(biasMode != nullptr)
        {
            miopen::deref(biasMode) = miopen::deref(rnnDesc).biasMode;
        }
        if(dirMode != nullptr)
        {
            miopen::deref(dirMode) = miopen::deref(rnnDesc).dirMode;
        }
        if(hiddenSize != nullptr)
        {
            miopen::deref(hiddenSize) = miopen::deref(rnnDesc).hsize;
        }

    });
}

extern "C" miopenStatus_t miopenSetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                 const int hsize,
                                                 const int nlayers,
                                                 miopenRNNInputMode_t inMode,
                                                 miopenRNNDirectionMode_t direction,
                                                 miopenRNNMode_t rnnMode,
                                                 miopenRNNBiasMode_t biasMode,
                                                 miopenRNNAlgo_t algo,
                                                 miopenDataType_t dataType)
{

    MIOPEN_LOG_FUNCTION(
        rnnDesc, hsize, nlayers, inMode, direction, rnnMode, biasMode, algo, dataType);
    return miopen::try_([&] {

        miopen::deref(rnnDesc) = miopen::RNNDescriptor(
            hsize, nlayers, rnnMode, inMode, direction, biasMode, algo, dataType);
    });
}

extern "C" miopenStatus_t miopenGetRNNWorkspaceSize(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int seqLen,
                                                    miopenTensorDescriptor_t* xDesc,
                                                    size_t* numBytes)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, seqLen, xDesc, numBytes);
    miopen::c_array_view<miopenTensorDescriptor_t> xDescArray{xDesc, size_t(seqLen)};
    return miopen::try_([&] {
        miopen::deref(numBytes) =
            miopen::deref(rnnDesc).GetWorkspaceSize(miopen::deref(handle), seqLen, xDescArray);
    });
}

extern "C" miopenStatus_t miopenGetRNNTrainingReserveSize(miopenHandle_t handle,
                                                          miopenRNNDescriptor_t rnnDesc,
                                                          int seqLen,
                                                          miopenTensorDescriptor_t* xDesc,
                                                          size_t* numBytes)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, seqLen, xDesc, numBytes);
    miopen::c_array_view<miopenTensorDescriptor_t> xDescArray{xDesc, size_t(seqLen)};
    return miopen::try_([&] {
        miopen::deref(numBytes) =
            miopen::deref(rnnDesc).GetReserveSize(miopen::deref(handle), seqLen, xDescArray);
    });
}

extern "C" miopenStatus_t miopenGetRNNParamsDescriptor(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       miopenTensorDescriptor_t xDesc,
                                                       miopenTensorDescriptor_t wDesc,
                                                       miopenDataType_t dtype)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, xDesc, wDesc, dtype);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).GetParamsDescriptor(
            miopen::deref(handle), miopen::deref(xDesc), miopen::deref(wDesc), dtype);
    });
}

extern "C" miopenStatus_t miopenGetRNNParamsSize(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 miopenTensorDescriptor_t xDesc,
                                                 size_t* numBytes,
                                                 miopenDataType_t dtype)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, xDesc, numBytes, dtype);
    return miopen::try_([&] {
        miopen::deref(numBytes) = miopen::deref(rnnDesc).GetParamsSize(
            miopen::deref(handle), miopen::deref(xDesc), dtype);
    });
}

extern "C" miopenStatus_t miopenGetRNNInputSuperTensorSize(miopenHandle_t handle,
                                                           miopenRNNDescriptor_t rnnDesc,
                                                           const int seqLen,
                                                           miopenTensorDescriptor_t* xDesc,
                                                           size_t* numBytes)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, seqLen, xDesc, numBytes);
    miopen::c_array_view<miopenTensorDescriptor_t> xDescArray{xDesc, size_t(seqLen)};
    return miopen::try_([&] {
        miopen::deref(numBytes) = miopen::deref(rnnDesc).GetRNNInputSuperTensorSize(
            miopen::deref(handle), seqLen, xDescArray);
    });
}

extern "C" miopenStatus_t miopenGetRNNHiddenSuperTensorSize(miopenHandle_t handle,
                                                            miopenRNNDescriptor_t rnnDesc,
                                                            const int seqLen,
                                                            miopenTensorDescriptor_t* xDesc,
                                                            size_t* numBytes)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, xDesc, numBytes);
    miopen::c_array_view<miopenTensorDescriptor_t> xDescArray{xDesc, size_t(seqLen)};
    return miopen::try_([&] {
        miopen::deref(numBytes) =
            miopen::deref(rnnDesc).GetRNNHiddenSuperTensorSize(miopen::deref(handle), xDescArray);
    });
}

/* Get weight super tensor size
temporary function assuming output matrix exists */
extern "C" miopenStatus_t miopenGetRNNWeightSuperTensorSize(miopenHandle_t handle,
                                                            miopenRNNDescriptor_t rnnDesc,
                                                            size_t* numBytes,
                                                            miopenTensorDescriptor_t xDesc,
                                                            miopenTensorDescriptor_t yDesc)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, numBytes, xDesc, yDesc);
    return miopen::try_([&] {
        miopen::deref(numBytes) = miopen::deref(rnnDesc).GetRNNWeightSuperTensorSize(
            miopen::deref(handle), miopen::deref(xDesc), miopen::deref(yDesc));
    });
}

extern "C" miopenStatus_t miopenGetRNNLayerParam(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenTensorDescriptor_t wDesc,
                                                 const void* w,
                                                 const int layerID,
                                                 miopenTensorDescriptor_t paramDesc,
                                                 void* layerParam)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, layer, xDesc, wDesc, w, layerID, paramDesc, layerParam);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).GetLayerParam(miopen::deref(handle),
                                             layer,
                                             miopen::deref(xDesc),
                                             miopen::deref(wDesc),
                                             DataCast(w),
                                             layerID,
                                             miopen::deref(paramDesc),
                                             DataCast(layerParam));
    });
    return miopenStatusSuccess;
}

extern "C" miopenStatus_t miopenGetRNNLayerBias(miopenHandle_t handle,
                                                miopenRNNDescriptor_t rnnDesc,
                                                const int layer,
                                                miopenTensorDescriptor_t xDesc,
                                                miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                const int layerID,
                                                miopenTensorDescriptor_t biasDesc,
                                                void* layerBias)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, layer, xDesc, wDesc, w, layerID, biasDesc, layerBias);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).GetLayerBias(miopen::deref(handle),
                                            layer,
                                            miopen::deref(xDesc),
                                            miopen::deref(wDesc),
                                            DataCast(w),
                                            layerID,
                                            miopen::deref(biasDesc),
                                            DataCast(layerBias));
    });
    return miopenStatusSuccess;
}

extern "C" miopenStatus_t miopenSetRNNLayerParam(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenTensorDescriptor_t wDesc,
                                                 void* w,
                                                 const int layerID,
                                                 miopenTensorDescriptor_t paramDesc,
                                                 void* layerParam)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, layer, xDesc, wDesc, w, layerID, paramDesc, layerParam);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).SetLayerParam(miopen::deref(handle),
                                             layer,
                                             miopen::deref(xDesc),
                                             miopen::deref(wDesc),
                                             DataCast(w),
                                             layerID,
                                             miopen::deref(paramDesc),
                                             DataCast(layerParam));
    });
    return miopenStatusSuccess;
}

extern "C" miopenStatus_t miopenSetRNNLayerBias(miopenHandle_t handle,
                                                miopenRNNDescriptor_t rnnDesc,
                                                const int layer,
                                                miopenTensorDescriptor_t xDesc,
                                                miopenTensorDescriptor_t wDesc,
                                                void* w,
                                                const int layerID,
                                                miopenTensorDescriptor_t biasDesc,
                                                void* layerBias)
{
    MIOPEN_LOG_FUNCTION(rnnDesc, layer, xDesc, wDesc, w, layerID, biasDesc, layerBias);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).SetLayerBias(miopen::deref(handle),
                                            layer,
                                            miopen::deref(xDesc),
                                            miopen::deref(wDesc),
                                            DataCast(w),
                                            layerID,
                                            miopen::deref(biasDesc),
                                            DataCast(layerBias));
    });
    return miopenStatusSuccess;
}

extern "C" miopenStatus_t miopenRNNForwardTraining(miopenHandle_t handle,
                                                   const miopenRNNDescriptor_t rnnDesc,
                                                   const int sequenceLen,
                                                   miopenTensorDescriptor_t* xDesc,
                                                   const void* x,
                                                   const miopenTensorDescriptor_t hxDesc,
                                                   const void* hx,
                                                   const miopenTensorDescriptor_t cxDesc,
                                                   const void* cx,
                                                   const miopenTensorDescriptor_t wDesc,
                                                   const void* w,
                                                   miopenTensorDescriptor_t* yDesc,
                                                   void* y,
                                                   const miopenTensorDescriptor_t hyDesc,
                                                   void* hy,
                                                   const miopenTensorDescriptor_t cyDesc,
                                                   void* cy,
                                                   void* workSpace,
                                                   size_t workSpaceNumBytes,
                                                   void* reserveSpace,
                                                   size_t reserveSpaceNumBytes)
{

    MIOPEN_LOG_FUNCTION(rnnDesc,
                        sequenceLen,
                        xDesc,
                        x,
                        hxDesc,
                        hx,
                        cxDesc,
                        cx,
                        wDesc,
                        w,
                        yDesc,
                        y,
                        hyDesc,
                        hy,
                        cyDesc,
                        cy,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);
    return miopen::try_([&] {

        miopen::c_array_view<miopenTensorDescriptor_t> xDescArray{xDesc, size_t(sequenceLen)};
        miopen::c_array_view<miopenTensorDescriptor_t> yDescArray{yDesc, size_t(sequenceLen)};
        miopen::deref(rnnDesc).RNNForwardTraining(miopen::deref(handle),
                                                  sequenceLen,
                                                  xDescArray,
                                                  DataCast(x),
                                                  miopen::deref(hxDesc),
                                                  DataCast(hx),
                                                  miopen::deref(cxDesc),
                                                  DataCast(cx),
                                                  miopen::deref(wDesc),
                                                  DataCast(w),
                                                  yDescArray,
                                                  DataCast(y),
                                                  miopen::deref(hyDesc),
                                                  DataCast(hy),
                                                  miopen::deref(cyDesc),
                                                  DataCast(cy),
                                                  DataCast(workSpace),
                                                  workSpaceNumBytes,
                                                  DataCast(reserveSpace),
                                                  reserveSpaceNumBytes);
    });
}

extern "C" miopenStatus_t miopenRNNBackwardData(miopenHandle_t handle,
                                                const miopenRNNDescriptor_t rnnDesc,
                                                const int sequenceLen,
                                                miopenTensorDescriptor_t* yDesc,
                                                const void* y,
                                                miopenTensorDescriptor_t* dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t dhyDesc,
                                                const void* dhy,
                                                const miopenTensorDescriptor_t dcyDesc,
                                                const void* dcy,
                                                const miopenTensorDescriptor_t wDesc,
                                                const void* w,
                                                const miopenTensorDescriptor_t hxDesc,
                                                const void* hx,
                                                const miopenTensorDescriptor_t cxDesc,
                                                const void* cx,
                                                miopenTensorDescriptor_t* dxDesc,
                                                void* dx,
                                                const miopenTensorDescriptor_t dhxDesc,
                                                void* dhx,
                                                const miopenTensorDescriptor_t dcxDesc,
                                                void* dcx,
                                                void* workSpace,
                                                size_t workSpaceNumBytes,
                                                const void* reserveSpace,
                                                size_t reserveSpaceNumBytes)
{
    MIOPEN_LOG_FUNCTION(rnnDesc,
                        sequenceLen,
                        yDesc,
                        y,
                        dyDesc,
                        dy,
                        dhyDesc,
                        dhy,
                        dcyDesc,
                        dcy,
                        wDesc,
                        w,
                        hxDesc,
                        hx,
                        cxDesc,
                        cx,
                        dxDesc,
                        dx,
                        dhxDesc,
                        dhx,
                        dcxDesc,
                        dcx,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);
    return miopen::try_([&] {

        miopen::c_array_view<miopenTensorDescriptor_t> yDescArray{yDesc, size_t(sequenceLen)};
        miopen::c_array_view<miopenTensorDescriptor_t> dyDescArray{dyDesc, size_t(sequenceLen)};
        miopen::c_array_view<miopenTensorDescriptor_t> dxDescArray{dxDesc, size_t(sequenceLen)};
        miopen::deref(rnnDesc).RNNBackwardData(miopen::deref(handle),
                                               sequenceLen,
                                               yDescArray,
                                               DataCast(y),
                                               dyDescArray,
                                               DataCast(dy),
                                               miopen::deref(dhyDesc),
                                               DataCast(dhy),
                                               miopen::deref(dcyDesc),
                                               DataCast(dcy),
                                               miopen::deref(wDesc),
                                               DataCast(w),
                                               miopen::deref(hxDesc),
                                               DataCast(hx),
                                               miopen::deref(cxDesc),
                                               DataCast(cx),
                                               dxDescArray,
                                               DataCast(dx),
                                               miopen::deref(dhxDesc),
                                               DataCast(dhx),
                                               miopen::deref(dcxDesc),
                                               DataCast(dcx),
                                               DataCast(workSpace),
                                               workSpaceNumBytes,
                                               DataCast(reserveSpace),
                                               reserveSpaceNumBytes);
    });
}

miopenStatus_t miopenRNNBackwardWeights(miopenHandle_t handle,
                                        const miopenRNNDescriptor_t rnnDesc,
                                        const int sequenceLen,
                                        miopenTensorDescriptor_t* xDesc,
                                        const void* x,
                                        const miopenTensorDescriptor_t hxDesc,
                                        const void* hx,
                                        miopenTensorDescriptor_t* dyDesc,
                                        const void* dy,
                                        const miopenTensorDescriptor_t dwDesc,
                                        void* dw,
                                        const void* workSpace,
                                        size_t workSpaceNumBytes,
                                        const void* reserveSpace,
                                        size_t reserveSpaceNumBytes)
{
    MIOPEN_LOG_FUNCTION(rnnDesc,
                        sequenceLen,
                        xDesc,
                        x,
                        hxDesc,
                        hx,
                        dyDesc,
                        dy,
                        dwDesc,
                        dw,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);
    return miopen::try_([&] {

        miopen::c_array_view<miopenTensorDescriptor_t> xDescArray{xDesc, size_t(sequenceLen)};
        miopen::c_array_view<miopenTensorDescriptor_t> dyDescArray{dyDesc, size_t(sequenceLen)};
        miopen::deref(rnnDesc).RNNBackwardWeights(miopen::deref(handle),
                                                  sequenceLen,
                                                  xDescArray,
                                                  DataCast(x),
                                                  miopen::deref(hxDesc),
                                                  DataCast(hx),
                                                  dyDescArray,
                                                  DataCast(dy),
                                                  miopen::deref(dwDesc),
                                                  DataCast(dw),
                                                  DataCast(workSpace),
                                                  workSpaceNumBytes,
                                                  DataCast(reserveSpace),
                                                  reserveSpaceNumBytes);
    });
}

extern "C" miopenStatus_t miopenRNNForwardInference(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    const int sequenceLen,
                                                    miopenTensorDescriptor_t* xDesc,
                                                    const void* x,
                                                    miopenTensorDescriptor_t hxDesc,
                                                    const void* hx,
                                                    miopenTensorDescriptor_t cxDesc,
                                                    const void* cx,
                                                    miopenTensorDescriptor_t wDesc,
                                                    const void* w,
                                                    miopenTensorDescriptor_t* yDesc,
                                                    void* y,
                                                    miopenTensorDescriptor_t hyDesc,
                                                    void* hy,
                                                    miopenTensorDescriptor_t cyDesc,
                                                    void* cy,
                                                    void* workSpace,
                                                    size_t workSpaceNumBytes)
{

    MIOPEN_LOG_FUNCTION(rnnDesc,
                        sequenceLen,
                        xDesc,
                        x,
                        hxDesc,
                        hx,
                        cxDesc,
                        cx,
                        wDesc,
                        w,
                        yDesc,
                        y,
                        hyDesc,
                        hy,
                        cyDesc,
                        cy,
                        workSpace,
                        workSpaceNumBytes);
    return miopen::try_([&] {
        miopen::c_array_view<miopenTensorDescriptor_t> xDescArray{xDesc, size_t(sequenceLen)};
        miopen::c_array_view<miopenTensorDescriptor_t> yDescArray{yDesc, size_t(sequenceLen)};
        miopen::deref(rnnDesc).RNNForwardInference(miopen::deref(handle),
                                                   sequenceLen,
                                                   xDescArray,
                                                   DataCast(x),
                                                   miopen::deref(hxDesc),
                                                   DataCast(hx),
                                                   miopen::deref(cxDesc),
                                                   DataCast(cx),
                                                   miopen::deref(wDesc),
                                                   DataCast(w),
                                                   yDescArray,
                                                   DataCast(y),
                                                   miopen::deref(hyDesc),
                                                   DataCast(hy),
                                                   miopen::deref(cyDesc),
                                                   DataCast(cy),
                                                   DataCast(workSpace),
                                                   workSpaceNumBytes);
    });
}

// CELL APIs below ---------------------------------

extern "C" miopenStatus_t miopenRNNForwardTrainCell(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    miopenTensorDescriptor_t xDesc,
                                                    const void* x,
                                                    miopenTensorDescriptor_t hxDesc,
                                                    const void* hx,
                                                    miopenTensorDescriptor_t wDesc,
                                                    const void* w,
                                                    miopenTensorDescriptor_t yDesc,
                                                    void* y,
                                                    miopenTensorDescriptor_t hyDesc,
                                                    void* hy,
                                                    void* workSpace,
                                                    size_t workSpaceNumBytes,
                                                    void* reserveSpace,
                                                    size_t reserveSpaceNumBytes)
{

    MIOPEN_LOG_FUNCTION(rnnDesc,
                        xDesc,
                        x,
                        hxDesc,
                        hx,
                        wDesc,
                        w,
                        yDesc,
                        y,
                        hyDesc,
                        hy,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).ForwardRNNTrainCell(miopen::deref(handle),
                                                   miopen::deref(xDesc),
                                                   DataCast(x),
                                                   miopen::deref(hxDesc),
                                                   DataCast(hx),
                                                   miopen::deref(wDesc),
                                                   DataCast(w),
                                                   miopen::deref(yDesc),
                                                   DataCast(y),
                                                   miopen::deref(hyDesc),
                                                   DataCast(hy),
                                                   DataCast(workSpace),
                                                   workSpaceNumBytes,
                                                   DataCast(reserveSpace),
                                                   reserveSpaceNumBytes);
    });
}

extern "C" miopenStatus_t miopenRNNBackwardDataCell(miopenHandle_t handle,
                                                    miopenRNNDescriptor_t rnnDesc,
                                                    miopenTensorDescriptor_t yDesc,
                                                    const void* y,
                                                    miopenTensorDescriptor_t dyDesc,
                                                    const void* dy,
                                                    miopenTensorDescriptor_t dhyDesc,
                                                    const void* dhy,
                                                    miopenTensorDescriptor_t wDesc,
                                                    const void* w,
                                                    miopenTensorDescriptor_t hxDesc,
                                                    const void* hx,
                                                    miopenTensorDescriptor_t dxDesc,
                                                    void* dx,
                                                    miopenTensorDescriptor_t dhxDesc,
                                                    void* dhx,
                                                    void* workSpace,
                                                    size_t workSpaceNumBytes,
                                                    const void* reserveSpace,
                                                    size_t reserveSpaceNumBytes)
{

    MIOPEN_LOG_FUNCTION(rnnDesc,
                        yDesc,
                        y,
                        dyDesc,
                        dy,
                        dhyDesc,
                        dhy,
                        wDesc,
                        w,
                        hxDesc,
                        hx,
                        dxDesc,
                        dx,
                        dhxDesc,
                        dhx,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).BackwardRNNDataCell(miopen::deref(handle),
                                                   miopen::deref(yDesc),
                                                   DataCast(y),
                                                   miopen::deref(dyDesc),
                                                   DataCast(dy),
                                                   miopen::deref(dhyDesc),
                                                   DataCast(dhy),
                                                   miopen::deref(wDesc),
                                                   DataCast(w),
                                                   miopen::deref(hxDesc),
                                                   DataCast(hx),
                                                   miopen::deref(dxDesc),
                                                   DataCast(dx),
                                                   miopen::deref(dhxDesc),
                                                   DataCast(dhx),
                                                   DataCast(workSpace),
                                                   workSpaceNumBytes,
                                                   DataCast(reserveSpace),
                                                   reserveSpaceNumBytes);
    });
}

extern "C" miopenStatus_t miopenRNNBackwardWeightsCell(miopenHandle_t handle,
                                                       miopenRNNDescriptor_t rnnDesc,
                                                       miopenTensorDescriptor_t xDesc,
                                                       const void* x,
                                                       miopenTensorDescriptor_t hxDesc,
                                                       const void* hx,
                                                       miopenTensorDescriptor_t yDesc,
                                                       const void* y,
                                                       miopenTensorDescriptor_t dwDesc,
                                                       void* dw,
                                                       const void* workSpace,
                                                       size_t workSpaceNumBytes,
                                                       const void* reserveSpace,
                                                       size_t reserveSpaceNumBytes)
{

    MIOPEN_LOG_FUNCTION(rnnDesc,
                        xDesc,
                        x,
                        hxDesc,
                        hx,
                        yDesc,
                        y,
                        dwDesc,
                        dw,
                        workSpace,
                        workSpaceNumBytes,
                        reserveSpace,
                        reserveSpaceNumBytes);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).BackwardRNNWeightsCell(miopen::deref(handle),
                                                      miopen::deref(xDesc),
                                                      DataCast(x),
                                                      miopen::deref(hxDesc),
                                                      DataCast(hx),
                                                      miopen::deref(yDesc),
                                                      DataCast(y),
                                                      miopen::deref(dwDesc),
                                                      DataCast(dw),
                                                      DataCast(workSpace),
                                                      workSpaceNumBytes,
                                                      DataCast(reserveSpace),
                                                      reserveSpaceNumBytes);
    });
}

extern "C" miopenStatus_t miopenRNNForwardInferenceCell(miopenHandle_t handle,
                                                        miopenRNNDescriptor_t rnnDesc,
                                                        miopenTensorDescriptor_t xDesc,
                                                        const void* x,
                                                        miopenTensorDescriptor_t hxDesc,
                                                        const void* hx,
                                                        miopenTensorDescriptor_t wDesc,
                                                        const void* w,
                                                        miopenTensorDescriptor_t yDesc,
                                                        void* y,
                                                        miopenTensorDescriptor_t hyDesc,
                                                        void* hy,
                                                        void* workSpace,
                                                        size_t workSpaceNumBytes)
{

    MIOPEN_LOG_FUNCTION(rnnDesc,
                        xDesc,
                        x,
                        hxDesc,
                        hx,
                        wDesc,
                        w,
                        yDesc,
                        y,
                        hyDesc,
                        hy,
                        workSpace,
                        workSpaceNumBytes);
    return miopen::try_([&] {
        miopen::deref(rnnDesc).ForwardRNNInferCell(miopen::deref(handle),
                                                   miopen::deref(xDesc),
                                                   DataCast(x),
                                                   miopen::deref(hxDesc),
                                                   DataCast(hx),
                                                   miopen::deref(wDesc),
                                                   DataCast(w),
                                                   miopen::deref(yDesc),
                                                   DataCast(y),
                                                   miopen::deref(hyDesc),
                                                   DataCast(hy),
                                                   DataCast(workSpace),
                                                   workSpaceNumBytes);
        //>>>>>>> rnn
    });
}
