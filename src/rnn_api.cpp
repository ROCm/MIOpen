
#include <miopen/rnn.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenCreateRNNDescriptor(miopenRNNDescriptor_t* rnnDesc)
{
    MIOPEN_LOG_FUNCTION(rnnDesc);
    return miopen::try_([&] { miopen::deref(rnnDesc) = new miopen::RNNDescriptor(); });
}

extern "C" miopenStatus_t miopenSetRNNDescriptor(miopenRNNDescriptor_t rnnDesc,
                                                 const int hsize,
                                                 const int nlayers,
                                                 miopenRNNInputMode_t inMode,
                                                 miopenRNNDirectionMode_t direction,
                                                 miopenRNNMode_t rnnMode,
                                                 miopenRNNAlgo_t algo,
                                                 miopenDataType_t dataType)
{

    MIOPEN_LOG_FUNCTION(rnnDesc, hsize, nlayers, inMode, direction, rnnMode, algo, dataType);
    return miopen::try_([&] {

        miopen::deref(rnnDesc) =
            miopen::RNNDescriptor(hsize, nlayers, inMode, direction, rnnMode, algo, dataType);
    });
}

/*
extern "C"
miopenStatus_t miopenGetRNNWorkspaceSize(miopenHandle_t handle,
                miopenRNNDescriptor_t           rnnDesc,
                const int                       sequenceLen,
                miopenTensorDescriptor_t        *xDesc,
                size_t          				*numBytes) {

        MIOPEN_LOG_FUNCTION(rnnDesc, sequenceLen, xDesc, numBytes);
        return miopen::try_([&] {
                miopen::deref(numBytes) = miopen::deref(rnnDesc).GetWorkspaceSize(
                        miopen::deref(handle),
                        sequenceLen,
                        miopen::deref(xDesc));
        });

}


extern "C"
miopenStatus_t miopenGetRNNTrainingReserveSize(miopenHandle_t handle,
                miopenRNNDescriptor_t       rnnDesc,
                int                         sequenceLen,
                miopenTensorDescriptor_t	*xDesc,
                size_t                      *numBytes) {

        MIOPEN_LOG_FUNCTION(rnnDesc, sequenceLen, xDesc, numBytes);
        return miopen::try_([&] {
                miopen::deref(numBytes) = miopen::deref(rnnDesc).GetReserveSize(
                        miopen::deref(handle),
                        sequenceLen,
                        miopen::deref(xDesc));
        });

}
*/

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

extern "C" miopenStatus_t miopenGetRNNLayerParam(miopenHandle_t handle,
                                                 miopenRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 miopenTensorDescriptor_t xDesc,
                                                 miopenTensorDescriptor_t wDesc,
                                                 const void* w,
                                                 const int layerID,
                                                 miopenTensorDescriptor_t paramDesc,
                                                 void** layerParam)
{

    // TODO (dlowell) implement this
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
                                                void** layerBias)
{

    // TODO (dlowell) implement this
    return miopenStatusSuccess;
}

extern "C" miopenStatus_t miopenDestroyRNNDescriptor(miopenRNNDescriptor_t rnnDesc)
{
    MIOPEN_LOG_FUNCTION(rnnDesc);
    return miopen::try_([&] { delete rnnDesc; });
}

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
        miopen::deref(rnnDesc).ForwardTrainRNNCell(miopen::deref(handle),
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
        miopen::deref(rnnDesc).BackwardDataRNNCell(miopen::deref(handle),
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
        miopen::deref(rnnDesc).BackwardWeightsRNNCell(miopen::deref(handle),
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
        miopen::deref(rnnDesc).ForwardInferRNNCell(miopen::deref(handle),
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
    });
}
