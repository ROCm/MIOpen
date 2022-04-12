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
#include <array>
#include <initializer_list>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenCreateTensorDescriptor(miopenTensorDescriptor_t* tensorDesc)
{
    MIOPEN_LOG_FUNCTION(tensorDesc);
    return miopen::try_([&] { miopen::deref(tensorDesc) = new miopen::TensorDescriptor(); });
}

extern "C" miopenStatus_t miopenSet4dTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, dataType, n, c, h, w);
    return miopen::try_([&] {
        std::initializer_list<int> lens = {n, c, h, w};
        miopen::deref(tensorDesc)       = miopen::TensorDescriptor(dataType, lens.begin(), 4);
    });
}

extern "C" miopenStatus_t miopenGet4dTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                      miopenDataType_t* dataType,
                                                      int* n,
                                                      int* c,
                                                      int* h,
                                                      int* w,
                                                      int* nStride,
                                                      int* cStride,
                                                      int* hStride,
                                                      int* wStride)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
    return miopen::try_([&] {
        miopen::deref(dataType)       = miopen::deref(tensorDesc).GetType();
        miopen::tie_deref(n, c, h, w) = miopen::tien<4>(miopen::deref(tensorDesc).GetLengths());
        miopen::tie_deref(nStride, cStride, hStride, wStride) =
            miopen::tien<4>(miopen::deref(tensorDesc).GetStrides());
    });
}

// Internal API
// MD: This should not be required to be exported. Temporary hack
MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptorLengths(
    miopenTensorDescriptor_t tensorDesc, int* n, int* c, int* h, int* w)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, n, c, h, w);
    return miopen::try_([&] {
        miopen::tie_deref(n, c, h, w) = miopen::tien<4>(miopen::deref(tensorDesc).GetLengths());
    });
}

// Internal API
MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptorStrides(
    miopenTensorDescriptor_t tensorDesc, int* nStride, int* cStride, int* hStride, int* wStride)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, nStride, cStride, hStride, wStride);
    return miopen::try_([&] {
        miopen::tie_deref(nStride, cStride, hStride, wStride) =
            miopen::tien<4>(miopen::deref(tensorDesc).GetStrides());
    });
}

MIOPEN_EXPORT miopenStatus_t miopenGet5dTensorDescriptorLengths(
    miopenTensorDescriptor_t tensorDesc, int* n, int* c, int* d, int* h, int* w)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, n, c, d, h, w);
    return miopen::try_([&] {
        miopen::tie_deref(n, c, d, h, w) = miopen::tien<5>(miopen::deref(tensorDesc).GetLengths());
    });
}

// Internal API
MIOPEN_EXPORT miopenStatus_t miopenGet5dTensorDescriptorStrides(miopenTensorDescriptor_t tensorDesc,
                                                                int* nStride,
                                                                int* cStride,
                                                                int* dStride,
                                                                int* hStride,
                                                                int* wStride)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, nStride, cStride, dStride, hStride, wStride);
    return miopen::try_([&] {
        miopen::tie_deref(nStride, cStride, dStride, hStride, wStride) =
            miopen::tien<5>(miopen::deref(tensorDesc).GetStrides());
    });
}

extern "C" miopenStatus_t miopenSetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                    miopenDataType_t dataType,
                                                    int nbDims,
                                                    int* dimsA,
                                                    int* stridesA)
{
    if(miopen::IsLoggingFunctionCalls())
    {
        const miopen::logger::CArray<int, int> dim(dimsA, nbDims);
        const miopen::logger::CArray<int, int> stride(stridesA, nbDims);
        MIOPEN_LOG_FUNCTION(tensorDesc, dataType, nbDims, dim.values, stride.values);
    }

    return miopen::try_([&] {
        if(stridesA == nullptr)
        {
            miopen::deref(tensorDesc) = miopen::TensorDescriptor(dataType, dimsA, nbDims);
        }
        else
        {
            miopen::deref(tensorDesc) = miopen::TensorDescriptor(dataType, dimsA, stridesA, nbDims);
        }
    });
}

extern "C" miopenStatus_t miopenGetTensorNumBytes(miopenTensorDescriptor_t tensorDesc,
                                                  size_t* numBytes)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, numBytes);
    return miopen::try_([&] { miopen::deref(numBytes) = miopen::deref(tensorDesc).GetNumBytes(); });
}

// Internal API
int miopenGetTensorDescriptorElementSize(miopenTensorDescriptor_t tensorDesc)
{
    return miopen::deref(tensorDesc).GetElementSize();
}

extern "C" miopenStatus_t miopenGetTensorDescriptorSize(miopenTensorDescriptor_t tensorDesc,
                                                        int* size)
{
    MIOPEN_LOG_FUNCTION(tensorDesc, size);
    return miopen::try_([&] { miopen::deref(size) = miopen::deref(tensorDesc).GetSize(); });
}

extern "C" miopenStatus_t miopenGetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                                    miopenDataType_t* dataType,
                                                    int* dimsA,
                                                    int* stridesA)
{

    MIOPEN_LOG_FUNCTION(tensorDesc, dataType, dimsA, stridesA);
    return miopen::try_([&] {
        if(dataType != nullptr)
        {
            *dataType = miopen::deref(tensorDesc).GetType();
        }
        if(dimsA != nullptr)
        {
            std::copy(miopen::deref(tensorDesc).GetLengths().begin(),
                      miopen::deref(tensorDesc).GetLengths().end(),
                      dimsA);
        }
        if(stridesA != nullptr)
        {
            std::copy(miopen::deref(tensorDesc).GetStrides().begin(),
                      miopen::deref(tensorDesc).GetStrides().end(),
                      stridesA);
        }
    });
}

extern "C" miopenStatus_t miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc)
{
    MIOPEN_LOG_FUNCTION(tensorDesc);
    return miopen::try_([&] { miopen_destroy_object(tensorDesc); });
}

static void LogCmdTensorOp(miopenTensorOp_t tensorOp,
                           const void* alpha,
                           const void* alpha2,
                           const void* beta,
                           const miopenTensorDescriptor_t aDesc,
                           bool is_set,
                           bool is_scale)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(miopen::deref(aDesc).GetType() == miopenHalf)
            ss << "tensoropfp16";
        else
            ss << "tensorop";
        assert(!(is_set && is_scale));
        if(!is_set && !is_scale)
        {
            // clang-format off
            ss << " -A " << std::to_string(*static_cast<const float*>(alpha)) 
               << " -B " << std::to_string(*static_cast<const float*>(alpha2)) 
               << " -G " << std::to_string(*static_cast<const float*>(beta));
            // clang-format on
        }
        // clang-format off
        ss << " -n " << miopen::deref(aDesc).GetLengths()[0] 
           << " -c " << miopen::deref(aDesc).GetLengths()[1]
           << " -H " << miopen::deref(aDesc).GetLengths()[2] 
           << " -W " << miopen::deref(aDesc).GetLengths()[3];
        // clag-format on
        if(is_set)
            ss << " -o 0 -v " << std::to_string(*static_cast<const float*>(alpha));
        else if(is_scale)
            ss << " -o 1 -v " << std::to_string(*static_cast<const float*>(alpha));
        else
            ss << " -o " << static_cast<int>(tensorOp) + 2;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenOpTensor(miopenHandle_t handle,
                                         miopenTensorOp_t tensorOp,
                                         const void* alpha1,
                                         const miopenTensorDescriptor_t aDesc,
                                         const void* A,
                                         const void* alpha2,
                                         const miopenTensorDescriptor_t bDesc,
                                         const void* B,
                                         const void* beta,
                                         const miopenTensorDescriptor_t cDesc,
                                         void* C)
{

    MIOPEN_LOG_FUNCTION(tensorOp, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
    LogCmdTensorOp(tensorOp, alpha1, alpha2, beta, aDesc, false, false);
    return miopen::try_([&] {
        OpTensor(miopen::deref(handle),
                 tensorOp,
                 alpha1,
                 miopen::deref(aDesc),
                 DataCast(A),
                 alpha2,
                 miopen::deref(bDesc),
                 DataCast(B),
                 beta,
                 miopen::deref(cDesc),
                 DataCast(C));
    });
}

extern "C" miopenStatus_t miopenSetTensor(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t yDesc,
                                          void* y,
                                          const void* alpha)
{

    MIOPEN_LOG_FUNCTION(handle, yDesc, y, alpha);
    LogCmdTensorOp(miopenTensorOpAdd, alpha, nullptr, nullptr, yDesc, true, false);
    return miopen::try_(
        [&] { SetTensor(miopen::deref(handle), miopen::deref(yDesc), DataCast(y), alpha); });
}

extern "C" miopenStatus_t miopenScaleTensor(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y,
                                            const void* alpha)
{

    MIOPEN_LOG_FUNCTION(handle, yDesc, y, alpha);
    LogCmdTensorOp(miopenTensorOpAdd, alpha,nullptr, nullptr, yDesc, false, true);
    return miopen::try_(
        [&] { ScaleTensor(miopen::deref(handle), miopen::deref(yDesc), DataCast(y), alpha); });
}

extern "C" miopenStatus_t miopenTransformTensor(miopenHandle_t handle,
                                                const void* alpha,
                                                const miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const void* beta,
                                                const miopenTensorDescriptor_t yDesc,
                                                void* y)
{
    // dstValue = alpha[0]*srcValue + beta[0]*priorDstValue
    MIOPEN_LOG_FUNCTION(handle, alpha, xDesc, x, beta, yDesc, y);
    float zero = 0.0f;
    LogCmdTensorOp(miopenTensorOpAdd, alpha, beta, static_cast<void*>(&zero), xDesc, false, false);
    return miopen::try_([&] {
        TransformTensor(miopen::deref(handle),
                        alpha,
                        miopen::deref(xDesc),
                        DataCast(x),
                        beta,
                        miopen::deref(yDesc),
                        DataCast(y));
    });
}
