/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include <miopen/reducetensor.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <algorithm>

extern "C" miopenStatus_t
miopenCreateReduceTensorDescriptor(miopenReduceTensorDescriptor_t* reduceTensorDesc)
{
    MIOPEN_LOG_FUNCTION(reduceTensorDesc);
    return miopen::try_(
        [&] { miopen::deref(reduceTensorDesc) = new miopen::ReduceTensorDescriptor(); });
};

extern "C" miopenStatus_t
miopenDestroyReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc)
{
    MIOPEN_LOG_FUNCTION(reduceTensorDesc);
    return miopen::try_([&] { miopen_destroy_object(reduceTensorDesc); });
};

extern "C" miopenStatus_t
miopenSetReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc,
                                miopenReduceTensorOp_t reduceTensorOp,
                                miopenDataType_t reduceTensorCompType,
                                miopenNanPropagation_t reduceTensorNanOpt,
                                miopenReduceTensorIndices_t reduceTensorIndices,
                                miopenIndicesType_t reduceTensorIndicesType)
{
    MIOPEN_LOG_FUNCTION(reduceTensorDesc,
                        reduceTensorOp,
                        reduceTensorCompType,
                        reduceTensorNanOpt,
                        reduceTensorIndices,
                        reduceTensorIndicesType);
    return miopen::try_([&] {
        miopen::deref(reduceTensorDesc) = miopen::ReduceTensorDescriptor(reduceTensorOp,
                                                                         reduceTensorCompType,
                                                                         reduceTensorNanOpt,
                                                                         reduceTensorIndices,
                                                                         reduceTensorIndicesType);
    });
};

extern "C" miopenStatus_t
miopenGetReduceTensorDescriptor(const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                miopenReduceTensorOp_t* reduceTensorOp,
                                miopenDataType_t* reduceTensorCompType,
                                miopenNanPropagation_t* reduceTensorNanOpt,
                                miopenReduceTensorIndices_t* reduceTensorIndices,
                                miopenIndicesType_t* reduceTensorIndicesType)
{
    MIOPEN_LOG_FUNCTION(reduceTensorDesc,
                        reduceTensorOp,
                        reduceTensorCompType,
                        reduceTensorNanOpt,
                        reduceTensorIndices,
                        reduceTensorIndicesType);
    return miopen::try_([&] {
        miopen::deref(reduceTensorOp)       = miopen::deref(reduceTensorDesc).reduceTensorOp_;
        miopen::deref(reduceTensorCompType) = miopen::deref(reduceTensorDesc).reduceTensorCompType_;
        miopen::deref(reduceTensorNanOpt)   = miopen::deref(reduceTensorDesc).reduceTensorNanOpt_;
        miopen::deref(reduceTensorIndices)  = miopen::deref(reduceTensorDesc).reduceTensorIndices_;
        miopen::deref(reduceTensorIndicesType) =
            miopen::deref(reduceTensorDesc).reduceTensorIndicesType_;
    });
};

extern "C" miopenStatus_t
miopenGetReductionIndicesSize(miopenHandle_t handle,
                              const miopenReduceTensorDescriptor_t reduceTensorDesc,
                              const miopenTensorDescriptor_t aDesc,
                              const miopenTensorDescriptor_t cDesc,
                              size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::deref(reduceTensorDesc)
                .GetIndicesSize(miopen::deref(aDesc), miopen::deref(cDesc));
    });
};

extern "C" miopenStatus_t
miopenGetReductionWorkspaceSize(miopenHandle_t handle,
                                const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                const miopenTensorDescriptor_t aDesc,
                                const miopenTensorDescriptor_t cDesc,
                                size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::deref(reduceTensorDesc)
                                         .GetWorkspaceSize(miopen::deref(handle),
                                                           miopen::deref(aDesc),
                                                           miopen::deref(cDesc));
    });
};

extern "C" miopenStatus_t miopenReduceTensor(miopenHandle_t handle,
                                             const miopenReduceTensorDescriptor_t reduceTensorDesc,
                                             void* indices,
                                             size_t indicesSizeInBytes,
                                             void* workspace,
                                             size_t workspaceSizeInBytes,
                                             const void* alpha,
                                             const miopenTensorDescriptor_t aDesc,
                                             const void* A,
                                             const void* beta,
                                             const miopenTensorDescriptor_t cDesc,
                                             void* C)
{
    MIOPEN_LOG_FUNCTION(handle,
                        reduceTensorDesc,
                        indices,
                        indicesSizeInBytes,
                        workspace,
                        workspaceSizeInBytes,
                        alpha,
                        aDesc,
                        A,
                        beta,
                        cDesc,
                        C);

    return miopen::try_([&] {
        miopen::deref(reduceTensorDesc)
            .ReduceTensor(miopen::deref(handle),
                          DataCast(indices),
                          indicesSizeInBytes,
                          DataCast(workspace),
                          workspaceSizeInBytes,
                          alpha,
                          miopen::deref(aDesc),
                          DataCast(A),
                          beta,
                          miopen::deref(cDesc),
                          DataCast(C));
    });
};
