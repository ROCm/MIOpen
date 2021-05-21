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

static void LogCmdRedux(const miopen::ReduceTensorDescriptor reduceTensorDesc,
                        const miopen::TensorDescriptor aDesc,
                        const miopen::TensorDescriptor cDesc,
                        const void* alpha,
                        const void* beta)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(aDesc.GetType() == miopenHalf)
            ss << "reducefp16";
        else if(aDesc.GetType() == miopenBFloat16)
            ss << "reducebfp16";
        else if(aDesc.GetType() == miopenInt8 || aDesc.GetType() == miopenInt8x4)
            ss << "reduceint8";
        else
            ss << "reduce";

        ss << " -A " << *reinterpret_cast<const float*>(alpha);
        ss << " -B " << *reinterpret_cast<const float*>(beta);
        ss << " -C " << reduceTensorDesc.reduceTensorCompType_;
        const auto& inLens = aDesc.GetLengths();
        std::vector<std::string> strInLens;
        std::transform(inLens.begin(),
                       inLens.end(),
                       std::back_inserter(strInLens),
                       [](const int x) { return std::to_string(x); });
        ss << " -D " << miopen::JoinStrings(strInLens, ",");
        ss << " -I " << reduceTensorDesc.reduceTensorIndices_;
        ss << " -N " << reduceTensorDesc.reduceTensorNanOpt_;
        ss << " -O " << reduceTensorDesc.reduceTensorOp_;
        std::vector<int> dimsToReduce;
        int i = 0;
        for(const auto& len : cDesc.GetLengths())
        {
            if(len == 1)
                dimsToReduce.push_back(i);
            ++i;
        }
        std::vector<std::string> strDimsToReduce;
        std::transform(dimsToReduce.begin(),
                       dimsToReduce.end(),
                       std::back_inserter(strDimsToReduce),
                       [](const int x) { return std::to_string(x); });
        ss << " -R " << miopen::JoinStrings(strDimsToReduce, ",");
        // const auto reduceIndicesType = reduceTensorDesc.reduceTensorIndicesType_;
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

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
    LogCmdRedux(
        miopen::deref(reduceTensorDesc), miopen::deref(aDesc), miopen::deref(cDesc), alpha, beta);

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
