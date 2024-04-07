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
#include <miopen/getitem.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdGetitem(const miopenTensorDescriptor_t dyDesc,
                          int32_t indexCount,
                          const miopenTensorDescriptor_t* indexDescs,
                          const miopenTensorDescriptor_t dxDesc,
                          int32_t dimCount,
                          const int32_t* dims,
                          int32_t sliceCount,
                          const int32_t* slices,
                          int32_t offset,
                          bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(dyDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "getitemfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "getitemfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "getitemf16";
        }

        std::string dy_s;
        auto dy_dims = miopen::deref(dyDesc).GetLengths();
        for(auto dy_dim : dy_dims)
        {
            dy_s += std::to_string(dy_dim);
            dy_s += ",";
        }
        dy_s.pop_back();
        ss << " -doutput " << dy_s;

        for(int i = 0; i < indexCount; i++)
        {
            std::string index_s;
            auto index_dims = miopen::deref(indexDescs[i]).GetLengths();
            for(auto index_dim : index_dims)
            {
                index_s += std::to_string(index_dim);
                index_s += ",";
            }
            index_s.pop_back();
            ss << " -index" << i + 1 << " " << index_s;
        }

        std::string dx_s;
        auto dx_dims = miopen::deref(dxDesc).GetLengths();
        for(auto dx_dim : dx_dims)
        {
            dx_s += std::to_string(dx_dim);
            dx_s += ",";
        }
        dx_s.pop_back();
        ss << " -dx " << dx_s;

        std::string dims_s;
        for(int i = 0; i < dimCount; i++)
        {
            dims_s += std::to_string(dims[i]);
            dims_s += ",";
        }
        dims_s.pop_back();
        ss << " -dims" << dims_s;

        std::string slices_s;
        for(int i = 0; i < sliceCount; i++)
        {
            slices_s += std::to_string(slices[i]);
            slices_s += ",";
        }
        slices_s.pop_back();
        ss << " -slice" << slices_s;

        ss << " -offset" << offset;
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenGetGetitemWorkspaceSize(miopenHandle_t handle,
                                                        int32_t indexCount,
                                                        const miopenTensorDescriptor_t* indexDescs,
                                                        size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, indexCount, indexDescs);

    return miopen::try_([&] {
        std::vector<ConstData_t> indexCast;
        std::vector<miopen::TensorDescriptor*> indexDescsCast;
        std::transform(indexDescs,
                       indexDescs + indexCount,
                       std::back_inserter(indexDescsCast),
                       [](const auto& indexDesc) { return &miopen::deref(indexDesc); });
        miopen::deref(sizeInBytes) = miopen::GetGetitemWorkspaceSize(
            miopen::deref(handle), indexCount, indexDescsCast.data());
    });
};

extern "C" miopenStatus_t miopenGetitemBackward(miopenHandle_t handle,
                                                void* workspace,
                                                size_t workspaceSizeInBytes,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                int32_t indexCount,
                                                const miopenTensorDescriptor_t* indexDescs,
                                                const void* const* indexs,
                                                const miopenTensorDescriptor_t yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                const miopenTensorDescriptor_t errorDesc,
                                                void* error,
                                                int32_t dimCount,
                                                const int32_t* dims,
                                                int32_t sliceCount,
                                                const int32_t* slices,
                                                int32_t offset)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        dyDesc,
                        dy,
                        xDesc,
                        x,
                        indexCount,
                        indexDescs,
                        indexs,
                        yDesc,
                        y,
                        dxDesc,
                        dx,
                        errorDesc,
                        error,
                        dimCount,
                        dims,
                        sliceCount,
                        slices,
                        offset);

    LogCmdGetitem(
        dyDesc, indexCount, indexDescs, dxDesc, dimCount, dims, sliceCount, slices, offset, true);
    return miopen::try_([&] {
        std::vector<ConstData_t> indexsCast;
        std::vector<miopen::TensorDescriptor*> indexDescsCast;
        std::transform(indexDescs,
                       indexDescs + indexCount,
                       std::back_inserter(indexDescsCast),
                       [](const auto& indexDesc) { return &miopen::deref(indexDesc); });
        std::transform(indexs,
                       indexs + indexCount,
                       std::back_inserter(indexsCast),
                       [](const void* index) { return DataCast(index); });

        miopen::GetitemBackward(miopen::deref(handle),
                                DataCast(workspace),
                                workspaceSizeInBytes,
                                miopen::deref(dyDesc),
                                DataCast(dy),
                                miopen::deref(xDesc),
                                DataCast(x),
                                indexCount,
                                indexDescsCast.data(),
                                indexsCast.data(),
                                miopen::deref(yDesc),
                                DataCast(y),
                                miopen::deref(dxDesc),
                                DataCast(dx),
                                miopen::deref(errorDesc),
                                DataCast(error),
                                dimCount,
                                dims,
                                sliceCount,
                                slices,
                                offset);
    });
}
