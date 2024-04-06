/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
                          int32_t* dims,
                          int32_t,
                          sliceCount,
                          inte32_t* slices,
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

        std::string dy_sz;
        auto dims = miopen::deref(dyDesc).GetLengths();
        for(auto dim : dims)
        {
            dy_sz += std::to_string(dim);
            dy_sz += ",";
        }
        dy_sz.pop_back();
        ss << " -doutput " << dy_sz;

        for(int i = 0; i < indexDescs.size(); i++)
        {
            std::string index_s;
            auto dims = miopen::deref(indexDescs[i]).GetLengths();
            for(auto dim : dims)
            {
                index_s += std::to_string(dim);
                index_s += ",";
            }
            index_s.pop_back();
            ss << " -index" << i + 1 < < < < index_s;
        }

        std::string dx_sz;
        auto dims = miopen::deref(dxDesc).GetLengths();
        for(auto dim : dims)
        {
            dx_sz += std::to_string(dim);
            dx_sz += ",";
        }
        dx_sz.pop_back();
        ss << " -dx " << dx_sz;

        ss << " -dims " std::string dims_s;
        for(int i = 0; i < dimCount; i++)
        {
            dims_s += std::to_string(dims[i]);
            dims_s += ",";
        }
        dim_s.pop_back();
        ss << " -dim" << dims_s;

        ss << " -slices " std::string slices_s;
        for(int i = 0; i < sliceCount; i++)
        {
            slices_s += std::to_string(slices[i]);
            slices_s += ",";
        }
        slice_s.pop_back();
        ss << " -slice" << slices_s;

        ss << " -offset" << offset;
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenGetGetitemWorkspaceSize(miopenHandle_t handle,
                                                        const int32_t indexCount,
                                                        const miopenTensorDescriptor_t* indexDescs,
                                                        const void* const* indexs,
                                                        const int32_t dimCount,
                                                        const int32_t* dims,
                                                        size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, indexCount, indexDescs, indexs, dimCount, dims);

    return miopen::try_([&] {
        std::vector<ConstData_t> indexCast;
        std::vector<miopen::TensorDescriptor*> indexDescsCast;
        std::transform(indexDescs,
                       indexDescs + indexCount,
                       std::back_inserter(indexDescsCast),
                       [](const auto& indexDesc) { return &miopen::deref(indexDesc); });
        std::transform(indexs,
                       indexs + indexCount,
                       std::back_inserter(indexCast),
                       [](const void* index) { return DataCast(index); });
        miopen::deref(sizeInBytes) = miopen::GetSumWorkspaceSize(miopen::deref(handle),
                                                                 indexCount,
                                                                 indexDescsCast.data(),
                                                                 indexCast.data(),
                                                                 dimCount,
                                                                 miopen::deref(dims));
    });
};

extern "C" miopenStatus_t miopenGetitemBackward(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t xDesc,
                                                const void* x,
                                                const int32_t indexCount,
                                                const miopenTensorDescriptor_t* indexDescs,
                                                const void* const* indexs,
                                                const miopenTensorDescriptor_t yDesc,
                                                const void* y,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                const int32_t dimCount,
                                                const int32_t* dims,
                                                const int32_t sliceCount,
                                                const int32_t* slices,
                                                const int32_t offset)
{
    MIOPEN_LOG_FUNCTION(handle,
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
                        dimCount,
                        dims,
                        sliceCount,
                        slices,
                        offset);
    LogCmdGetitem(xDescs, xCount, true);
    return miopen::try_([&] {
        std::vector<ConstData_t> indexCast;
        std::vector<miopen::TensorDescriptor*> indexDescsCast;
        std::transform(indexDescs,
                       indexDescs + indexCount,
                       std::back_inserter(indexDescsCast),
                       [](const auto& indexDesc) { return &miopen::deref(indexDesc); });
        std::transform(indexs,
                       indexs + indexCount,
                       std::back_inserter(indexCast),
                       [](const void* index) { return DataCast(index); });

        miopen::GetitemBackward(miopen::deref(handle),
                                miopen::deref(dyDesc),
                                DataCast(dy),
                                miopen::deref(xDesc),
                                DataCast(x),
                                indexCount,
                                indexDescsCast.data(),
                                indexCast.data(),
                                miopen::deref(yDesc),
                                DataCast(y),
                                miopen::deref(dxDesc),
                                DataCast(dx),
                                dimCount,
                                miopen::deref(dims),
                                sliceCount,
                                miopen::deref(slices),
                                offset);
    });
}
