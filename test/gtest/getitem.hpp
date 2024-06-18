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

#include "../driver/tensor_driver.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/getitem.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_getitem_backward(tensor<T> dy,
                          uint32_t indexCount,
                          std::vector<tensor<int32_t>> indexs,
                          tensor<T>& ref_dx,
                          tensor<int32_t>& ref_error,
                          uint32_t dimCount,
                          int32_t* dims,
                          uint32_t sliceCount,
                          int32_t* slices,
                          uint32_t offset)
{
    auto dy_dims  = dy.desc.GetLengths();
    auto dy_numel = std::accumulate(dy_dims.begin(), dy_dims.end(), 1L, std::multiplies<int64_t>());
    auto dx_dims  = ref_dx.desc.GetLengths();
    auto index_dims = indexs[0].desc.GetLengths();
    auto index_numel =
        std::accumulate(index_dims.begin(), index_dims.end(), 1L, std::multiplies<int64_t>());
    auto element_index = std::vector<int32_t>(indexCount * index_numel + indexCount);

    std::vector<int32_t> output_dims;
    for(int32_t i = 0; i < dimCount; i++)
    {
        output_dims.push_back(dx_dims[dims[i]]);
    }

    auto dim_info_offset = indexCount > 0 ? indexCount * index_dims[0] : 0;
    auto start_dim       = dims[0];

    auto dy_tv     = miopen::get_inner_expanded_tv<5>(dy.desc);
    auto ref_dx_tv = miopen::get_inner_expanded_tv<5>(ref_dx.desc);
    miopen::slice_tv<5>(ref_dx_tv, sliceCount, slices);

    // Get element index form indexs
    for(int j = 0; j < indexCount; j++)
    {
        auto index_dim = dims[j];
        auto dim_size  = output_dims[j];

        par_ford(index_numel)([&](int32_t o) {
            int32_t getitem_index = indexs[j][o];

            if(getitem_index >= 0 && getitem_index < dim_size)
            {
                element_index[(o * indexCount) + j] = getitem_index;
            }
            else if(getitem_index >= -dim_size && getitem_index < 0)
            {
                element_index[(o * indexCount) + j] = getitem_index + dim_size;
            }
            else
            {
                ref_error[j] = -1;
            }

            if(o == 0)
            {
                element_index[dim_info_offset + j] = index_dim;
            }
        });
    }

    // GetItem
    par_ford(dy_numel)([&](int32_t o) {
        tensor_layout_t<5> ncdhw(dy_tv, o);
        tensor_layout_t<5> idx(ncdhw);

        if(indexCount > 0)
        {
            size_t dim_cursor = ncdhw.layout[start_dim];
            size_t i          = start_dim;
            size_t j          = 0;

            for(; i < start_dim + indexCount; ++i, ++j)
            {
                size_t dim_idx      = element_index[dim_info_offset + j];
                idx.layout[dim_idx] = element_index[(dim_cursor * indexCount) + j];
            }

            i          = element_index[dim_info_offset + indexCount - 1] + 1;
            dim_cursor = start_dim + 1;
            for(; i < 5; ++i, ++dim_cursor)
            {
                idx.layout[i] = ncdhw.layout[dim_cursor];
            }
        }

        ref_dx[ref_dx_tv.get_tensor_view_idx(idx)] += dy[dy_tv.get_tensor_view_idx(ncdhw)];
    });
}

struct GetitemTestCase
{
    std::vector<int32_t> dy;
    std::vector<std::vector<int32_t>> indexs;
    std::vector<int32_t> dx;
    std::vector<int32_t> dims;
    std::vector<std::vector<int32_t>> slices;
    uint32_t offset;

    friend std::ostream& operator<<(std::ostream& os, const GetitemTestCase& tc)
    {

        os << " dy:";
        auto dy_s = tc.dy;
        os << dy_s[0];
        for(int32_t i = 1; i < dy_s.size(); i++)
        {
            os << "x" << dy_s[i];
        }

        os << " indexs:";
        for(int32_t i = 0; i < tc.indexs.size(); i++)
        {
            auto index_s = tc.indexs[i];
            if(i != 0)
                os << ",";
            os << index_s[0];
            for(int32_t j = 1; j < index_s.size(); j++)
            {
                os << "index" << index_s[j];
            }
        }

        os << " dx:";
        auto dx_s = tc.dx;
        os << dx_s[0];
        for(int32_t i = 1; i < dx_s.size(); i++)
        {
            os << "x" << dx_s[i];
        }

        os << " dims:";
        auto dims_s = tc.dims;
        os << dims_s[0];
        for(int32_t i = 1; i < dims_s.size(); i++)
        {
            os << "," << dims_s[i];
        }

        os << " slices:";
        for(int32_t i = 0; i < tc.slices.size(); i++)
        {
            auto slice_s = tc.slices[i];
            if(i != 0)
                os << ",";
            os << slice_s[0];
            for(int32_t j = 1; j < slice_s.size(); j++)
            {
                os << "slice" << slice_s[j];
            }
        }

        os << " offset:" << tc.offset;

        return os;
    }

    std::vector<int32_t> GetDy() { return dy; }

    std::vector<std::vector<int32_t>> GetIndexs() { return indexs; }

    std::vector<int32_t> GetDx() { return dx; }

    std::vector<int32_t> GetDims() { return dims; }

    std::vector<std::vector<int32_t>> GetSlices() { return slices; }
};

std::vector<GetitemTestCase> GetitemTestConfigs()
{ // dy indexs dx dims slices offset
    // clang-format off
    return {
        { {128, 128}, {{128}},  {128, 128},   {0}, {}, 0}, //llama2
        { {16, 4},    {{16}},   {3234, 4},    {0}, {}, 0}, //ssdlite
        { {149, 128}, {{1490}}, {1490, 1128}, {0}, {}, 0}, //llama2_7b
        { {10, 128},  {{10}},   {160, 128},   {0}, {}, 0},
        { {4260, 4},  {{4300}}, {4300, 4},    {0}, {}, 0}, //fasterrcnn
        { {4260},     {{4300}}, {4300},       {0}, {}, 0}  //maskrcnn
      };
    // clang-format on
}

template <typename T = float>
struct GetitemBwdTest : public ::testing::TestWithParam<GetitemTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        getitem_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dims   = getitem_config.GetDims();
        slices = getitem_config.GetSlices();
        offset = getitem_config.offset;

        for(auto slice : slices)
        {
            for(int32_t i = 0; i < 4; i++)
            {
                slices_flat.push_back(slice[i]);
            }
        }

        auto dy_dim     = getitem_config.GetDy();
        auto indexs_dim = getitem_config.GetIndexs();
        auto dx_dim     = getitem_config.GetDx();
        std::vector<int32_t> error_dim;
        error_dim.push_back(indexs_dim.size());

        dy = tensor<T>{dy_dim}.generate(gen_value);

        auto output_dims = std::vector<int32_t>{};
        for(auto dim : dims)
        {
            output_dims.push_back(static_cast<int32_t>(dx_dim[dim]));
        }

        for(int32_t i = 0; i < indexs_dim.size(); i++)
        {
            auto index       = tensor<int32_t>{indexs_dim[i]};
            auto index_dims  = index.desc.GetLengths();
            auto index_numel = std::accumulate(
                index_dims.begin(), index_dims.end(), 1L, std::multiplies<int64_t>());
            for(int32_t j = 0; j < index_numel; j++)
            {
                index[j] = prng::gen_0_to_B<int32_t>(output_dims[i]);
            }
            indexs.push_back(index);
        }

        dx = tensor<T>{dx_dim};
        std::fill(dx.begin(), dx.end(), static_cast<T>(0));

        error = tensor<int32_t>{error_dim};
        std::fill(error.begin(), error.end(), static_cast<int32_t>(0));

        ref_error = tensor<int32_t>{error_dim};
        std::fill(ref_error.begin(), ref_error.end(), static_cast<int32_t>(0));

        ref_dx = tensor<T>{dx_dim};
        std::fill(ref_dx.begin(), ref_dx.end(), static_cast<T>(0));

        std::vector<miopen::TensorDescriptor*> indexDescs;

        std::transform(indexs.begin(),
                       indexs.end(),
                       std::back_inserter(indexDescs),
                       [](auto& index) { return &index.desc; });

        std::vector<size_t> workspace_dims;
        ws_sizeInBytes =
            miopen::GetGetitemWorkspaceSize(handle, indexDescs.size(), indexDescs.data());

        workspace_dims.push_back(ws_sizeInBytes / sizeof(T));
        if(ws_sizeInBytes != 0)
        {
            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), static_cast<T>(0));
            workspace_dev = handle.Write(workspace.data);
        }

        dy_dev = handle.Write(dy.data);

        std::transform(indexs.begin(),
                       indexs.end(),
                       std::back_inserter(indexs_dev),
                       [&](auto& index) { return handle.Write(index.data); });

        dx_dev    = handle.Write(dx.data);
        error_dev = handle.Write(error.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_getitem_backward<T>(dy,
                                indexs.size(),
                                indexs,
                                ref_dx,
                                ref_error,
                                dims.size(),
                                dims.data(),
                                slices.size(),
                                slices_flat.data(),
                                offset);

        std::vector<miopen::TensorDescriptor*> indexDescs;
        std::vector<ConstData_t> indexData;

        std::transform(indexs.begin(),
                       indexs.end(),
                       std::back_inserter(indexDescs),
                       [](auto& index) { return &index.desc; });
        std::transform(indexs_dev.begin(),
                       indexs_dev.end(),
                       std::back_inserter(indexData),
                       [](auto& index_dev) { return index_dev.get(); });

        miopenStatus_t status = miopen::GetitemBackward(handle,
                                                        workspace_dev.get(),
                                                        ws_sizeInBytes,
                                                        dy.desc,
                                                        dy_dev.get(),
                                                        indexDescs.size(),
                                                        indexDescs.data(),
                                                        indexData.data(),
                                                        dx.desc,
                                                        dx_dev.get(),
                                                        error.desc,
                                                        error_dev.get(),
                                                        dims.size(),
                                                        dims.data(),
                                                        slices.size(),
                                                        slices_flat.data(),
                                                        offset);

        EXPECT_EQ(status, miopenStatusSuccess);

        dx.data    = handle.Read<T>(dx_dev, dx.data.size());
        error.data = handle.Read<int32_t>(error_dev, error.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        // In the case of layernorm, there is a cumulative sum operation, and in the case of
        // floating point operation, the result value can change if the order of the summed values
        // is changed. So apply a threshold that is 10 times larger than other operations.
        auto threshold = std::is_same<T, float>::value ? 1.5e-4 : 8.2e-1;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        // If there is an atomic operation on the GPU kernel, a large error occurs depending on the
        // calculation order, so it is multiplied by 10 times.
        if(std::is_same<T, bfloat16>::value)
            threshold *= 8000.0;

        auto error_dx = miopen::rms_range(ref_dx, dx);
        EXPECT_TRUE(miopen::range_distance(ref_dx) == miopen::range_distance(dx));
        EXPECT_TRUE(error_dx < threshold * 10) << "Error dx beyond tolerance Error:" << error_dx
                                               << ",  Thresholdx10: " << threshold * 10;

        auto error_error = miopen::rms_range(ref_error, error);
        EXPECT_TRUE(miopen::range_distance(ref_error) == miopen::range_distance(error));
        EXPECT_TRUE(std::abs(static_cast<float>(error_error)) == 0.0f) << "Error dx is not equal";
    }
    GetitemTestCase getitem_config;

    tensor<T> dy;
    std::vector<tensor<int32_t>> indexs;
    tensor<T> dx;
    tensor<T> workspace;
    tensor<int32_t> error;

    tensor<T> ref_dx;
    tensor<int32_t> ref_error;

    miopen::Allocator::ManageDataPtr dy_dev;
    std::vector<miopen::Allocator::ManageDataPtr> indexs_dev;
    miopen::Allocator::ManageDataPtr dx_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr error_dev;

    size_t ws_sizeInBytes;

    std::vector<int32_t> dims;
    std::vector<std::vector<int32_t>> slices;
    std::vector<int32_t> slices_flat;
    uint32_t offset;
};
