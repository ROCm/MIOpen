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

template <class T>
void cpu_getitem_backward(tensor<T> dy,
                          tensor<T> x,
                          std::vector<tensor<int32_t>> indexs,
                          tensor<T> y,
                          tensor<T>& ref_dx,
                          std::vector<int32_t> dims,
                          std::vector<std::vector<int32_t>> slices,
                          int32_t offset)
{
    auto;

    auto dy_dims   = dy.desc.GetLengths();
    auto dystrides = dy.desc.GetStrides();
    auto dy_numel = std::accumulate(dy_dims.begin(), dy_dims.end(), 1L, std::multiplies<int64_t>());
    auto dx_dims  = ref_dx.desc.GetLengths();
    auto dx_strides = ref_dx.desc.GetStrides();
    auto index_dims = indexs[0].desc.GetLengths();
    auto index_numel =
        std::accumulate(index_dims.begin(), index_dims.end(), 1L, std::multiplies<int64_t>());
    auto indexs_len    = indexs.size();
    auto element_index = std::vector<int32_t>(indexs_len * index_numel);

    std::vector<int32_t> output_dims;
    for(auto dim : dims)
    {
        output_dims.push_back(dxlengths[dim]);
    }

    int32_t dim_info_offset = indexs_len * index_dims[0];
    auto start_dim          = dims[0];

    // Get element index form indexs

    for(int j = 0; j < indexs_len; j++)
    {
        auto dim_size = output_dims[j];
        int32_t error;
        par_ford(index_numel)([&](int32_t o) {
            size_t getitem_index = indexs[o];

            if(getitem_index >= 0 && getitem_index < dim_size)
            {
                element_index[(o * indexs_len) + j] = getitem_index;
            }
            else if(getitem_index >= -dim_size && getitem_index < 0)
            {
                element_index[(o * indexs_len) + j] = getitem_index + dim_size;
            }
            else
            {
                error = -1;
            }

            if(o == 0)
            {
                element_index[dim_info_offset + j] = dim_size;
            }
        });
    }

    // Apply slice to dx
    for(auto slice : slices)
    {
        int32_t dim   = slice[0];
        int32_t start = slice[1];
        int32_t end   = slice[2];
        int32_t step  = slice[3];

        if(end > static_cast<int32_t>(dx_dims[dim]))
            end = dx_dims[dim];

        auto len = end - start;

        dx_dims[dim] = (len + step - 1) / step;
        dx_strides[dim] *= step;
    }

    // GetItem
    par_ford(dy_numel)([&](int32_t o) {
        tensor_view_5d_t tv_5d = get_inner_expanded_tv(dyDesc);
        size_t NCDHW[5], NCDHW2[5];
        size_t ncdh = (o) / tv_5d.size[4];
        NCDHW[4]    = (o) % tv_5d.size[4];
        size_t ncd  = ncdh / tv_5d.size[3];
        NCDHW[3]    = ncdh % tv_5d.size[3];
        size_t nc   = ncd / tv_5d.size[2];
        NCDHW[2]    = ncd % tv_5d.size[2];
        NCDHW[0]    = nc / tv_5d.size[1];
        NCDHW[1]    = nc % tv_5d.size[1];

        for(int i = 0; i < 5; i++)
        {
            NCDHW2[i] = NCDHW[i];
        }

        if(indexs_len > 0)
        {
            size_t dim_cursor = NCDHW[start_dim];
            size_t i          = start_dim;
            size_t j          = 0;

            for(; i < start_dim + indexs_len; ++i, ++j)
            {
                size_t dim_idx  = element_index[dim_info_offset + j];
                NCDHW2[dim_idx] = element_index[(dim_cursor * indexs_len) + j];
            }

            i          = element_index[dim_info_offset + indexs_len - 1] + 1;
            dim_cursor = start_dim + 1;
            for(; i < 5; ++i, ++dim_cursor)
            {
                NCDHW2[i] = NCDHW[dim_cursor];
            }
        }

        auto dy_idx = dy_strides[4] * (NCDHW2[4]) + dy_strides[3] * (NCDHW2[3]) +
                      dy_strides[2] * (NCDHW2[2]) + dy_strides[1] * (NCDHW2[1]) +
                      dy_strides[0] * (NCDHW2[0]);
        auto dx_idx = dx_strides[4] * (NCDHW[4]) + dx_strides[3] * (NCDHW[3]) +
                      dx_strides[2] * (NCDHW[2]) + dx_strides[1] * (NCDHW[1]) +
                      dx_strides[0] * (NCDHW[0]);

        dx[dx_idx] += dy[dy_idx];
    });
}

struct GetitemTestCase
{
    std::vector<int32_t> dy;
    std::vector<int32_t> x;
    std::vector<std::vector<int32_t>> indexs;
    std::vector<int32_t> y;
    std::vector<int32_t> dims;
    std::vector<std::vector<int32_t>> slices;
    int32_t offset;

    friend std::ostream& operator<<(std::ostream& os, const GetitemTestCase& tc)
    {

        os << " dy:" auto dy = tc.dy;
        os << dy[0];
        for(int32_t i = 1; i < dy.size(); i++)
        {
            os << "x" << dy[i];
        }

        os << " x:" auto x = tc.x;
        os << x[0];
        for(int32_t i = 1; i < x.size(); i++)
        {
            os << "x" << x[i];
        }

        os << " indexs:" for(int32_t i = 0; i < tc.indexs.size(); i++)
        {
            auto index = tc.indexs[i];
            if(i != 0)
                os << ",";
            os << index[0];
            for(int32_t j = 1; j < index.size(); j++)
            {
                os << "x" << index[j];
            }
        }

        os << " y:" auto y = tc.y;
        os << y[0];
        for(int32_t i = 1; i < y.size(); i++)
        {
            os << "x" << y[i];
        }

        os << " dx:" auto dx = tc.dx;
        os << dx[0];
        for(int32_t i = 1; i < dx.size(); i++)
        {
            os << "x" << dx[i];
        }

        os << " dims:" auto dims = tc.dims;
        os << dims[0];
        for(int32_t i = 1; i < dims.size(); i++)
        {
            os << "," << dims[i];
        }

        os << " slices:" for(int32_t i = 0; i < tc.slices.size(); i++)
        {
            auto slice = tc.slices[i];
            if(i != 0)
                os << ",";
            os << slice[0];
            for(int32_t j = 1; j < slice.size(); j++)
            {
                os << "x" << slice[j];
            }
        }

        os << " offset:" << offset;

        return os;
    }

    std::vector<size_t> GetDy() { return dy; }

    std::vector<size_t> GetX() { return x; }

    std::vector<std::vector<size_t>> GetIndexs() { return indexs; }

    std::vector<size_t> GetY() { return y; }

    std::vector<size_t> GetDx() { return dx; }

    std::vector<size_t> GetDims() { return dims; }

    std::vector<std::vector<size_t>> GetSlices() { return slices; }
};

std::vector<GetitemTestCase> GetitemTestConfigs()
{ // dy x indexs y dims slices offset
    // clang-format off
    return {
        { {}, {}, {{}}, {{}},  {{0}},  {{}}, 0}
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
        auto x_dim      = getitem_config.GetX();
        auto indexs_dim = getitem_config.GetIndexs();
        auto y_dim      = getitem_config.GetY();
        auto dx_dim     = getitem_config.GetDx();

        dy = tensor<T>{dy_dim}.generate(gen_value);
        x  = tensor<T>{x_dim}.generate(gen_value);
        y  = tensor<T>{y_dim}.generate(gen_value);

        auto output_dims = std::vector<int32_t>{};
        for(auto dim : dims)
        {
            output_dims.push_back(static_cast<int32_t>(dx_dim[dim]));
        }

        for(int32_t i = 0; i < indexs_dim.size(); i++)
        {
            auto gen_value_int = [](auto...) { return prng::gen_0_to_B<int32_t>(output_dims[i]); };
            indexs.push_back(tensor<int32_t>{indexs_dim[i]}.generate(gen_value_int));
        }

        dx = tensor<T>{dx_dim};
        std::fill(dx.begin(), dx.end(), std::numeric_limits<T>::quiet_NaN());

        ref_dx = tensor<T>{dx_dim};
        std::fill(ref_dx.begin(), ref_dx.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_dims;
        ws_sizeInBytes = miopen::GetGetItemWorkspaceSize(
            handle, indexDescs.size(), indexDescs.data(), dims.size(), dims.data());
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        workspace_dims.push_back(ws_sizeInBytes / sizeof(T));
        if(ws_sizeInBytes != 0)
        {
            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), std::numeric_limits<T>::quiet_NaN());
            workspace_dev = handle.Write(workspace.data);
        }

        dy_dev = handle.Write(dy.data);
        x_dev  = handle.Write(x.data);
        y_dev  = handle.Write(y.data);

        std::transform(indexs.begin(),
                       indexs.end(),
                       std::back_inserter(indexs_dev),
                       [&](auto& index) { return handle.Write(index.data); });

        dx_dev = handle.Write(dx.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_getitem_backward<T>(dy, x, indexs, y, ref_dx, dims, slices, offset);

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
                                                        x.desc,
                                                        x_dev.get(),
                                                        indexDescs.size() indexDescs.data(),
                                                        indexData.get(),
                                                        y.desc,
                                                        y_dev.get(),
                                                        dx.desc,
                                                        dx_dev.get(),
                                                        dims.size(),
                                                        dims.data(),
                                                        slices.size(),
                                                        slices_flat.data(),
                                                        offset);

        EXPECT_EQ(status, miopenStatusSuccess);

        dx.data = handle.Read<T>(dx_dev, dx.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        // In the case of layernorm, there is a cumulative sum operation, and in the case of
        // floating point operation, the result value can change if the order of the summed values
        // is changed. So apply a threshold that is 10 times larger than other operations.
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;

        auto error = miopen::rms_range(ref_dx, dx);
        EXPECT_TRUE(miopen::range_distance(ref_dx) == miopen::range_distance(dx));
        EXPECT_TRUE(error < threshold)
            << "Error dx beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    GetitemTestCase getitem_config;

    tensor<T> dy;
    tensor<T> x;
    std::vector<tensor<int32_t>> indexs;
    tensor<T> y;
    tensor<T> dx;
    tensor<T> workspace;

    tensor<T> ref_dx;

    miopen::Allocator::ManageDataPtr dy_dev;
    miopen::Allocator::ManageDataPtr x_dev;
    std::vector<miopen::Allocator::ManageDataPtr> indexs_dev;
    miopen::Allocator::ManageDataPtr y_dev;
    miopen::Allocator::ManageDataPtr dx_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;

    std::vector<int32_t> dims;
    std::vector<std::vector<int32_t>> slices;
    std::vector<int32_t> slices_flat;
    int32_t offset;
};