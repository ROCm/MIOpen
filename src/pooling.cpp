/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
#include <miopen/pooling.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/logger.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/pooling/invoke_params.hpp>
#include <miopen/pooling/solvers.hpp>
#include <miopen/subbuffers.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>

#include <cassert>
#include <cmath>

namespace miopen {

template <class T, class U>
T iciel_div(T x, U y)
{
    auto rem = x % y;
    if(rem > 0)
        rem = 1;
    return ((x - rem) * y);
}

static auto PoolingForwardSolvers()
{
    return solver::SolverContainer<solver::pooling::PoolingForward2d,
                                   solver::pooling::PoolingForwardNd,
                                   solver::pooling::PoolingForwardNaive,
                                   solver::pooling::TransposedPoolingFwd2d,
                                   solver::pooling::TransposedPoolingFwdNd>{};
}

static auto PoolingBackwardSolvers()
{
    return solver::SolverContainer<solver::pooling::PoolingBackward2d,
                                   solver::pooling::PoolingBackwardNd,
                                   solver::pooling::TransposedPoolingBwd2d,
                                   solver::pooling::TransposedPoolingBwdNd>{};
}

PoolingDescriptor::PoolingDescriptor() {}

PoolingDescriptor::PoolingDescriptor(miopenPoolingMode_t m,
                                     miopenPaddingMode_t pm,
                                     const int* plens,
                                     const int* ppads,
                                     const int* pstrides,
                                     int size)
    : lens(plens, plens + size),
      strides(pstrides, pstrides + size),
      pads(ppads, ppads + size),
      mode(m),
      pmode(pm),
      indexType(miopenIndexUint8),
      workspaceIndexMode(size == 3 ? miopenPoolingWorkspaceIndexImage
                                   : miopenPoolingWorkspaceIndexMask)
{
}

PoolingDescriptor::PoolingDescriptor(miopenPoolingMode_t m,
                                     miopenPaddingMode_t pm,
                                     const std::vector<int>& plens,
                                     const std::vector<int>& pstrides,
                                     const std::vector<int>& ppads)
    : lens(plens),
      strides(pstrides),
      pads(ppads),
      mode(m),
      pmode(pm),
      indexType(miopenIndexUint8),
      workspaceIndexMode(miopenPoolingWorkspaceIndexMask)
{
    if(plens.size() == 3)
        workspaceIndexMode = miopenPoolingWorkspaceIndexImage;
}

void PoolingDescriptor::SetIndexType(miopenIndexType_t index_type) { indexType = index_type; }

miopenIndexType_t PoolingDescriptor::GetIndexType() const { return indexType; }

void PoolingDescriptor::SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t workspace_index)
{
    workspaceIndexMode = workspace_index;
}

miopenPoolingWorkspaceIndexMode_t PoolingDescriptor::GetWorkspaceIndexMode() const
{
    return workspaceIndexMode;
}

miopenPoolingMode_t PoolingDescriptor::GetMode() const { return mode; }

miopenPaddingMode_t PoolingDescriptor::GetPaddingMode() const { return (pmode); }

const std::vector<int>& PoolingDescriptor::GetLengths() const { return lens; }

const std::vector<int>& PoolingDescriptor::GetStrides() const { return strides; }

const std::vector<int>& PoolingDescriptor::GetPads() const { return pads; }

int PoolingDescriptor::GetSize() const
{
    assert(lens.size() == strides.size() && lens.size() == pads.size());
    return lens.size();
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
PoolingDescriptor::GetForwardOutputDim(const TensorDescriptor& xDesc) const
{

    assert(xDesc.GetLengths().size() == 4);

    std::size_t input_n;
    std::size_t input_c;
    std::size_t input_h;
    std::size_t input_w;

    std::tie(input_n, input_c, input_h, input_w) = miopen::tien<4>(xDesc.GetLengths());

    int stride_h, stride_w, pad_h, pad_w, window_h, window_w;
    std::tie(stride_h, stride_w) = miopen::tien<2>(GetStrides());
    std::tie(pad_h, pad_w)       = miopen::tien<2>(GetPads());
    std::tie(window_h, window_w) = miopen::tien<2>(GetLengths());
    miopenPaddingMode_t _pMode   = GetPaddingMode();

    assert(stride_h > 0);
    assert(stride_w > 0);
    assert(window_h < (input_h + static_cast<std::size_t>(2) * pad_h));
    assert(window_w < (input_w + static_cast<std::size_t>(2) * pad_w));

    auto output_h = std::max<std::ptrdiff_t>(
        1, ((input_h + 2 * static_cast<std::ptrdiff_t>(pad_h) - window_h) / stride_h + 1));
    auto output_w = std::max<std::ptrdiff_t>(
        1, ((input_w + 2 * static_cast<std::ptrdiff_t>(pad_w) - window_w) / stride_w + 1));

    if(_pMode == miopenPaddingSame)
    {
        output_h = std::ceil(static_cast<double>(input_h) / static_cast<double>(stride_h));
        output_w = std::ceil(static_cast<double>(input_w) / static_cast<double>(stride_w));
    }
    else if(_pMode == miopenPaddingValid)
    {
        output_h =
            std::ceil(static_cast<double>(input_h - window_h + 1) / static_cast<double>(stride_h));
        output_w =
            std::ceil(static_cast<double>(input_w - window_w + 1) / static_cast<double>(stride_w));
    }

    return std::make_tuple(input_n, input_c, output_h, output_w);
}

void PoolingDescriptor::GetForwardOutputDimNd(const TensorDescriptor& xDesc,
                                              int dims,
                                              int* tensorDimArr) const
{
    assert(xDesc.GetLengths().size() == dims && xDesc.GetLengths().size() <= 5 &&
           xDesc.GetLengths().size() >= 4); // currently only support 2D/3D pooling
    std::vector<int> out_dim;
    auto input_dim             = xDesc.GetLengths();
    auto strs                  = GetStrides();
    auto padd                  = GetPads();
    auto kernels               = GetLengths();
    miopenPaddingMode_t _pMode = GetPaddingMode();

    assert(strs.size() + 2 == input_dim.size() && strs.size() == padd.size() &&
           strs.size() == kernels.size());
    assert(std::all_of(kernels.begin(), kernels.end(), [](int s) { return s > 0; }));
    assert(std::all_of(strs.begin(), strs.end(), [](int s) { return s > 0; }));
    assert(std::all_of(padd.begin(), padd.end(), [](int s) { return s >= 0; }));

    auto in_itr = input_dim.begin();
    out_dim.push_back(int(*(in_itr++))); // n
    out_dim.push_back(int(*(in_itr++))); // c

    auto str_itr = strs.begin();
    auto pad_itr = padd.begin();
    auto ker_itr = kernels.begin();

    while(in_itr != input_dim.end())
    {
        int out_tmp = std::max<std::ptrdiff_t>(
            1, ((*in_itr + 2 * static_cast<std::ptrdiff_t>(*pad_itr) - *ker_itr) / (*str_itr) + 1));

        if(_pMode == miopenPaddingSame)
        {
            out_tmp =
                std::max<std::ptrdiff_t>(1,
                                         std::ptrdiff_t(std::ceil(static_cast<double>(*in_itr) /
                                                                  static_cast<double>(*str_itr))));
        }
        else if(_pMode == miopenPaddingValid)
        {
            out_tmp = std::max<std::ptrdiff_t>(
                1,
                std::ptrdiff_t(std::ceil(static_cast<double>(*in_itr - *ker_itr + 1) /
                                         static_cast<double>(*str_itr))));
        }

        in_itr++;
        str_itr++;
        pad_itr++;
        ker_itr++;

        out_dim.push_back(out_tmp);
    }

    std::copy(out_dim.begin(), out_dim.begin() + dims, tensorDimArr);
}

TensorDescriptor PoolingDescriptor::GetForwardOutputTensor(const TensorDescriptor& xDesc) const
{
    std::vector<int> out_dim(xDesc.GetNumDims());
    GetForwardOutputDimNd(xDesc, xDesc.GetNumDims(), out_dim.data());

    const std::string default_layout = tensor_layout_get_default(xDesc.GetNumDims());
    const std::string in_layout      = xDesc.GetLayout(default_layout);
    std::vector<int> out_strides;
    tensor_layout_to_strides(out_dim, default_layout, in_layout, out_strides);

    return {xDesc.GetType(), out_dim, out_strides};
}

std::size_t PoolingDescriptor::GetWorkSpaceSize(const TensorDescriptor& yDesc) const
{
    const auto y_size       = yDesc.GetElementSize();
    const auto index_e_size = get_data_size(GetIndexType());

    const auto main_ws = GetMode() == miopenPoolingMax ? y_size * index_e_size : 0;

    const auto labels        = tensor_layout_get_default(yDesc.GetNumDims());
    std::size_t transpose_ws = 0;

    if(yDesc.GetLayout(labels) != labels)
    {
        const auto e_size       = get_data_size(yDesc.GetType());
        auto transposed_strides = std::vector<std::size_t>{};
        const auto in_layout    = yDesc.GetLayout(labels);
        tensor_layout_to_strides(yDesc.GetLengths(), labels, in_layout, transposed_strides);
        const auto transposed_y =
            TensorDescriptor{yDesc.GetType(), yDesc.GetLengths(), transposed_strides};

        const auto y_transpose_size = transposed_y.GetElementSpace() * e_size;
        // Todo: We do not have xDesc, so we infer that. But currently this is incorrect, x tensor
        // is larger than y and should be calculated properly.
        const auto x_transpose_size = y_transpose_size * 2;
        const auto subbuffer_align  = GetSubbufferAlignment();
        const auto align_after_mask = (subbuffer_align - main_ws) % subbuffer_align;
        const auto align_after_x    = (subbuffer_align - x_transpose_size) % subbuffer_align;

        transpose_ws = align_after_mask + x_transpose_size + align_after_x + y_transpose_size;
    }

    return main_ws + transpose_ws;
}

miopenStatus_t PoolingDescriptor::Forward(const Handle& handle,
                                          const void* alpha,
                                          const TensorDescriptor& xDesc,
                                          ConstData_t x,
                                          const void* beta,
                                          const TensorDescriptor& yDesc,
                                          Data_t y,
                                          bool save_index,
                                          Data_t workSpace,
                                          size_t workSpaceSize) const
{
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        if(!float_equal(*(static_cast<const float*>(beta)), 0))
        {
            miopen::checkNumericsInput(handle, yDesc, y);
        }
    }

    unsigned pool_dim = xDesc.GetNumDims();
    if(pool_dim != 4 && pool_dim != 5)
    {
        MIOPEN_THROW("Unsupported pooling dimension");
    }

    auto index_max = get_index_max(GetIndexType());

    /// \anchor max_pooling_index_max_restriction
    /// For kernel implementation max pooling backward pass,
    /// "index_max" means ghost, and thus should not be reached.
    if(mode == miopenPoolingMax && save_index)
    {
        if((workspaceIndexMode == miopenPoolingWorkspaceIndexMask                                 //
            && index_max <= std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<int>())) //
           ||                                                                                     //
           (workspaceIndexMode == miopenPoolingWorkspaceIndexImage                                //
            && index_max <= std::accumulate(xDesc.GetLengths().begin() + 2,
                                            xDesc.GetLengths().end(),
                                            1,
                                            std::multiplies<int>())))
        {
            MIOPEN_THROW("Index range not enough for max pooling bwd");
        }

        if(workSpace == nullptr)
        {
            throw std::invalid_argument("workSpace cannot be NULL in Forward Pooling MAX mode when "
                                        "backward pass is requested");
        }
    }

    // So far, all pooling solvers implement the Direct (trivial) computation algorithm.
    const auto algo_name = AlgorithmName{"miopenPoolingForwardDirect"};
    const auto problem   = pooling::ProblemDescription{*this, xDesc, yDesc, save_index};

    const auto invoke_params = [&]() {
        auto tmp           = pooling::FwdInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.xDesc          = xDesc;
        tmp.yDesc          = yDesc;
        tmp.pooling        = *this;
        tmp.x              = x;
        tmp.y              = y;
        tmp.workspace      = workSpace;
        tmp.workspace_size = workSpaceSize;
        return tmp;
    }();

    PoolingForwardSolvers().ExecutePrimitive(handle, problem, algo_name, invoke_params);

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }

    return miopenStatusSuccess;
}

miopenStatus_t PoolingDescriptor::Backward(const Handle& handle,
                                           const void* alpha,
                                           const TensorDescriptor& yDesc,
                                           ConstData_t /*y*/,
                                           const TensorDescriptor& dyDesc,
                                           ConstData_t dy,
                                           const TensorDescriptor& xDesc,
                                           ConstData_t /*x*/,
                                           const void* beta,
                                           const TensorDescriptor& dxDesc,
                                           Data_t dx,
                                           Data_t workSpace) const
{
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, dyDesc, dy);
        if(!float_equal(*(static_cast<const float*>(beta)), 0))
        {
            miopen::checkNumericsInput(handle, dxDesc, dx);
        }
    }

    assert(yDesc.GetElementSize() == dyDesc.GetElementSize() &&
           xDesc.GetElementSize() == dxDesc.GetElementSize());

    unsigned pool_dim = dyDesc.GetNumDims();
    if(pool_dim != 4 && pool_dim != 5)
    {
        MIOPEN_THROW("Unsupported pooling dimension");
    }

    const auto problem   = pooling::ProblemDescription{*this, xDesc, yDesc, dxDesc, dyDesc};
    const auto algo_name = AlgorithmName{"miopenPoolingBackwardDirect"};

    const auto invoke_params = [&]() {
        auto tmp      = pooling::BwdInvokeParams{};
        tmp.type      = InvokeType::Run;
        tmp.dxDesc    = dxDesc;
        tmp.dyDesc    = dyDesc;
        tmp.pooling   = *this;
        tmp.dx        = dx;
        tmp.dy        = dy;
        tmp.workspace = workSpace;
        return tmp;
    }();

    PoolingBackwardSolvers().ExecutePrimitive(handle, problem, algo_name, invoke_params);

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, dxDesc, dx);
    }

    return miopenStatusSuccess;
}

std::ostream& operator<<(std::ostream& stream, const PoolingDescriptor& x)
{
    MIOPEN_LOG_ENUM(stream, x.mode, miopenPoolingMax, miopenPoolingAverage) << ", ";
    LogRange(stream, x.lens, ", ") << ", ";
    LogRange(stream, x.pads, ", ") << ", ";
    LogRange(stream, x.strides, ", ") << ", ";
    return stream;
}

} // namespace miopen
