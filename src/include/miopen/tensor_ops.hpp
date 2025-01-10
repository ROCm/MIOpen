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
#ifndef GUARD_MIOPEN_TENSOR_OPPS_HPP_
#define GUARD_MIOPEN_TENSOR_OPPS_HPP_

#include <miopen/common.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/tensor.hpp>
#include <miopen/functional.hpp>
#include <vector>
#include <boost/range/combine.hpp>
#include <boost/range/adaptor/filtered.hpp>

namespace miopen {

struct Handle;

struct f_length_is_not_1_t
{
    template <typename T>
    bool operator()(T&& v)
    {
        return boost::get<0>(v) > 1;
    }
};

MIOPEN_INTERNALS_EXPORT TensorDescriptor GetFlattenedTensorDescriptor(const TensorDescriptor& desc);

template <typename... TDescriptors>
std::tuple<TDescriptors...>
GetConsistentFlattenedTensorDescriptors(const TDescriptors&... real_descriptor_pack)
{
    constexpr std::size_t NTensor = sizeof...(TDescriptors);
    std::integral_constant<std::size_t, NTensor> NTensorConstant;

    std::array<const TensorDescriptor*, NTensor> real_descriptors{{(&real_descriptor_pack)...}};

#ifndef NDEBUG
    // sanity check: all input TensorDescriptors should have the same GetLengths()
    const auto& real_desc_0_lens = real_descriptors[0]->GetLengths();

    for(std::size_t itensor = 1; itensor < NTensor; ++itensor)
    {
        if(real_desc_0_lens != real_descriptors[itensor]->GetLengths())
            MIOPEN_THROW(miopenStatusBadParm, "Lengths of Tensors are different.");
    }
#endif

    // if tensors are all packed
    bool is_all_packed = true;
    for(std::size_t itensor = 0; itensor < NTensor; ++itensor)
        is_all_packed &= real_descriptors[itensor]->IsPacked();

    bool is_all_same_strided        = true;
    const auto& real_desc_0_strides = real_descriptors[0]->GetStrides();
    for(std::size_t itensor = 1; itensor < NTensor; ++itensor)
    {
        if(real_desc_0_strides != real_descriptors[itensor]->GetStrides())
        {
            is_all_same_strided = false;
            break;
        }
    }

    auto non1_length_strides =
        boost::combine(real_descriptors[0]->GetLengths(), real_descriptor_pack.GetStrides()...) |
        boost::adaptors::filtered(f_length_is_not_1_t());

    if((is_all_packed && is_all_same_strided) || non1_length_strides.empty())
    {
        auto sz = real_descriptors[0]->GetElementSize();
        return create_tuple<NTensor>([&](auto itensor) {
            return TensorDescriptor{
                real_descriptors[itensor]->GetType(), {sz}, {static_cast<std::size_t>(1)}};
        });
    }

    // start flattening tensors
    std::array<std::vector<std::size_t>, NTensor> array_of_flat_lengths;
    std::array<std::vector<std::size_t>, NTensor> array_of_flat_strides;

    auto i               = non1_length_strides.begin();
    std::size_t flat_len = boost::get<0>(*i);
    auto i_previous      = i++;

    // the 0-th dimension full-length doesn't matter
    for(; i != non1_length_strides.end(); ++i)
    {
        std::size_t len = boost::get<0>(*i);

        bool is_all_full_length = true;
        repeat_n(
            [&](auto itensor) {
                const std::size_t stride          = boost::get<itensor + 1>(*i);
                const std::size_t previous_stride = boost::get<itensor + 1>(*i_previous);
                const std::size_t full_len        = previous_stride / stride;
                is_all_full_length &= (len == full_len);
            },
            NTensorConstant);

        if(is_all_full_length)
        {
            flat_len *= len;
        }
        else
        {
            array_of_flat_lengths[0].push_back(flat_len);

            repeat_n(
                [&](auto itensor) {
                    std::size_t previous_stride = boost::get<itensor + 1>(*i_previous);
                    array_of_flat_strides[itensor].push_back(previous_stride);
                },
                NTensorConstant);
            flat_len = len;
        }
        i_previous = i;
    }
    // lengths of all flattend tensors are the same
    array_of_flat_lengths[0].push_back(flat_len);

    // strides of all flattend tensors are different
    repeat_n(
        [&](auto itensor) {
            std::size_t previous_stride = boost::get<itensor + 1>(*i_previous);
            array_of_flat_strides[itensor].push_back(previous_stride);
        },
        NTensorConstant);

    for(std::size_t itensor = 1; itensor < NTensor; ++itensor)
        array_of_flat_lengths[itensor] = array_of_flat_lengths[0];

    return create_tuple<NTensor>([&](auto itensor) {
        return TensorDescriptor{real_descriptors[itensor]->GetType(),
                                std::move(array_of_flat_lengths[itensor]),
                                std::move(array_of_flat_strides[itensor])};
    });
}

MIOPEN_INTERNALS_EXPORT void ScaleTensor(const Handle& handle,
                                         const TensorDescriptor& yDesc,
                                         Data_t y,
                                         const void* alpha,
                                         int offset = 0);

MIOPEN_INTERNALS_EXPORT void SetTensor(const Handle& handle,
                                       const TensorDescriptor& yDesc,
                                       Data_t y,
                                       const void* alpha,
                                       int offset = 0);

MIOPEN_INTERNALS_EXPORT void OpTensor(const Handle& handle,
                                      miopenTensorOp_t tensorOp,
                                      const void* alpha0,
                                      const TensorDescriptor& aTensorDesc,
                                      ConstData_t ATensor,
                                      const void* alpha1,
                                      const TensorDescriptor& bTensorDesc,
                                      ConstData_t BTensor,
                                      const void* beta,
                                      const TensorDescriptor& cTensorDesc,
                                      Data_t CTensor,
                                      size_t Aoffset         = 0,
                                      size_t Boffset         = 0,
                                      size_t Coffset         = 0,
                                      bool nonStandardSquash = false);

MIOPEN_INTERNALS_EXPORT void CopyTensor(const Handle& handle,
                                        const TensorDescriptor& srcDesc,
                                        ConstData_t src,
                                        const TensorDescriptor& dstDesc,
                                        Data_t dst,
                                        int srcOffset   = 0,
                                        int dstOffset   = 0,
                                        bool forseAsync = false);

MIOPEN_INTERNALS_EXPORT void CastTensor(const Handle& handle,
                                        const void* alpha,
                                        bool clamping, // Set to false for Bwd and WrW convolutions.
                                        const TensorDescriptor& srcDesc,
                                        ConstData_t src,
                                        const TensorDescriptor& dstDesc,
                                        Data_t dst,
                                        int srcOffset = 0,
                                        int dstOffset = 0);

MIOPEN_INTERNALS_EXPORT void TransformTensor(const Handle& handle,
                                             const void* alpha,
                                             const TensorDescriptor& xDesc,
                                             ConstData_t x,
                                             const void* beta,
                                             const TensorDescriptor& yDesc,
                                             Data_t y,
                                             size_t Xoffset = 0,
                                             size_t Yoffset = 0);
} // namespace miopen
#endif // GUARD_MIOPEN_TENSOR_OPPS_HPP_
