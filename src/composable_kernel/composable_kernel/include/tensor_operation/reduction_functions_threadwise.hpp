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
#ifndef CK_REDUCTION_FUNCTIONS_THREADWISE_HPP
#define CK_REDUCTION_FUNCTIONS_THREADWISE_HPP

#include "data_type.hpp"

#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_binop.hpp"

namespace ck {

template <typename BufferType, typename opReduce, NanPropagation_t nanPropaOpt>
struct ThreadReduce
{
    using compType = typename opReduce::dataType;

    static_assert(BufferType::IsStaticBuffer(), "Thread-wise reduction needs use StaticBuffer!");

    static_assert(
        std::is_same<typename BufferType::type, compType>::value,
        "Data type of StaticBuffer for Thread-wise reduction should be same as the compType!");

    static constexpr index_t ThreadBufferLen = BufferType::Size();

    using binop = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    // This interface does not accumulate on indices
    __device__ static void Reduce(const BufferType& thread_buffer, compType& accuData)
    {
        static_for<0, ThreadBufferLen, 1>{}(
            [&](auto I) { binop::calculate(accuData, thread_buffer[I]); });
    };

    // This interface accumulates on both data values and indices and
    // is called by Direct_ThreadWise reduction method at first-time reduction
    __device__ static void
    Reduce2(const BufferType& thread_buffer, compType& accuData, int& accuIndex, int indexStart)
    {
        static_for<0, ThreadBufferLen, 1>{}([&](auto I) {
            int currIndex = I + indexStart;
            binop::calculate(accuData, thread_buffer[I], accuIndex, currIndex);
        });
    };

    // Set the elements in the per-thread buffer to a specific value
    // cppcheck-suppress constParameter
    __device__ static void set_buffer_value(BufferType& thread_buffer, compType value)
    {
        static_for<0, ThreadBufferLen, 1>{}([&](auto I) { thread_buffer(I) = value; });
    };

    // Execute unary operation on the per-thread buffer elements
    template <typename unary_op_type>
    __device__ static void operate_on_elements(unary_op_type& unary_op, BufferType& thread_buffer)
    {
        static_for<0, ThreadBufferLen, 1>{}(
            [&](auto I) { thread_buffer(I) = unary_op(thread_buffer[I]); });
    };
};

template <typename BufferType,
          typename IdxBufferType,
          typename opReduce,
          NanPropagation_t nanPropaOpt>
struct ThreadReduceWithIndicesInput
{
    using compType = typename opReduce::dataType;

    static_assert(BufferType::IsStaticBuffer(), "Thread-wise reduction needs use StaticBuffer!");
    static_assert(IdxBufferType::IsStaticBuffer(),
                  "Thread-wise reduction needs use StaticBuffer for indices!");

    static_assert(
        std::is_same<typename BufferType::type, compType>::value,
        "Data type of StaticBuffer for Thread-wise reduction should be same as the compType!");
    static_assert(std::is_same<typename IdxBufferType::type, index_t>::value,
                  "Indices type of StaticBuffer for Thread-wise reduction should be index_t!");

    static_assert(BufferType::Size() == IdxBufferType::Size(),
                  "StaticBuffers for data and indices should have the same sizes!");

    static constexpr index_t ThreadBufferLen = BufferType::Size();

    using binop = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    // This interface accumulates on both data values and indices and
    // is called by Direct_ThreadWise reduction method at second-time reduction
    __device__ static void Reduce(const BufferType& thread_buffer,
                                  const IdxBufferType& thread_indices_buffer,
                                  compType& accuData,
                                  int& accuIndex)
    {
        static_for<0, ThreadBufferLen, 1>{}([&](auto I) {
            binop::calculate(accuData, thread_buffer[I], accuIndex, thread_indices_buffer[I]);
        });
    };

    // Set the elements in the per-thread buffer to a specific value
    // cppcheck-suppress constParameter
    __device__ static void set_buffer_value(BufferType& thread_buffer, compType value)
    {
        static_for<0, ThreadBufferLen, 1>{}([&](auto I) { thread_buffer(I) = value; });
    };

    // Execute unary operation on the per-thread buffer elements
    template <typename unary_op_type>
    __device__ static void operate_on_elements(unary_op_type& unary_op, BufferType& thread_buffer)
    {
        static_for<0, ThreadBufferLen, 1>{}(
            [&](auto I) { thread_buffer(I) = unary_op(thread_buffer[I]); });
    };
};

}; // end of namespace ck

#endif
