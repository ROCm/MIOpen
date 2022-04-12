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
#ifndef CK_REDUCTION_FUNCTIONS_WARPWISE_HPP
#define CK_REDUCTION_FUNCTIONS_WARPWISE_HPP

#include "data_type.hpp"

#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_binop.hpp"

namespace ck {

template <typename BufferType, index_t BlockSize, typename opReduce, NanPropagation_t nanPropaOpt>
struct WarpReduce
{
    using compType = typename opReduce::dataType;
    using binop    = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    static_assert(BufferType::IsStaticBuffer(),
                  "Per-thread buffer for WarpWise reduction should be StaticBuffer!");
    static_assert(std::is_same<typename BufferType::type, compType>::value,
                  "Data type of per-thread StaticBuffer for WarpWise reduction should be same as "
                  "the compType!");

    static constexpr index_t ThreadBufferLen = BufferType::Size();
    static constexpr bool have_builtin_shuffle =
        std::is_same<compType, float>::value || std::is_same<compType, double>::value;

    // This interface does not accumulate on indices
    __device__ static void Reduce(const BufferType& thread_buffer, compType& accuData)
    {
        if constexpr(have_builtin_shuffle)
            ReduceImpl1(thread_buffer, accuData);
        else
            ReduceImpl2(thread_buffer, accuData);
    };

    // This interface implementation uses HIP built-in device shuffling functions
    __device__ static void ReduceImpl1(const BufferType& thread_buffer, compType& accuData)
    {
        compType lAccuData = opReduce::GetZeroVal();

        static_for<0, ThreadBufferLen, 1>{}(
            [&](auto I) { binop::calculate(lAccuData, thread_buffer[I]); });

        // synchronize among all threads in this warp
        __all(1);

        for(index_t stride = warpSize / 2; stride > 0; stride /= 2)
        {
            compType tmpVal = __shfl_down(lAccuData, stride, warpSize);
            binop::calculate(lAccuData, tmpVal);
            __all(1);
        }

        binop::calculate(accuData, lAccuData);
    };

    // This interface implementation does not use HIP built-in device shuffling functions
    // since for fp16, built-in shuffling functions is not provided by HIP
    __device__ static void ReduceImpl2(const BufferType& thread_buffer, compType& accuData)
    {
        compType lAccuData = opReduce::GetZeroVal();

        static_for<0, ThreadBufferLen, 1>{}(
            [&](auto I) { binop::calculate(lAccuData, thread_buffer[I]); });

        __syncthreads();

        index_t thread_id        = get_thread_local_1d_id();
        index_t warpId           = thread_id / warpSize;
        index_t thread_inwarp_id = thread_id % warpSize;

        __shared__ compType shuffle_buffer[BlockSize];

        compType* myBuffer = &shuffle_buffer[warpId * warpSize];

        myBuffer[thread_inwarp_id] = lAccuData;

        __syncthreads();

        for(index_t stride = warpSize / 2; stride > 0; stride /= 2)
        {
            if(thread_inwarp_id < stride)
            {
                compType currVal1 = myBuffer[thread_inwarp_id];
                compType currVal2 = myBuffer[thread_inwarp_id + stride];

                binop::calculate(currVal1, currVal2);

                myBuffer[thread_inwarp_id] = currVal1;
            }

            __syncthreads();
        }
        if(thread_inwarp_id == 0)
            binop::calculate(accuData, myBuffer[0]);
    };

    // This interface accumulates on both data values and indices and is called by Direct_WarpWise
    // reduction method at first-time reduction
    __device__ static void
    Reduce2(const BufferType& thread_buffer, compType& accuData, int& accuIndex, int indexStart)
    {
        if constexpr(have_builtin_shuffle)
            Reduce2Impl1(thread_buffer, accuData, accuIndex, indexStart);
        else
            Reduce2Impl2(thread_buffer, accuData, accuIndex, indexStart);
    };

    // This interface implementation uses HIP built-in device shuffling functions
    __device__ static void Reduce2Impl1(const BufferType& thread_buffer,
                                        compType& accuData,
                                        int& accuIndex,
                                        int indexStart)
    {
        compType lAccuData       = opReduce::GetZeroVal();
        int lAccuIndex           = 0;
        index_t thread_inwarp_id = get_thread_local_1d_id() % warpSize;

        static_for<0, ThreadBufferLen, 1>{}([&](auto I) {
            int currIndex = thread_inwarp_id * ThreadBufferLen + I + indexStart;
            binop::calculate(lAccuData, thread_buffer[I], lAccuIndex, currIndex);
        });

        // synchronize among all threads in this warp
        __all(1);

        for(index_t stride = 1; stride < warpSize; stride *= 2)
        {
            compType tmpVal = __shfl_down(lAccuData, stride, warpSize);
            int tmpIndex    = __shfl_down(lAccuIndex, stride, warpSize);

            binop::calculate(lAccuData, tmpVal, lAccuIndex, tmpIndex);
            __all(1);
        }

        if(thread_inwarp_id == 0)
            binop::calculate(accuData, lAccuData, accuIndex, lAccuIndex);
    };

    // This interface implementation does not use HIP built-in device shuffling functions since for
    // fp16, built-in shuffling functions is not provided by HIP
    __device__ static void Reduce2Impl2(const BufferType& thread_buffer,
                                        compType& accuData,
                                        int& accuIndex,
                                        int indexStart)
    {
        compType lAccuData       = opReduce::GetZeroVal();
        int lAccuIndex           = 0;
        index_t thread_id        = get_thread_local_1d_id();
        index_t warpId           = thread_id / warpSize;
        index_t thread_inwarp_id = thread_id % warpSize;

        static_for<0, ThreadBufferLen, 1>{}([&](auto I) {
            int currIndex = thread_inwarp_id * ThreadBufferLen + I + indexStart;
            binop::calculate(lAccuData, thread_buffer[I], lAccuIndex, currIndex);
        });

        __shared__ compType shuffle_data_buffer[BlockSize];
        __shared__ int shuffle_indices_buffer[BlockSize];

        compType* myDataBuffer = &shuffle_data_buffer[warpId * warpSize];
        int* myIndicesBuffer   = &shuffle_indices_buffer[warpId * warpSize];

        myDataBuffer[thread_inwarp_id]    = lAccuData;
        myIndicesBuffer[thread_inwarp_id] = lAccuIndex;

        __syncthreads();

        for(index_t stride = 1; stride < warpSize; stride *= 2)
        {
            compType currVal1 = myDataBuffer[thread_inwarp_id];
            compType currVal2 = myDataBuffer[thread_inwarp_id + stride];
            int currIndex1    = myIndicesBuffer[thread_inwarp_id];
            int currIndex2    = myIndicesBuffer[thread_inwarp_id + stride];

            binop::calculate(currVal1, currVal2, currIndex1, currIndex2);

            myDataBuffer[thread_inwarp_id]    = currVal1;
            myIndicesBuffer[thread_inwarp_id] = currIndex1;

            __syncthreads();
        }

        if(thread_inwarp_id == 0)
            binop::calculate(accuData, myDataBuffer[0], accuIndex, myIndicesBuffer[0]);
    };

    // cppcheck-suppress constParameter
    __device__ static void set_buffer_value(BufferType& thread_buffer, compType value)
    {
        static_for<0, ThreadBufferLen, 1>{}([&](auto I) { thread_buffer(I) = value; });

        __all(1);
    };

    // Execute unary operation on the per-thread buffer elements
    template <typename unary_op_type>
    __device__ static void operate_on_elements(unary_op_type& unary_op, BufferType& thread_buffer)
    {
        static_for<0, ThreadBufferLen, 1>{}(
            [&](auto I) { thread_buffer(I) = unary_op(thread_buffer[I]); });

        __all(1);
    };
};

template <typename BufferType,
          typename IdxBufferType,
          index_t BlockSize,
          typename opReduce,
          NanPropagation_t nanPropaOpt>
struct WarpReduceWithIndicesInput
{
    using compType = typename opReduce::dataType;
    using binop    = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    static_assert(BufferType::IsStaticBuffer(),
                  "Per-thread buffer for WarpWise reduction should be StaticBuffer!");
    static_assert(IdxBufferType::IsStaticBuffer(),
                  "Per-thread buffer for WarpWise reduction should be StaticBuffer for indices!");

    static_assert(std::is_same<typename BufferType::type, compType>::value,
                  "Data type of per-thread StaticBuffer for WarpWise reduction should be same as "
                  "the compType!");
    static_assert(
        std::is_same<typename IdxBufferType::type, index_t>::value,
        "Indices type per-thread of StaticBuffer for WarpWise reduction should be index_t!");

    static_assert(BufferType::Size() == IdxBufferType::Size(),
                  "StaticBuffers for data and indices should have the same sizes!");

    static constexpr index_t ThreadBufferLen = BufferType::Size();
    static constexpr bool have_builtin_shuffle =
        std::is_same<compType, float>::value || std::is_same<compType, double>::value;

    // This interface accumulates on both data values and indices and is called by Direct_WarpWise
    // reduction method at second-time reduction
    __device__ static void Reduce(const BufferType& thread_buffer,
                                  const IdxBufferType& thread_indices_buffer,
                                  compType& accuData,
                                  int& accuIndex)
    {
        if constexpr(have_builtin_shuffle)
            ReduceImpl1(thread_buffer, thread_indices_buffer, accuData, accuIndex);
        else
            ReduceImpl2(thread_buffer, thread_indices_buffer, accuData, accuIndex);
    };

    // This interface implementation uses HIP built-in device shuffling functions
    __device__ static void ReduceImpl1(const BufferType& thread_buffer,
                                       const IdxBufferType& thread_indices_buffer,
                                       compType& accuData,
                                       int& accuIndex)
    {
        compType lAccuData = opReduce::GetZeroVal();
        int lAccuIndex     = 0;

        static_for<0, ThreadBufferLen, 1>{}([&](auto I) {
            binop::calculate(lAccuData, thread_buffer[I], lAccuIndex, thread_indices_buffer[I]);
        });

        // synchronize among all threads in this warp
        __all(1);

        for(index_t stride = 1; stride < warpSize; stride *= 2)
        {
            compType tmpVal = __shfl_down(lAccuData, stride, warpSize);
            int tmpIndex    = __shfl_down(lAccuIndex, stride, warpSize);

            binop::calculate(lAccuData, tmpVal, lAccuIndex, tmpIndex);
            __all(1);
        }

        binop::calculate(accuData, lAccuData, accuIndex, lAccuIndex);
    };

    // This interface implementation does not use HIP built-in device shuffling functions
    // since for fp16, built-in shuffling functions is not provided by HIP
    __device__ static void ReduceImpl2(const BufferType& thread_buffer,
                                       const IdxBufferType& thread_indices_buffer,
                                       compType& accuData,
                                       int& accuIndex)
    {
        compType lAccuData       = opReduce::GetZeroVal();
        int lAccuIndex           = 0;
        index_t thread_id        = get_thread_local_1d_id();
        index_t warpId           = thread_id / warpSize;
        index_t thread_inwarp_id = thread_id % warpSize;

        static_for<0, ThreadBufferLen, 1>{}([&](auto I) {
            binop::calculate(lAccuData, thread_buffer[I], lAccuIndex, thread_indices_buffer[I]);
        });

        __shared__ compType shuffle_data_buffer[BlockSize];
        __shared__ int shuffle_indices_buffer[BlockSize];

        compType* myDataBuffer = &shuffle_data_buffer[warpId * warpSize];
        int* myIndicesBuffer   = &shuffle_indices_buffer[warpId * warpSize];

        myDataBuffer[thread_inwarp_id]    = lAccuData;
        myIndicesBuffer[thread_inwarp_id] = lAccuIndex;

        __syncthreads();

        for(index_t stride = 1; stride < warpSize; stride *= 2)
        {
            compType currVal1 = myDataBuffer[thread_inwarp_id];
            compType currVal2 = myDataBuffer[thread_inwarp_id + stride];
            int currIndex1    = myIndicesBuffer[thread_inwarp_id];
            int currIndex2    = myIndicesBuffer[thread_inwarp_id + stride];

            binop::calculate(currVal1, currVal2, currIndex1, currIndex2);

            myDataBuffer[thread_inwarp_id]    = currVal1;
            myIndicesBuffer[thread_inwarp_id] = currIndex1;

            __syncthreads();
        }

        if(thread_inwarp_id == 0)
            binop::calculate(accuData, myDataBuffer[0], accuIndex, myIndicesBuffer[0]);
    };

    // cppcheck-suppress constParameter
    __device__ static void set_buffer_value(BufferType& thread_buffer, compType value)
    {
        static_for<0, ThreadBufferLen, 1>{}([&](auto I) { thread_buffer(I) = value; });

        __all(1);
    };

    // Execute unary operation on the per-thread buffer elements
    template <typename unary_op_type>
    __device__ static void operate_on_elements(unary_op_type& unary_op, BufferType& thread_buffer)
    {
        static_for<0, ThreadBufferLen, 1>{}(
            [&](auto I) { thread_buffer(I) = unary_op(thread_buffer[I]); });

        __all(1);
    };
};

}; // end of namespace ck

#endif
