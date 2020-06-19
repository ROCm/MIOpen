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
#ifndef CK_REDUCTION_FUNCTIONS_HPP
#define CK_REDUCTION_FUNCTIONS_HPP

#include <config.hpp>

#include "reduction_common.hpp"
#include "reduction_operator.hpp"

namespace ck {
namespace detail {

template <typename T>
__device__ bool IsNan(T x)
{
    // for float and double, use the builtin hip kernel functions
    return (isnan(x));
};

template <>
__device__ bool IsNan<half>(half x)
{
    return (__hisnan(x));
};

template <ckNanPropagation_t nanPropaOpt, typename opReduce, typename compType>
struct binop_with_nan_check;

template <typename opReduce, typename compType>
struct binop_with_nan_check<CK_NOT_PROPAGATE_NAN, opReduce, compType>
{
    __device__ static inline void calculate(const compType& accuVal, compType currVal)
    {
        opReduce{}(const_cast<compType&>(accuVal), currVal);
    };

    // this method can only be called when the opReduce is indexable
    __device__ static inline void
    calculate(const compType& accuVal, compType currVal, int& accuIndex, int currIndex)
    {
        bool changed = false;

        opReduce{}(const_cast<compType&>(accuVal), currVal, changed);

        if(changed)
            accuIndex = currIndex;
    };
};

template <typename opReduce, typename compType>
struct binop_with_nan_check<CK_PROPAGATE_NAN, opReduce, compType>
{
    __device__ static inline void calculate(compType& accuVal, compType currVal)
    {
        if(IsNan(currVal))
            accuVal = currVal;
        else
            opReduce{}(accuVal, currVal);
    };

    // this method can only be called when the opReduce is indexable
    __device__ static inline void
    calculate(compType& accuVal, compType currVal, int& accuIndex, int currIndex)
    {
        if(IsNan(currVal))
        {
            accuVal   = currVal;
            accuIndex = currIndex;
        }
        else
        {
            bool changed = false;

            opReduce{}(accuVal, currVal, changed);

            if(changed)
                accuIndex = currIndex;
        }
    };
};
}; // namespace detail

template <typename DataType, int ThreadBufferLen, typename opReduce, ckNanPropagation_t nanPropaOpt>
struct thread_reduce
{
    using compType = typename opReduce::dataType;
    using binop    = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    __device__ static void reduce(const DataType* p_thread_buffer, compType& accuData)
    {
        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            binop::calculate(accuData, currVal);
        }
    };

    // This operator is used by Direct_ThreadWise reduction method at first-time reduction
    __device__ static void
    reduce2(const DataType* p_thread_buffer, compType& accuData, int& accuIndex, int indexStart)
    {
        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            int currIndex    = i + indexStart;
            binop::calculate(accuData, currVal, accuIndex, currIndex);
        }
    };

    // This operator is used by Direct_ThreadWise reduction method at second-time reduction
    __device__ static void reduce3(const DataType* p_thread_buffer,
                                   const int* thread_indices_buffer,
                                   compType& accuData,
                                   int& accuIndex)
    {
        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            int currIndex    = thread_indices_buffer[i];
            binop::calculate(accuData, currVal, accuIndex, currIndex);
        }
    };

    __device__ static void set_buffer_value(DataType* p_thread_buffer, DataType value)
    {
        for(int i              = 0; i < ThreadBufferLen; i++)
            p_thread_buffer[i] = value;
    };
};

template <typename DataType,
          int BlockSize,
          int ThreadBufferLen,
          typename opReduce,
          ckNanPropagation_t nanPropaOpt>
struct warp_reduce
{
    using compType = typename opReduce::dataType;
    using binop    = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;
    constexpr static bool have_builtin_shuffle = std::is_same<compType, float>::value;

    __device__ static void reduce(const DataType* p_thread_buffer, compType& accuData)
    {
        static_if<have_builtin_shuffle>{}([&](auto) {
            reduceImpl1(p_thread_buffer, accuData);
        }).Else([&](auto) { reduceImpl2(p_thread_buffer, accuData); });
    };

    __device__ static void reduceImpl1(const DataType* p_thread_buffer, compType& accuData)
    {
        compType lAccuData = opReduce::getZeroVal();

        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            binop::calculate(lAccuData, currVal);
        }

        // synchronize among all threads in this warp
        __all(1);

        for(int stride = warpSize / 2; stride > 0; stride /= 2)
        {
            compType tmpVal = __shfl_down(lAccuData, stride, warpSize);
            binop::calculate(lAccuData, tmpVal);
            __all(1);
        }

        binop::calculate(accuData, lAccuData);
    };

    __device__ static void reduceImpl2(const DataType* p_thread_buffer, compType& accuData)
    {
        compType lAccuData = opReduce::getZeroVal();

        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            binop::calculate(lAccuData, currVal);
        }

        __syncthreads();

        int thread_id        = get_thread_local_1d_id();
        int warpId           = thread_id / warpSize;
        int thread_inwarp_id = thread_id % warpSize;

        __shared__ compType shuffle_buffer[BlockSize];

        compType* myBuffer = &shuffle_buffer[warpId * warpSize];

        myBuffer[thread_inwarp_id] = lAccuData;

        __syncthreads();

        for(int stride = warpSize / 2; stride > 0; stride /= 2)
        {
            if(thread_inwarp_id < warpSize)
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

    __device__ static void
    reduce2(const DataType* p_thread_buffer, compType& accuData, int& accuIndex, int indexStart)
    {
        static_if<have_builtin_shuffle>{}([&](auto) {
            reduce2Impl1(p_thread_buffer, accuData, accuIndex, indexStart);
        }).Else([&](auto) { reduce2Impl2(p_thread_buffer, accuData, accuIndex, indexStart); });
    };

    __device__ static void reduce2Impl1(const DataType* p_thread_buffer,
                                        compType& accuData,
                                        int& accuIndex,
                                        int indexStart)
    {
        compType lAccuData   = opReduce::getZeroVal();
        int lAccuIndex       = 0;
        int thread_inwarp_id = get_thread_local_1d_id() % warpSize;

        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            int currIndex    = thread_inwarp_id * ThreadBufferLen + i + indexStart;
            binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
        }

        // synchronize among all threads in this warp
        __all(1);

        for(int stride = 1; stride < warpSize; stride *= 2)
        {
            compType tmpVal = __shfl_down(lAccuData, stride, warpSize);
            int tmpIndex    = __shfl_down(lAccuIndex, stride, warpSize);

            binop::calculate(lAccuData, tmpVal, lAccuIndex, tmpIndex);
            __all(1);
        }

        if(thread_inwarp_id == 0)
            binop::calculate(accuData, lAccuData, accuIndex, lAccuIndex);
    };

    __device__ static void reduce2Impl2(const DataType* p_thread_buffer,
                                        compType& accuData,
                                        int& accuIndex,
                                        int indexStart)
    {
        compType lAccuData   = opReduce::getZeroVal();
        int lAccuIndex       = 0;
        int thread_id        = get_thread_local_1d_id();
        int warpId           = thread_id / warpSize;
        int thread_inwarp_id = thread_id % warpSize;

        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            int currIndex    = thread_inwarp_id * ThreadBufferLen + i + indexStart;
            binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
        }

        __shared__ compType shuffle_data_buffer[BlockSize];
        __shared__ int shuffle_indices_buffer[BlockSize];

        compType* myDataBuffer = &shuffle_data_buffer[warpId * warpSize];
        int* myIndicesBuffer   = &shuffle_indices_buffer[warpId * warpSize];

        myDataBuffer[thread_inwarp_id]    = lAccuData;
        myIndicesBuffer[thread_inwarp_id] = lAccuIndex;

        __syncthreads();

        for(int stride = 1; stride < warpSize; stride *= 2)
        {
            if(thread_inwarp_id < warpSize)
            {
                compType currVal1 = myDataBuffer[thread_inwarp_id];
                compType currVal2 = myDataBuffer[thread_inwarp_id + stride];
                int currIndex1    = myIndicesBuffer[thread_inwarp_id];
                int currIndex2    = myIndicesBuffer[thread_inwarp_id + stride];

                binop::calculate(currVal1, currVal2, currIndex1, currIndex2);

                myDataBuffer[thread_inwarp_id]    = currVal1;
                myIndicesBuffer[thread_inwarp_id] = currIndex1;
            }
            __syncthreads();
        }

        if(thread_inwarp_id == 0)
            binop::calculate(accuData, myDataBuffer[0], accuIndex, myIndicesBuffer[0]);
    };

    __device__ static void reduce3(const DataType* p_thread_buffer,
                                   const int* thread_indices_buffer,
                                   compType& accuData,
                                   int& accuIndex)
    {
        static_if<have_builtin_shuffle>{}([&](auto) {
            reduce3Impl1(p_thread_buffer, thread_indices_buffer, accuData, accuIndex);
        }).Else([&](auto) {
            reduce3Impl2(p_thread_buffer, thread_indices_buffer, accuData, accuIndex);
        });
    };

    __device__ static void reduce3Impl1(const DataType* p_thread_buffer,
                                        const int* thread_indices_buffer,
                                        compType& accuData,
                                        int& accuIndex)
    {
        compType lAccuData = opReduce::getZeroVal();
        int lAccuIndex     = 0;

        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal   = type_convert<compType>{}(p_thread_buffer[i]);
            compType currIndex = thread_indices_buffer[i];
            binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
        }

        // synchronize among all threads in this warp
        __all(1);

        for(int stride = 1; stride < warpSize; stride *= 2)
        {
            compType tmpVal = __shfl_down(lAccuData, stride, warpSize);
            int tmpIndex    = __shfl_down(lAccuIndex, stride, warpSize);

            binop::calculate(lAccuData, tmpVal, lAccuIndex, tmpIndex);
            __all(1);
        }

        binop::calculate(accuData, lAccuData, accuIndex, lAccuIndex);
    };

    __device__ static void reduce3Impl2(const DataType* p_thread_buffer,
                                        const int* thread_indices_buffer,
                                        compType& accuData,
                                        int& accuIndex)
    {
        compType lAccuData   = opReduce::getZeroVal();
        int lAccuIndex       = 0;
        int thread_id        = get_thread_local_1d_id();
        int warpId           = thread_id / warpSize;
        int thread_inwarp_id = thread_id % warpSize;

        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal   = type_convert<compType>{}(p_thread_buffer[i]);
            compType currIndex = thread_indices_buffer[i];
            binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
        }

        __shared__ compType shuffle_data_buffer[BlockSize];
        __shared__ int shuffle_indices_buffer[BlockSize];

        compType* myDataBuffer = &shuffle_data_buffer[warpId * warpSize];
        int* myIndicesBuffer   = &shuffle_indices_buffer[warpId * warpSize];

        myDataBuffer[thread_inwarp_id]    = lAccuData;
        myIndicesBuffer[thread_inwarp_id] = lAccuIndex;

        __syncthreads();

        for(int stride = 1; stride < warpSize; stride *= 2)
        {
            if(thread_inwarp_id < warpSize)
            {
                compType currVal1 = myDataBuffer[thread_inwarp_id];
                compType currVal2 = myDataBuffer[thread_inwarp_id + stride];
                int currIndex1    = myIndicesBuffer[thread_inwarp_id];
                int currIndex2    = myIndicesBuffer[thread_inwarp_id + stride];

                binop::calculate(currVal1, currVal2, currIndex1, currIndex2);

                myDataBuffer[thread_inwarp_id]    = currVal1;
                myIndicesBuffer[thread_inwarp_id] = currIndex1;
            }
            __syncthreads();
        }

        if(thread_inwarp_id == 0)
            binop::calculate(accuData, myDataBuffer[0], accuIndex, myIndicesBuffer[0]);
    };

    __device__ static void set_buffer_value(DataType* p_thread_buffer, DataType value)
    {
        for(int i              = 0; i < ThreadBufferLen; i++)
            p_thread_buffer[i] = value;

        __all(1);
    };
};

template <typename buffer2dDesc,
          typename DataType,
          bool blockIsOneRow,
          typename opReduce,
          ckNanPropagation_t nanPropaOpt>
struct BlockwiseReduction_2d_block_buffer
{
    using compType = typename opReduce::dataType;
    constexpr static int BlockSize =
        blockIsOneRow ? buffer2dDesc::GetLengths()[1] : buffer2dDesc::GetLengths()[0];
    constexpr static int NumBlocks =
        blockIsOneRow ? buffer2dDesc::GetLengths()[0] : buffer2dDesc::GetLengths()[1];
    using binop = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    __device__ static void reduce(DataType* p_block_buffer, int toReduceBlocks, compType& accuData)
    {
        const int thread_local_id = get_thread_local_1d_id();
        compType lAccuData        = opReduce::getZeroVal();

        int offset;
        for(int otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
        {
            offset = blockIsOneRow ? buffer2dDesc::CalculateOffset({otherDimInd, thread_local_id})
                                   : buffer2dDesc::CalculateOffset({thread_local_id, otherDimInd});
            compType opData = type_convert<compType>{}(p_block_buffer[offset]);

            binop::calculate(lAccuData, opData);
        }

        offset = blockIsOneRow ? buffer2dDesc::CalculateOffset({0, thread_local_id})
                               : buffer2dDesc::CalculateOffset({thread_local_id, 0});

        p_block_buffer[offset] = lAccuData;

        __syncthreads();

        for(int indOffset = BlockSize / 2; indOffset > 0; indOffset /= 2)
        {
            if(thread_local_id < indOffset)
            {
                int offset1 = blockIsOneRow ? buffer2dDesc::CalculateOffset({0, thread_local_id})
                                            : buffer2dDesc::CalculateOffset({thread_local_id, 0});

                int offset2 = blockIsOneRow
                                  ? buffer2dDesc::CalculateOffset({0, thread_local_id + indOffset})
                                  : buffer2dDesc::CalculateOffset({thread_local_id + indOffset, 0});

                compType opData1 = type_convert<compType>{}(p_block_buffer[offset1]);
                compType opData2 = type_convert<compType>{}(p_block_buffer[offset2]);
                binop::calculate(opData1, opData2);
                p_block_buffer[offset1] = type_convert<DataType>{}(opData1);
            }

            __syncthreads();
        }

        if(thread_local_id == 0)
        {
            compType tmpVal = type_convert<compType>{}(p_block_buffer[0]);

            binop::calculate(accuData, tmpVal);
        }
    };

    __device__ static void reduce2(DataType* p_block_buffer,
                                   int* block_indices_buffer,
                                   int toReduceBlocks,
                                   compType& accuData,
                                   int& accuIndex)
    {
        const int thread_local_id = get_thread_local_1d_id();
        compType lAccuData        = opReduce::getZeroVal();
        int lAccuIndex            = 0;

        static_if<blockIsOneRow>{}([&](auto) {
            for(int otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
            {
                for(int indOffset = 1; indOffset < BlockSize; indOffset *= 2)
                {
                    if(thread_local_id % (indOffset * 2) == 0)
                    {
                        int offset1 = buffer2dDesc::CalculateOffset({otherDimInd, thread_local_id});
                        int offset2 = buffer2dDesc::CalculateOffset(
                            {otherDimInd, thread_local_id + indOffset});

                        compType currVal1 = type_convert<compType>{}(p_block_buffer[offset1]);
                        compType currVal2 = type_convert<compType>{}(p_block_buffer[offset2]);
                        int currIndex1    = block_indices_buffer[offset1];
                        int currIndex2    = block_indices_buffer[offset2];

                        binop::calculate(currVal1, currVal2, currIndex1, currIndex2);
                        p_block_buffer[offset1]       = type_convert<DataType>{}(currVal1);
                        block_indices_buffer[offset1] = currIndex1;
                    }
                }
                __syncthreads();
            }

            if(thread_local_id == 0)
            {
                for(int otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
                {
                    int offset = buffer2dDesc::CalculateOffset({otherDimInd, 0});

                    compType tmpVal = type_convert<compType>{}(p_block_buffer[offset]);
                    int tmpIndex    = block_indices_buffer[offset];

                    binop::calculate(lAccuData, tmpVal, lAccuIndex, tmpIndex);
                }

                binop::calculate(accuData, lAccuData, accuIndex, lAccuIndex);
            }
        }).Else([&](auto) {
            int offset;

            for(int otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
            {
                offset           = buffer2dDesc::CalculateOffset({thread_local_id, otherDimInd});
                compType currVal = type_convert<compType>{}(p_block_buffer[offset]);
                int currIndex    = block_indices_buffer[offset];

                binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
            }

            offset = buffer2dDesc::CalculateOffset({thread_local_id, 0});

            p_block_buffer[offset]       = lAccuData;
            block_indices_buffer[offset] = lAccuIndex;

            __syncthreads();

            for(int indOffset = 1; indOffset < BlockSize; indOffset *= 2)
            {
                if(thread_local_id % (indOffset * 2) == 0)
                {
                    int offset1 = buffer2dDesc::CalculateOffset({thread_local_id, 0});
                    int offset2 = buffer2dDesc::CalculateOffset({thread_local_id + indOffset, 0});

                    compType currVal1 = type_convert<compType>{}(p_block_buffer[offset1]);
                    compType currVal2 = type_convert<compType>{}(p_block_buffer[offset2]);
                    int currIndex1    = block_indices_buffer[offset1];
                    int currIndex2    = block_indices_buffer[offset2];

                    binop::calculate(currVal1, currVal2, currIndex1, currIndex2);
                    p_block_buffer[offset1]       = type_convert<DataType>{}(currVal1);
                    block_indices_buffer[offset1] = currIndex1;
                }

                __syncthreads();
            }

            if(thread_local_id == 0)
            {
                compType tmpVal = type_convert<compType>{}(p_block_buffer[0]);
                int tmpIndex    = block_indices_buffer[0];

                binop::calculate(accuData, tmpVal, accuIndex, tmpIndex);
            }
        });
    };

    __device__ static void set_buffer_value(DataType* p_block_buffer, DataType value)
    {
        int thread_id = get_thread_local_1d_id();

        for(int otherDimInd = 0; otherDimInd < NumBlocks; otherDimInd++)
        {
            int offset = blockIsOneRow ? buffer2dDesc::CalculateOffset({otherDimInd, thread_id})
                                       : buffer2dDesc::CalculateOffset({thread_id, otherDimInd});

            p_block_buffer[offset] = value;

            __syncthreads();
        }
    };

    __device__ static void init_buffer_indices(int* block_indices_buffer, int indexStart)
    {
        int thread_id = get_thread_local_1d_id();

        for(int otherDimInd = 0; otherDimInd < NumBlocks; otherDimInd++)
        {
            int offset = blockIsOneRow ? buffer2dDesc::CalculateOffset({otherDimInd, thread_id})
                                       : buffer2dDesc::CalculateOffset({thread_id, otherDimInd});

            block_indices_buffer[offset] = offset + indexStart;

            __syncthreads();
        }
    };
};

}; // end of namespace ck

#endif
