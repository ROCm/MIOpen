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
#ifndef _CK_REDUCTION_FUNCTIONS_HPP_
#define _CK_REDUCTION_FUNCTIONS_HPP_ 1

#include "reduction_common.hpp"
#include "reduction_operator.hpp"

namespace ck {
namespace detail {
template <ckNanPropagation_t nanPropaOpt, typename opReduce, typename compType>
struct binop_with_nan_check;

template <typename opReduce, typename compType>
struct binop_with_nan_check<CK_NOT_PROPAGATE_NAN, opReduce, compType>
{
    __device__ static void calculate(compType& accuVal, compType& currVal)
    {
        accuVal = opReduce{}(accuVal, currVal);
    };

    // this method can only be called when the opReduce is indexable
    __device__ static void
    calculate(compType& accuVal, compType currVal, int& accuIndex, int currIndex)
    {
        auto accuVal_new = opReduce{}(accuVal, currVal);

        if(accuVal != accuVal_new)
        {
            accuIndex = currIndex;
            accuVal   = accuVal_new;
        };
    };
};

template <typename opReduce, typename compType>
struct binop_with_nan_check<CK_PROPAGATE_NAN, opReduce, compType>
{
    __device__ static void calculate(compType& accuVal, compType& currVal)
    {
        if(isnan(currVal))
            accuVal = currVal;
        else
            accuVal = opReduce{}(accuVal, currVal);
    };

    // this method can only be called when the opReduce is indexable
    __device__ static void
    calculate(compType& accuVal, compType currVal, int& accuIndex, int currIndex)
    {
        compType accuVal_new;

        if(isnan(currVal))
            accuVal_new = currVal;
        else
            accuVal_new = opReduce{}(accuVal, currVal);

        if(accuVal != accuVal_new)
        {
            accuIndex = currIndex;
            accuVal   = accuVal_new;
        };
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
            compType currVal = static_cast<compType>(p_thread_buffer[i]);
            binop::calculate(accuData, currVal);
        };
    };

    // This operator is used by Direct_ThreadWise reduction method at first-time reduction
    __device__ static void
    reduce2(const DataType* p_thread_buffer, compType& accuData, int& accuIndex, int indexStart)
    {
        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = static_cast<compType>(p_thread_buffer[i]);
            int currIndex    = i + indexStart;
            binop::calculate(accuData, currVal, accuIndex, currIndex);
        };
    };

    // This operator is used by Direct_ThreadWise reduction method at second-time reduction
    __device__ static void reduce3(const DataType* p_thread_buffer,
                                   const int* thread_indices_buffer,
                                   compType& accuData,
                                   int& accuIndex)
    {
        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = static_cast<compType>(p_thread_buffer[i]);
            int currIndex    = thread_indices_buffer[i];
            binop::calculate(accuData, currVal, accuIndex, currIndex);
        };
    };

    __device__ static void set_buffer_value(DataType* p_thread_buffer, DataType value)
    {
        for(int i = 0; i < ThreadBufferLen; i++)
            p_thread_buffer[i] = value;
    };
};

template <typename DataType, int ThreadBufferLen, typename opReduce, ckNanPropagation_t nanPropaOpt>
struct warp_reduce
{
    using compType = typename opReduce::dataType;
    using binop    = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    __device__ static void reduce(const DataType* p_thread_buffer, compType& accuData)
    {
        compType lAccuData = opReduce::zeroVal;

        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = static_cast<compType>(p_thread_buffer[i]);
            binop::calculate(lAccuData, currVal);
        };

        // synchronize among all threads in this warp
        __all(1);

        for(int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            compType tmpVal;
            tmpVal = __shfl_down(lAccuData, offset, warpSize);
            binop::calculate(lAccuData, tmpVal);
            __all(1);
        };

        binop::calculate(accuData, lAccuData);
    };

    __device__ static void
    reduce2(const DataType* p_thread_buffer, compType& accuData, int& accuIndex, int indexStart)
    {
        compType lAccuData   = opReduce::zeroVal;
        int lAccuIndex       = 0;
        int thread_inwarp_id = get_thread_local_1d_id() % warpSize;

        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = static_cast<compType>(p_thread_buffer[i]);
            int currIndex    = thread_inwarp_id * ThreadBufferLen + i + indexStart;
            binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
        };

        // synchronize among all threads in this warp
        __all(1);

        for(int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            compType tmpVal;
            int tmpIndex;
            tmpVal   = __shfl_down(lAccuData, offset, warpSize);
            tmpIndex = __shfl_down(lAccuIndex, offset, warpSize);

            binop::calculate(lAccuData, tmpVal, lAccuIndex, tmpIndex);
            __all(1);
        };

        binop::calculate(accuData, lAccuData, accuIndex, lAccuIndex);
    };

    __device__ static void reduce3(const DataType* p_thread_buffer,
                                   const int* thread_indices_buffer,
                                   compType& accuData,
                                   int& accuIndex)
    {
        compType lAccuData   = opReduce::zeroVal;
        int lAccuIndex       = 0;
        int thread_inwarp_id = get_thread_local_1d_id() % warpSize;

        for(int i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal   = static_cast<compType>(p_thread_buffer[i]);
            compType currIndex = thread_indices_buffer[i];
            binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
        };

        // synchronize among all threads in this warp
        __all(1);

        for(int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            compType tmpVal;
            int tmpIndex;
            tmpVal   = __shfl_down(lAccuData, offset, warpSize);
            tmpIndex = __shfl_down(lAccuIndex, offset, warpSize);

            binop::calculate(lAccuData, tmpVal, lAccuIndex, tmpIndex);
            __all(1);
        };

        binop::calculate(accuData, lAccuData, accuIndex, lAccuIndex);
    };

    __device__ static void set_buffer_value(DataType* p_thread_buffer, DataType value)
    {
        for(int i = 0; i < ThreadBufferLen; i++)
            p_thread_buffer[i] = value;

        __all(1);
    };
};

template <typename bufferMatrixDesc,
          typename DataType,
          bool blockIsOneRow,
          typename opReduce,
          ckNanPropagation_t nanPropaOpt>
struct BlockwiseReduction_2d_block_buffer
{
    using compType = typename opReduce::dataType;
    constexpr static int BlockSize =
        blockIsOneRow ? bufferMatrixDesc::NCol() : bufferMatrixDesc::NRow();
    constexpr static int NumBlocks =
        blockIsOneRow ? bufferMatrixDesc::NRow() : bufferMatrixDesc::NCol();
    using binop = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    __device__ static void reduce(DataType* p_block_buffer, int toReduceBlocks, compType& accuData)
    {
        const int thread_local_id = get_thread_local_1d_id();
        compType lAccuData        = static_cast<compType>(opReduce::zeroVal);

        int offset;

        for(int otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
        {
            offset = blockIsOneRow
                         ? bufferMatrixDesc::CalculateOffset(otherDimInd, thread_local_id)
                         : bufferMatrixDesc::CalculateOffset(thread_local_id, otherDimInd);
            compType opData = static_cast<compType>(p_block_buffer[offset]);

            binop::calculate(lAccuData, opData);
        };

        offset = blockIsOneRow ? bufferMatrixDesc::CalculateOffset(0, thread_local_id)
                               : bufferMatrixDesc::CalculateOffset(thread_local_id, 0);

        p_block_buffer[offset] = lAccuData;

        __syncthreads();

        for(int indOffset = BlockSize / 2; indOffset > 0; indOffset /= 2)
        {
            if(thread_local_id < indOffset)
            {
                int offset1 = blockIsOneRow ? bufferMatrixDesc::CalculateOffset(0, thread_local_id)
                                            : bufferMatrixDesc::CalculateOffset(thread_local_id, 0);

                int offset2 =
                    blockIsOneRow
                        ? bufferMatrixDesc::CalculateOffset(0, thread_local_id + indOffset)
                        : bufferMatrixDesc::CalculateOffset(thread_local_id + indOffset, 0);

                compType opData1 = static_cast<compType>(p_block_buffer[offset1]);
                compType opData2 = static_cast<compType>(p_block_buffer[offset2]);
                binop::calculate(opData1, opData2);
                p_block_buffer[offset1] = static_cast<DataType>(opData1);
            };

            __syncthreads();
        };

        if(thread_local_id == 0)
        {
            compType tmpVal = static_cast<compType>(p_block_buffer[0]);

            binop::calculate(accuData, tmpVal);
        };
    };

    __device__ static void reduce2(DataType* p_block_buffer,
                                   int* block_indices_buffer,
                                   int toReduceBlocks,
                                   compType& accuData,
                                   int& accuIndex)
    {
        const int thread_local_id = get_thread_local_1d_id();
        compType lAccuData        = static_cast<compType>(opReduce::zeroVal);
        int lAccuIndex            = 0;

        for(int otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
        {
            for(int indOffset = BlockSize / 2; indOffset > 0; indOffset /= 2)
            {
                if(thread_local_id < indOffset)
                {
                    int offset1 =
                        blockIsOneRow
                            ? bufferMatrixDesc::CalculateOffset(otherDimInd, thread_local_id)
                            : bufferMatrixDesc::CalculateOffset(thread_local_id, otherDimInd);

                    int offset2 = blockIsOneRow ? bufferMatrixDesc::CalculateOffset(
                                                      otherDimInd, thread_local_id + indOffset)
                                                : bufferMatrixDesc::CalculateOffset(
                                                      thread_local_id + indOffset, otherDimInd);

                    compType currVal1 = static_cast<compType>(p_block_buffer[offset1]);
                    compType currVal2 = static_cast<compType>(p_block_buffer[offset2]);
                    int currIndex1    = static_cast<int>(block_indices_buffer[offset1]);
                    int currIndex2    = static_cast<int>(block_indices_buffer[offset2]);

                    binop::calculate(currVal1, currVal2, currIndex1, currIndex2);
                    p_block_buffer[offset1]       = static_cast<DataType>(currVal1);
                    block_indices_buffer[offset1] = currIndex1;
                };
                __syncthreads();
            };
            __syncthreads();
        };

        if(thread_local_id == 0)
        {
            for(int otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
            {
                int offset = blockIsOneRow ? bufferMatrixDesc::CalculateOffset(otherDimInd, 0)
                                           : bufferMatrixDesc::CalculateOffset(0, otherDimInd);

                compType tmpVal = static_cast<compType>(p_block_buffer[offset]);
                int tmpIndex    = static_cast<int>(block_indices_buffer[offset]);

                binop::calculate(lAccuData, tmpVal, lAccuIndex, tmpIndex);
            };

            binop::calculate(accuData, lAccuData, accuIndex, lAccuIndex);
        };
    };

    __device__ static void set_buffer_value(DataType* p_block_buffer, DataType value)
    {
        int thread_id = get_thread_local_1d_id();
        int offset;

        for(int otherDimInd = 0; otherDimInd < NumBlocks; otherDimInd++)
        {
            offset = blockIsOneRow ? bufferMatrixDesc::CalculateOffset(otherDimInd, thread_id)
                                   : bufferMatrixDesc::CalculateOffset(thread_id, otherDimInd);

            p_block_buffer[offset] = value;

            __syncthreads();
        };
    };

    __device__ static void init_buffer_indices(int* block_indices_buffer, int indexStart)
    {
        int thread_id = get_thread_local_1d_id();
        int offset;

        for(int otherDimInd = 0; otherDimInd < NumBlocks; otherDimInd++)
        {
            offset = blockIsOneRow ? bufferMatrixDesc::CalculateOffset(otherDimInd, thread_id)
                                   : bufferMatrixDesc::CalculateOffset(thread_id, otherDimInd);

            block_indices_buffer[offset] = offset + indexStart;

            __syncthreads();
        };
    };
};
}; // end of namespace ck

#endif
