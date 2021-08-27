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
#ifndef CK_REDUCTION_FUNCTIONS_BLOCKWISE_HPP
#define CK_REDUCTION_FUNCTIONS_BLOCKWISE_HPP

#include "data_type.hpp"

#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_binop.hpp"

namespace ck {

template <typename buffer2dDescType,
          bool blockIsOneRow,
          typename opReduce,
          NanPropagation_t nanPropaOpt>
struct BlockwiseReduction_2d_block_buffer
{
    using compType = typename opReduce::dataType;

    static constexpr auto buffer2dDesc = buffer2dDescType{};

    static constexpr index_t BlockSize =
        blockIsOneRow ? buffer2dDesc.GetLength(Number<1>{}) : buffer2dDesc.GetLength(Number<0>{});
    static constexpr index_t NumBlocks =
        blockIsOneRow ? buffer2dDesc.GetLength(Number<0>{}) : buffer2dDesc.GetLength(Number<1>{});
    using binop = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    // This interface does not accumulate on indices
    template <typename BufferType>
    __device__ static void
    Reduce(BufferType& block_buffer, index_t toReduceBlocks, compType& accuData)
    {
        const index_t thread_local_id = get_thread_local_1d_id();
        compType lAccuData            = opReduce::GetZeroVal();

        index_t offset;
        for(index_t otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
        {
            offset = blockIsOneRow
                         ? buffer2dDesc.CalculateOffset(make_tuple(otherDimInd, thread_local_id))
                         : buffer2dDesc.CalculateOffset(make_tuple(thread_local_id, otherDimInd));
            compType opData = type_convert<compType>{}(block_buffer[offset]);

            binop::calculate(lAccuData, opData);
        }

        offset = blockIsOneRow ? buffer2dDesc.CalculateOffset(make_tuple(0, thread_local_id))
                               : buffer2dDesc.CalculateOffset(make_tuple(thread_local_id, 0));

        block_buffer(offset) = lAccuData;

        __syncthreads();

        for(index_t indOffset = BlockSize / 2; indOffset > 0; indOffset /= 2)
        {
            if(thread_local_id < indOffset)
            {
                index_t offset1 =
                    blockIsOneRow ? buffer2dDesc.CalculateOffset(make_tuple(0, thread_local_id))
                                  : buffer2dDesc.CalculateOffset(make_tuple(thread_local_id, 0));

                index_t offset2 =
                    blockIsOneRow
                        ? buffer2dDesc.CalculateOffset(make_tuple(0, thread_local_id + indOffset))
                        : buffer2dDesc.CalculateOffset(make_tuple(thread_local_id + indOffset, 0));

                compType opData1 = type_convert<compType>{}(block_buffer[offset1]);
                compType opData2 = type_convert<compType>{}(block_buffer[offset2]);
                binop::calculate(opData1, opData2);
                block_buffer(offset1) = type_convert<compType>{}(opData1);
            }

            __syncthreads();
        }

        if(thread_local_id == 0)
        {
            compType tmpVal = type_convert<compType>{}(block_buffer[0]);

            binop::calculate(accuData, tmpVal);
        }
    };

    // This interface accumulates on both data values and indices
    template <typename BufferType, typename IdxBufferType>
    __device__ static void Reduce2(BufferType& block_buffer,
                                   IdxBufferType& block_indices_buffer,
                                   index_t toReduceBlocks,
                                   compType& accuData,
                                   int& accuIndex)
    {
        const index_t thread_local_id = get_thread_local_1d_id();
        compType lAccuData            = opReduce::GetZeroVal();
        int lAccuIndex                = 0;

        if constexpr(blockIsOneRow)
        {
            for(index_t otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
            {
                for(index_t indOffset = 1; indOffset < BlockSize; indOffset *= 2)
                {
                    if(thread_local_id % (indOffset * 2) == 0)
                    {
                        index_t offset1 =
                            buffer2dDesc.CalculateOffset(make_tuple(otherDimInd, thread_local_id));
                        index_t offset2 = buffer2dDesc.CalculateOffset(
                            make_tuple(otherDimInd, thread_local_id + indOffset));

                        compType currVal1 = type_convert<compType>{}(block_buffer[offset1]);
                        compType currVal2 = type_convert<compType>{}(block_buffer[offset2]);
                        int currIndex1    = block_indices_buffer[offset1];
                        int currIndex2    = block_indices_buffer[offset2];

                        binop::calculate(currVal1, currVal2, currIndex1, currIndex2);
                        block_buffer(offset1)         = type_convert<compType>{}(currVal1);
                        block_indices_buffer(offset1) = currIndex1;
                    }
                    __syncthreads();
                }
            }

            if(thread_local_id == 0)
            {
                for(index_t otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
                {
                    index_t offset = buffer2dDesc.CalculateOffset(make_tuple(otherDimInd, 0));

                    compType tmpVal = type_convert<compType>{}(block_buffer[offset]);
                    int tmpIndex    = block_indices_buffer[offset];

                    binop::calculate(lAccuData, tmpVal, lAccuIndex, tmpIndex);
                }

                binop::calculate(accuData, lAccuData, accuIndex, lAccuIndex);
            }
        }
        else
        {
            index_t offset;

            for(index_t otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
            {
                offset = buffer2dDesc.CalculateOffset(make_tuple(thread_local_id, otherDimInd));
                compType currVal = type_convert<compType>{}(block_buffer[offset]);
                int currIndex    = block_indices_buffer[offset];

                binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
            }

            offset = buffer2dDesc.CalculateOffset(make_tuple(thread_local_id, 0));

            block_buffer(offset)         = lAccuData;
            block_indices_buffer(offset) = lAccuIndex;

            __syncthreads();

            for(index_t indOffset = 1; indOffset < BlockSize; indOffset *= 2)
            {
                if(thread_local_id % (indOffset * 2) == 0)
                {
                    index_t offset1 = buffer2dDesc.CalculateOffset(make_tuple(thread_local_id, 0));
                    index_t offset2 =
                        buffer2dDesc.CalculateOffset(make_tuple(thread_local_id + indOffset, 0));

                    compType currVal1 = type_convert<compType>{}(block_buffer[offset1]);
                    compType currVal2 = type_convert<compType>{}(block_buffer[offset2]);
                    int currIndex1    = block_indices_buffer[offset1];
                    int currIndex2    = block_indices_buffer[offset2];

                    binop::calculate(currVal1, currVal2, currIndex1, currIndex2);
                    block_buffer(offset1)         = type_convert<compType>{}(currVal1);
                    block_indices_buffer(offset1) = currIndex1;
                }

                __syncthreads();
            }

            if(thread_local_id == 0)
            {
                compType tmpVal = type_convert<compType>{}(block_buffer[0]);
                int tmpIndex    = block_indices_buffer[0];

                binop::calculate(accuData, tmpVal, accuIndex, tmpIndex);
            }
        }
    };

    template <typename BufferType>
    __device__ static void set_buffer_value(BufferType& block_buffer, compType value)
    {
        index_t thread_id = get_thread_local_1d_id();

        for(index_t otherDimInd = 0; otherDimInd < NumBlocks; otherDimInd++)
        {
            index_t offset = blockIsOneRow
                                 ? buffer2dDesc.CalculateOffset(make_tuple(otherDimInd, thread_id))
                                 : buffer2dDesc.CalculateOffset(make_tuple(thread_id, otherDimInd));

            block_buffer(offset) = value;

            __syncthreads();
        }
    };

    // Initialize the block-wise indices buffer, the index for each element in the block-wise data
    // buffer
    // is calculated according to its position in the buffer and the global starting index
    template <typename IdxBufferType>
    __device__ static void init_buffer_indices(IdxBufferType& block_indices_buffer, int indexStart)
    {
        index_t thread_id = get_thread_local_1d_id();

        for(index_t otherDimInd = 0; otherDimInd < NumBlocks; otherDimInd++)
        {
            index_t offset = blockIsOneRow
                                 ? buffer2dDesc.CalculateOffset(make_tuple(otherDimInd, thread_id))
                                 : buffer2dDesc.CalculateOffset(make_tuple(thread_id, otherDimInd));

            block_indices_buffer(offset) = offset + indexStart;

            __syncthreads();
        }
    };

    // Execute unary operation on the block buffer elements
    template <typename unary_op_type, typename BufferType>
    __device__ static void operate_on_elements(unary_op_type& unary_op, BufferType& block_buffer)
    {
        index_t thread_id = get_thread_local_1d_id();

        for(index_t otherDimInd = 0; otherDimInd < NumBlocks; otherDimInd++)
        {
            index_t offset = blockIsOneRow
                                 ? buffer2dDesc.CalculateOffset(make_tuple(otherDimInd, thread_id))
                                 : buffer2dDesc.CalculateOffset(make_tuple(thread_id, otherDimInd));

            block_buffer(offset) = unary_op(block_buffer[offset]);

            __syncthreads();
        }
    };
};

}; // end of namespace ck

#endif
