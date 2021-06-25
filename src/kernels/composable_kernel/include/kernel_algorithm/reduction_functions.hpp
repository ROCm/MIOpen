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

#include "float_type.hpp"

#include "reduction_common.hpp"
#include "reduction_operator.hpp"

namespace ck {
namespace detail {

static inline __device__ bool isnan(half_t x) { return __hisnan(x); };

template <NanPropagation_t nanPropaOpt, typename opReduce, typename compType>
struct binop_with_nan_check;

// ToDo: remove the "const" from the "accuVal" parameter declaration, this is added
//       to avoid the "constParameter" warning during tidy checking
template <typename opReduce, typename compType>
struct binop_with_nan_check<NanPropagation_t::NOT_PROPAGATE_NAN, opReduce, compType>
{
    __device__ static inline void calculate(const compType& accuVal, compType currVal)
    {
        opReduce{}(const_cast<compType&>(accuVal), currVal);
    };

    // The method is called when the opReduce is indexable and the user asked for indices
    __device__ static inline void calculate(const compType& accuVal,
                                            compType currVal,
                                            VOLATILE_WA_274384 int& accuIndex,
                                            int currIndex)
    {
        VOLATILE_WA_274384 bool changed = false;

        opReduce{}(const_cast<compType&>(accuVal), currVal, changed);

        if(changed)
            accuIndex = currIndex;
    };
};

template <typename opReduce, typename compType>
struct binop_with_nan_check<NanPropagation_t::PROPAGATE_NAN, opReduce, compType>
{
    __device__ static inline void calculate(compType& accuVal, compType currVal)
    {
        if(isnan(currVal))
            accuVal = currVal;
        else
            opReduce{}(accuVal, currVal);
    };

    // The method is called when the opReduce is indexable and the user asked for indices
    __device__ static inline void
    calculate(compType& accuVal, compType currVal, VOLATILE_WA_274384 int& accuIndex, int currIndex)
    {
        if(isnan(currVal))
        {
            accuVal   = currVal;
            accuIndex = currIndex;
        }
        else
        {
            VOLATILE_WA_274384 bool changed = false;

            opReduce{}(accuVal, currVal, changed);

            if(changed)
                accuIndex = currIndex;
        }
    };
};
}; // namespace detail

template <typename DataType,
          index_t ThreadBufferLen,
          typename opReduce,
          NanPropagation_t nanPropaOpt>
struct ThreadReduce
{
    using compType = typename opReduce::dataType;
    using binop    = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    // This interface does not accumulate on indices
    __device__ static void Reduce(const DataType* p_thread_buffer, compType& accuData)
    {
        for(index_t i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            binop::calculate(accuData, currVal);
        }
    };

    // This interface accumulates on both data values and indices and
    // is called by Direct_ThreadWise reduction method at first-time reduction
    __device__ static void
    Reduce2(const DataType* p_thread_buffer, compType& accuData, int& accuIndex, int indexStart)
    {
        for(index_t i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            int currIndex    = i + indexStart;
            binop::calculate(accuData, currVal, accuIndex, currIndex);
        }
    };

    // This interface accumulates on both data values and indices and
    // is called by Direct_ThreadWise reduction method at second-time reduction
    __device__ static void Reduce3(const DataType* p_thread_buffer,
                                   const int* thread_indices_buffer,
                                   compType& accuData,
                                   int& accuIndex)
    {
        for(index_t i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            int currIndex    = thread_indices_buffer[i];
            binop::calculate(accuData, currVal, accuIndex, currIndex);
        }
    };

    // Set the elements in the per-thread buffer to a specific value
    __device__ static void set_buffer_value(DataType* p_thread_buffer, DataType value)
    {
        for(index_t i          = 0; i < ThreadBufferLen; i++)
            p_thread_buffer[i] = value;
    };

    // Execute unary operation on the per-thread buffer elements
    template <typename unary_op>
    __device__ static void operate_on_elements(DataType* p_thread_buffer)
    {
        for(index_t i = 0; i < ThreadBufferLen; i++)
            unary_op{}(p_thread_buffer[i]);
    };
};

template <typename DataType,
          index_t BlockSize,
          index_t ThreadBufferLen,
          typename opReduce,
          NanPropagation_t nanPropaOpt>
struct WarpReduce
{
    using compType = typename opReduce::dataType;
    using binop    = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;
    constexpr static bool have_builtin_shuffle =
        std::is_same<compType, float>::value || std::is_same<compType, double>::value;

    // This interface does not accumulate on indices
    __device__ static void Reduce(const DataType* p_thread_buffer, compType& accuData)
    {
        static_if<have_builtin_shuffle>{}([&](auto) {
            ReduceImpl1(p_thread_buffer, accuData);
        }).Else([&](auto) { ReduceImpl2(p_thread_buffer, accuData); });
    };

    // This interface implementation uses HIP built-in device shuffling functions
    __device__ static void ReduceImpl1(const DataType* p_thread_buffer, compType& accuData)
    {
        compType lAccuData = opReduce::GetZeroVal();

        for(index_t i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            binop::calculate(lAccuData, currVal);
        }

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
    __device__ static void ReduceImpl2(const DataType* p_thread_buffer, compType& accuData)
    {
        compType lAccuData = opReduce::GetZeroVal();

        for(index_t i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            binop::calculate(lAccuData, currVal);
        }

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

    // This interface accumulates on both data values and indices and
    // is called by Direct_WarpWise reduction method at first-time reduction
    __device__ static void
    Reduce2(const DataType* p_thread_buffer, compType& accuData, int& accuIndex, int indexStart)
    {
        static_if<have_builtin_shuffle>{}([&](auto) {
            Reduce2Impl1(p_thread_buffer, accuData, accuIndex, indexStart);
        }).Else([&](auto) { Reduce2Impl2(p_thread_buffer, accuData, accuIndex, indexStart); });
    };

    // This interface implementation uses HIP built-in device shuffling functions
    __device__ static void Reduce2Impl1(const DataType* p_thread_buffer,
                                        compType& accuData,
                                        int& accuIndex,
                                        int indexStart)
    {
        compType lAccuData       = opReduce::GetZeroVal();
        int lAccuIndex           = 0;
        index_t thread_inwarp_id = get_thread_local_1d_id() % warpSize;

        for(index_t i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            int currIndex    = thread_inwarp_id * ThreadBufferLen + i + indexStart;
            binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
        }

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

    // This interface implementation does not use HIP built-in device shuffling functions
    // since for fp16, built-in shuffling functions is not provided by HIP
    __device__ static void Reduce2Impl2(const DataType* p_thread_buffer,
                                        compType& accuData,
                                        int& accuIndex,
                                        int indexStart)
    {
        compType lAccuData       = opReduce::GetZeroVal();
        int lAccuIndex           = 0;
        index_t thread_id        = get_thread_local_1d_id();
        index_t warpId           = thread_id / warpSize;
        index_t thread_inwarp_id = thread_id % warpSize;

        for(index_t i = 0; i < ThreadBufferLen; i++)
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

    // This interface accumulates on both data values and indices and
    // is called by Direct_WarpWise reduction method at second-time reduction
    __device__ static void Reduce3(const DataType* p_thread_buffer,
                                   const int* thread_indices_buffer,
                                   compType& accuData,
                                   int& accuIndex)
    {
        static_if<have_builtin_shuffle>{}([&](auto) {
            Reduce3Impl1(p_thread_buffer, thread_indices_buffer, accuData, accuIndex);
        }).Else([&](auto) {
            Reduce3Impl2(p_thread_buffer, thread_indices_buffer, accuData, accuIndex);
        });
    };

    // This interface implementation uses HIP built-in device shuffling functions
    __device__ static void Reduce3Impl1(const DataType* p_thread_buffer,
                                        const int* thread_indices_buffer,
                                        compType& accuData,
                                        int& accuIndex)
    {
        compType lAccuData = opReduce::GetZeroVal();
        int lAccuIndex     = 0;

        for(index_t i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            int currIndex    = thread_indices_buffer[i];
            binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
        }

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
    __device__ static void Reduce3Impl2(const DataType* p_thread_buffer,
                                        const int* thread_indices_buffer,
                                        compType& accuData,
                                        int& accuIndex)
    {
        compType lAccuData       = opReduce::GetZeroVal();
        int lAccuIndex           = 0;
        index_t thread_id        = get_thread_local_1d_id();
        index_t warpId           = thread_id / warpSize;
        index_t thread_inwarp_id = thread_id % warpSize;

        for(index_t i = 0; i < ThreadBufferLen; i++)
        {
            compType currVal = type_convert<compType>{}(p_thread_buffer[i]);
            int currIndex    = thread_indices_buffer[i];
            binop::calculate(lAccuData, currVal, lAccuIndex, currIndex);
        }

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

    __device__ static void set_buffer_value(DataType* p_thread_buffer, DataType value)
    {
        for(index_t i          = 0; i < ThreadBufferLen; i++)
            p_thread_buffer[i] = value;

        __all(1);
    };

    // Execute unary operation on the per-thread buffer elements
    template <typename unary_op>
    __device__ static void operate_on_elements(DataType* p_thread_buffer)
    {
        for(index_t i = 0; i < ThreadBufferLen; i++)
            unary_op{}(p_thread_buffer[i]);

        __all(1);
    };
};

template <typename buffer2dDesc,
          typename DataType,
          bool blockIsOneRow,
          typename opReduce,
          NanPropagation_t nanPropaOpt>
struct BlockwiseReduction_2d_block_buffer
{
    using compType = typename opReduce::dataType;
    constexpr static index_t BlockSize =
        blockIsOneRow ? buffer2dDesc::GetLengths()[1] : buffer2dDesc::GetLengths()[0];
    constexpr static index_t NumBlocks =
        blockIsOneRow ? buffer2dDesc::GetLengths()[0] : buffer2dDesc::GetLengths()[1];
    using binop = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;

    // This interface does not accumulate on indices
    __device__ static void
    Reduce(DataType* p_block_buffer, index_t toReduceBlocks, compType& accuData)
    {
        const index_t thread_local_id = get_thread_local_1d_id();
        compType lAccuData            = opReduce::GetZeroVal();

        index_t offset;
        for(index_t otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
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

        for(index_t indOffset = BlockSize / 2; indOffset > 0; indOffset /= 2)
        {
            if(thread_local_id < indOffset)
            {
                index_t offset1 = blockIsOneRow
                                      ? buffer2dDesc::CalculateOffset({0, thread_local_id})
                                      : buffer2dDesc::CalculateOffset({thread_local_id, 0});

                index_t offset2 =
                    blockIsOneRow ? buffer2dDesc::CalculateOffset({0, thread_local_id + indOffset})
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

    // This interface accumulates on both data values and indices
    __device__ static void Reduce2(DataType* p_block_buffer,
                                   int* block_indices_buffer,
                                   index_t toReduceBlocks,
                                   compType& accuData,
                                   int& accuIndex)
    {
        const index_t thread_local_id     = get_thread_local_1d_id();
        compType lAccuData                = opReduce::GetZeroVal();
        VOLATILE_WA_274384 int lAccuIndex = 0;

        static_if<blockIsOneRow>{}([&](auto) {
            for(index_t otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
            {
                for(index_t indOffset = 1; indOffset < BlockSize; indOffset *= 2)
                {
                    if(thread_local_id % (indOffset * 2) == 0)
                    {
                        index_t offset1 =
                            buffer2dDesc::CalculateOffset({otherDimInd, thread_local_id});
                        index_t offset2 = buffer2dDesc::CalculateOffset(
                            {otherDimInd, thread_local_id + indOffset});

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
            }

            if(thread_local_id == 0)
            {
                for(index_t otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
                {
                    index_t offset = buffer2dDesc::CalculateOffset({otherDimInd, 0});

                    compType tmpVal = type_convert<compType>{}(p_block_buffer[offset]);
                    int tmpIndex    = block_indices_buffer[offset];

                    binop::calculate(lAccuData, tmpVal, lAccuIndex, tmpIndex);
                }

                binop::calculate(accuData, lAccuData, accuIndex, lAccuIndex);
            }
        }).Else([&](auto) {
            index_t offset;

            for(index_t otherDimInd = 0; otherDimInd < toReduceBlocks; otherDimInd++)
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

            for(index_t indOffset = 1; indOffset < BlockSize; indOffset *= 2)
            {
                if(thread_local_id % (indOffset * 2) == 0)
                {
                    index_t offset1 = buffer2dDesc::CalculateOffset({thread_local_id, 0});
                    index_t offset2 =
                        buffer2dDesc::CalculateOffset({thread_local_id + indOffset, 0});

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
        index_t thread_id = get_thread_local_1d_id();

        for(index_t otherDimInd = 0; otherDimInd < NumBlocks; otherDimInd++)
        {
            index_t offset = blockIsOneRow
                                 ? buffer2dDesc::CalculateOffset({otherDimInd, thread_id})
                                 : buffer2dDesc::CalculateOffset({thread_id, otherDimInd});

            p_block_buffer[offset] = value;

            __syncthreads();
        }
    };

    // Initialize the block-wise indices buffer, the index for each element in the block-wise data
    // buffer
    // is calculated according to its position in the buffer and the global starting index
    __device__ static void init_buffer_indices(int* block_indices_buffer, int indexStart)
    {
        index_t thread_id = get_thread_local_1d_id();

        for(index_t otherDimInd = 0; otherDimInd < NumBlocks; otherDimInd++)
        {
            index_t offset = blockIsOneRow
                                 ? buffer2dDesc::CalculateOffset({otherDimInd, thread_id})
                                 : buffer2dDesc::CalculateOffset({thread_id, otherDimInd});

            block_indices_buffer[offset] = offset + indexStart;

            __syncthreads();
        }
    };

    // Execute unary operation on the block buffer elements
    template <typename unary_op>
    __device__ static void operate_on_elements(DataType* p_block_buffer)
    {
        index_t thread_id = get_thread_local_1d_id();

        for(index_t otherDimInd = 0; otherDimInd < NumBlocks; otherDimInd++)
        {
            index_t offset = blockIsOneRow
                                 ? buffer2dDesc::CalculateOffset({otherDimInd, thread_id})
                                 : buffer2dDesc::CalculateOffset({thread_id, otherDimInd});

            unary_op{}(p_block_buffer[offset]);

            __syncthreads();
        }
    };
};

}; // end of namespace ck

#endif
