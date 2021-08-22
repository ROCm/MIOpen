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
#ifndef CK_REDUCTION_FUNCTIONS_BINOP_HPP
#define CK_REDUCTION_FUNCTIONS_BINOP_HPP

#include "data_type.hpp"

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
    calculate(compType& accuVal, compType currVal, int& accuIndex, int currIndex)
    {
        if(isnan(currVal))
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
}; // end of namespace ck

#endif
