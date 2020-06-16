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
#ifndef GUARD_MIOPEN_REDUCE_COMMON_HPP
#define GUARD_MIOPEN_REDUCE_COMMON_HPP

#include <half.hpp>
#include <limits>
#include <cmath>
#include <miopen/miopen.h>
#include <miopen/float_equal.hpp>

#include "bfloat16.hpp"

namespace reduce {

enum ReductionMethod_t
{
    Reduce_DirectThreadWise = 1,
    Reduce_DirectWarpWise   = 2,
    Reduce_BlockWise        = 3,
    Reduce_MultiBlock       = 4
};

// data type conversion
template <typename T>
struct type_convert
{
    template <typename X>
    T operator()(X x) const
    {
        return static_cast<T>(x);
    }
};

template <>
template <>
inline float type_convert<float>::operator()<half_float::half>(half_float::half x) const
{
    return half_float::half_cast<float>(x);
};

template <>
template <>
inline half_float::half type_convert<half_float::half>::operator()<float>(float x) const
{
    return half_float::half_cast<half_float::half>(x);
};

template <>
template <>
inline float type_convert<float>::operator()<bfloat16>(bfloat16 x) const
{
    return float(x);
};

template <>
template <>
inline bfloat16 type_convert<bfloat16>::operator()<float>(float x) const
{
    return bfloat16(x);
};

template <typename compType>
static inline std::function<compType(compType, compType)> ReduceOpFn(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD: return ([&](compType a_, compType b_) { return a_ + b_; });

    case MIOPEN_REDUCE_TENSOR_MUL: return ([&](compType a_, compType b_) { return a_ * b_; });

    case MIOPEN_REDUCE_TENSOR_MIN:
        return ([&](compType a_, compType b_) {
            return (a_ > b_) ? b_ : a_;
        }); // a is selected when they are equal

    case MIOPEN_REDUCE_TENSOR_MAX:
        return ([&](compType a_, compType b_) {
            return (a_ < b_) ? b_ : a_;
        }); // a is selected when they are equal
    }

    return (std::function<compType(compType, compType)>{});
};

template <typename compType>
static inline compType ReduceOpZeroVal(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD: return (type_convert<compType>{}(0.0));

    case MIOPEN_REDUCE_TENSOR_MUL: return (type_convert<compType>{}(1.0));

    case MIOPEN_REDUCE_TENSOR_MIN: return (std::numeric_limits<compType>::max());

    case MIOPEN_REDUCE_TENSOR_MAX: return (std::numeric_limits<compType>::min());
    }

    return (type_convert<compType>{}(0.0));
};

template <>
inline half_float::half ReduceOpZeroVal<half_float::half>(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD: return (type_convert<half_float::half>{}(0.0));

    case MIOPEN_REDUCE_TENSOR_MUL: return (type_convert<half_float::half>{}(1.0));

    case MIOPEN_REDUCE_TENSOR_MIN:
        return (type_convert<half_float::half>{}(std::numeric_limits<float>::max()));

    case MIOPEN_REDUCE_TENSOR_MAX:
        return (type_convert<half_float::half>{}(std::numeric_limits<float>::min()));
    }

    return (type_convert<half_float::half>{}(0.0));
};

template <typename T>
static inline bool IsNan(T x)
{
    // C++ isnan() is used for float and double
    return (std::isnan(x));
};

template <>
inline bool IsNan<half_float::half>(half_float::half x)
{
    return (half_float::isnan(x));
};

template <typename T>
static inline bool IsFinite(T x)
{
    // C++ isfinite() is used for float and double
    return (std::isfinite(x));
};

template <>
inline bool IsFinite<half_float::half>(half_float::half x)
{
    return (half_float::isfinite(x));
};

struct float_equal_one
{
    template <class T>
    static bool apply(T x)
    {
        return std::isfinite(x) and
               std::nextafter(x, std::numeric_limits<T>::lowest()) <= static_cast<T>(1.0) and
               std::nextafter(x, std::numeric_limits<T>::max()) >= static_cast<T>(1.0);
    }

    template <class T>
    bool operator()(T x)
    {
        return (float_equal_one::apply(x));
    };
};

struct float_equal_zero
{
    template <class T>
    static bool apply(T x)
    {
        return std::isfinite(x) and
               std::nextafter(x, std::numeric_limits<T>::lowest()) <= static_cast<T>(0.0) and
               std::nextafter(x, std::numeric_limits<T>::max()) >= static_cast<T>(0.0);
    }

    template <class T>
    bool operator()(T x)
    {
        return (float_equal_zero::apply(x));
    };
};

template <>
inline bool float_equal_one::apply<half_float::half>(half_float::half x)
{
    return half_float::isfinite(x) and x <= type_convert<half_float::half>{}(1.0) and
           x >= type_convert<half_float::half>{}(1.0);
};

template <>
inline bool float_equal_zero::apply<half_float::half>(half_float::half x)
{
    return half_float::isfinite(x) and x <= type_convert<half_float::half>{}(0.0) and
           x >= type_convert<half_float::half>{}(0.0);
};

template <typename compType>
static inline void binop_with_nan_check(miopenNanPropagation_t nanOpt, std::function<compType(compType, compType)> opReduce, compType& accuVal, compType currVal)
{
    if(nanOpt == MIOPEN_NOT_PROPAGATE_NAN)                   
       accuVal = opReduce(accuVal, currVal);                
    else                                                     
    {                                                        
        if(reduce::IsNan(currVal))                           
           accuVal = currVal;                               
        else                                                 
           accuVal = opReduce(accuVal, currVal);            
    };                                                       
}; 

template <typename compType>
static inline void binop_with_nan_check2(miopenNanPropagation_t nanOpt, std::function<compType(compType, compType)> opReduce, compType& accuVal, compType currVal, int& accuIndex, int currIndex)
{
    if(nanOpt == MIOPEN_NOT_PROPAGATE_NAN)                                       
    {                                                                               
       auto accuVal_new = opReduce(accuVal, currVal);                              
       if(!miopen::float_equal(accuVal, accuVal_new))                              
       {                                                                          
           accuIndex = currIndex;                                                 
           accuVal   = accuVal_new;                                              
       };                                                                       
    }                                                                          
    else                                                                      
    {                                                                        
       compType accuVal_new = reduce::IsNan(currVal)? currVal : opReduce(accuVal, currVal);                     
                                                                       
       if(!miopen::float_equal(accuVal, accuVal_new))                 
       {                                                             
          accuIndex = currIndex;                                    
          accuVal   = accuVal_new;                                 
       };                                                         
    };                                                         
}; 

}; // end of namespace reduce

#endif
