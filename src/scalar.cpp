/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <miopen/scalar.hpp>
#include <miopen/conv/problem_description.hpp>

template <typename T>
bool isCloseToZero(T value)
{
    return std::abs(value) <= std::numeric_limits<T>::epsilon();
}

template <typename T>
bool isCloseToOne(float value)
{
    return std::abs(value - T(1)) <= std::numeric_limits<T>::epsilon();
}

namespace miopen {
namespace conv {

AlphaBetaCase GetAlphaBetaCase(const Scalar& alpha, const Scalar& beta)
{
    // default T as double since we are comparing
    // numerical values just to find enum type.
    using T = double;

    T alpha_val = alpha.GetAsDouble();
    T beta_val  = beta.GetAsDouble();

    bool alpha_one  = isCloseToOne<T>(alpha_val);
    bool beta_zero  = isCloseToZero<T>(beta_val);
    bool alpha_zero = isCloseToZero<T>(alpha_val);

    if(alpha_one && beta_zero)
    {
        return AlphaBetaCase::IDENTITY;
    }

    if((!alpha_one && !alpha_zero) && !beta_zero)
    {
        return AlphaBetaCase::BILINEAR;
    }

    if(!alpha_zero && beta_zero)
    {
        return AlphaBetaCase::SCALE;
    }

    return AlphaBetaCase::ERROR_STATE;
}
} // namespace conv
} // namespace miopen
