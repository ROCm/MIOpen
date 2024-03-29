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

#include <miopen/errors.hpp>
#include <miopen/graphapi/rng.hpp>

namespace miopen {

namespace graphapi {

RngBuilder& RngBuilder::setNormalStdev(double normalStdev)
{
    if(normalStdev > 0)
    {
        mRng.mNormalStdev = normalStdev;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return *this;
}

RngBuilder& RngBuilder::setBernoulliProb(double bernoulliProb)
{
    if(bernoulliProb >= 0.0 && bernoulliProb <= 1.0)
    {
        mRng.mBernoulliProb = bernoulliProb;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return *this;
}

Rng RngBuilder::build() const
{
    if((mRng.mDistribution == MIOPEN_RNG_DISTRIBUTION_NORMAL && mRng.mNormalStdev > 0.0) ||
       (mRng.mDistribution == MIOPEN_RNG_DISTRIBUTION_UNIFORM &&
        mRng.mUniformMin <= mRng.mUniformMax) ||
       (mRng.mDistribution == MIOPEN_RNG_DISTRIBUTION_BERNOULLI && mRng.mBernoulliProb >= 0.0 &&
        mRng.mBernoulliProb <= 1.0))
    {
        return mRng;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace graphapi

} // namespace miopen
