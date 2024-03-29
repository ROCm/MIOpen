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
#pragma once

#include <miopen/miopen.h>

namespace miopen {

namespace graphapi {

class Rng
{
private:
    miopenRngDistribution_t mDistribution = MIOPEN_RNG_DISTRIBUTION_BERNOULLI;
    double mNormalMean                    = -1.0;
    double mNormalStdev                   = -1.0;
    double mUniformMin                    = -1.0;
    double mUniformMax                    = -1.0;
    double mBernoulliProb                 = -1.0;

public:
    Rng() = default;
    Rng(miopenRngDistribution_t distribution,
        double normalMean,
        double normalStdev,
        double uniformMin,
        double uniformMax,
        double bernoulliProb)
        : mDistribution(distribution),
          mNormalMean(normalMean),
          mNormalStdev(normalStdev),
          mUniformMin(uniformMin),
          mUniformMax(uniformMax),
          mBernoulliProb(bernoulliProb)
    {
    }

    miopenRngDistribution_t getDistribution() const noexcept { return mDistribution; }
    double getNormalMean() const noexcept { return mNormalMean; }
    double getNormalStdev() const noexcept { return mNormalStdev; }
    double getUniformMin() const noexcept { return mUniformMin; }
    double getUniformMax() const noexcept { return mUniformMax; }
    double getBernoulliProb() const noexcept { return mBernoulliProb; }

private:
    friend class RngBuilder;
};

class RngBuilder
{
private:
    Rng mRng;

public:
    RngBuilder& setDistribution(miopenRngDistribution_t distribution) noexcept
    {
        mRng.mDistribution = distribution;
        return *this;
    }
    RngBuilder& setNormalMean(double normalMean) noexcept
    {
        mRng.mNormalMean = normalMean;
        return *this;
    }

    RngBuilder& setNormalStdev(double normalStdev);

    RngBuilder& setUniformMin(double uniformMin) noexcept
    {
        mRng.mUniformMin = uniformMin;
        return *this;
    }
    RngBuilder& setUniformMax(double uniformMax) noexcept
    {
        mRng.mUniformMax = uniformMax;
        return *this;
    }

    RngBuilder& setBernoulliProb(double bernoulliProb);

    Rng build() const;
};

} // namespace graphapi

} // namespace miopen
