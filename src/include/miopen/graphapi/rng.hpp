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
#include <miopen/graphapi/graphapi.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/tensor.hpp>

#include <cstdint>
#include <variant>

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

class MIOPEN_INTERNALS_EXPORT RngBuilder
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

class MIOPEN_INTERNALS_EXPORT BackendRngDescriptor : public BackendDescriptor
{
private:
    RngBuilder mBuilder;
    Rng mRng;

public:
    virtual void setAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              void* arrayOfElements) override;
    virtual void finalize() override;
    virtual void getAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) override;

    const Rng* getRng() const noexcept { return &mRng; }
    Rng* getRng() noexcept { return &mRng; }
};

class MIOPEN_INTERNALS_EXPORT OperationRng : public OpNode
{
private:
    Rng* mRng                            = nullptr;
    Tensor* mOutput                      = nullptr;
    std::variant<int64_t, Tensor*> mSeed = 0; // Don't change the order of variant alternatives
    Tensor* mOffset                      = nullptr;

    friend class OperationRngBuilder;

public:
    OperationRng() noexcept = default;
    OperationRng(Rng* rng, Tensor* output, int64_t seed, Tensor* offset) noexcept
        : mRng(rng), mOutput(output), mSeed(seed), mOffset(offset)
    {
    }
    OperationRng(Rng* rng, Tensor* output, Tensor* seed, Tensor* offset) noexcept
        : mRng(rng), mOutput(output), mSeed(seed), mOffset(offset)
    {
    }

    Rng* getRng() const noexcept { return mRng; }
    Tensor* getOutput() const noexcept { return mOutput; }
    std::variant<int64_t, Tensor*> getSeed() const noexcept { return mSeed; }
    Tensor* getOffset() const noexcept { return mOffset; }

    virtual const std::string& signName() const override;
    virtual std::vector<Tensor*> getInTensors() const override;
    virtual std::vector<Tensor*> getOutTensors() const override;
};

class MIOPEN_INTERNALS_EXPORT OperationRngBuilder
{
private:
    OperationRng mOperationRng;

public:
    OperationRngBuilder& setRng(Rng* rng);
    OperationRngBuilder& setOutput(Tensor* output);
    OperationRngBuilder& setSeed(int64_t seed) noexcept;
    OperationRngBuilder& setSeed(Tensor* seed);
    OperationRngBuilder& setOffset(Tensor* offset);

    OperationRng build();
};

class MIOPEN_INTERNALS_EXPORT BackendOperationRngDescriptor : public BackendDescriptor
{
private:
    OperationRngBuilder mBuilder;
    OperationRng mOperationRng;
    miopenBackendDescriptor_t mRngDescriptor    = nullptr;
    miopenBackendDescriptor_t mOutputDescriptor = nullptr; // sometimes called Y
    miopenBackendDescriptor_t mSeedDescriptor   = nullptr;
    miopenBackendDescriptor_t mOffsetDescriptor = nullptr;

public:
    void setAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      void* arrayOfElements) override;
    void finalize() override;
    void getAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) override;

    OpNode* getOperation() override;

    const OperationRng* getRng() const { return &mOperationRng; }
    OperationRng* getRng() { return &mOperationRng; }
};

} // namespace graphapi

} // namespace miopen
