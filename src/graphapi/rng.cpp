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

#include <miopen/algorithm.hpp>
#include <miopen/errors.hpp>
#include <miopen/graphapi/rng.hpp>

#include <cstdint>

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

void BackendRngDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                        miopenBackendAttributeType_t attributeType,
                                        int64_t elementCount,
                                        void* arrayOfElements)
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    using Setter = RngBuilder& (RngBuilder::*)(double);

    auto callSetter = [=](Setter setter) {
        if(attributeType == MIOPEN_TYPE_DOUBLE && elementCount == 1)
        {
            (mBuilder.*setter)(*static_cast<double*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
    };

    switch(attributeName)
    {
    case MIOPEN_ATTR_RNG_DISTRIBUTION:
        if(attributeType == MIOPEN_TYPE_RNG_DISTRIBUTION && elementCount == 1)
        {
            mBuilder.setDistribution(*static_cast<miopenRngDistribution_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_RNG_NORMAL_DIST_MEAN: callSetter(&RngBuilder::setNormalMean); break;

    case MIOPEN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION:
        callSetter(&RngBuilder::setNormalStdev);
        break;

    case MIOPEN_ATTR_RNG_UNIFORM_DIST_MINIMUM: callSetter(&RngBuilder::setUniformMin); break;

    case MIOPEN_ATTR_RNG_UNIFORM_DIST_MAXIMUM: callSetter(&RngBuilder::setUniformMax); break;

    case MIOPEN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY:
        callSetter(&RngBuilder::setBernoulliProb);
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendRngDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mRng       = mBuilder.build();
    mFinalized = true;
}

void BackendRngDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
                                        miopenBackendAttributeType_t attributeType,
                                        int64_t requestedElementCount,
                                        int64_t* elementCount,
                                        void* arrayOfElements)
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    using Getter = double (Rng::*)() const;

    auto callGetter = [=](Getter getter) {
        if(attributeType == MIOPEN_TYPE_DOUBLE && requestedElementCount == 1)
        {
            *elementCount                          = 1;
            *static_cast<double*>(arrayOfElements) = (mRng.*getter)();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
    };

    switch(attributeName)
    {
    case MIOPEN_ATTR_RNG_DISTRIBUTION:
        if(attributeType == MIOPEN_TYPE_RNG_DISTRIBUTION && requestedElementCount == 1)
        {
            *elementCount                                           = 1;
            *static_cast<miopenRngDistribution_t*>(arrayOfElements) = mRng.getDistribution();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_RNG_NORMAL_DIST_MEAN: callGetter(&Rng::getNormalMean); break;

    case MIOPEN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION: callGetter(&Rng::getNormalStdev); break;

    case MIOPEN_ATTR_RNG_UNIFORM_DIST_MINIMUM: callGetter(&Rng::getUniformMin); break;

    case MIOPEN_ATTR_RNG_UNIFORM_DIST_MAXIMUM: callGetter(&Rng::getUniformMax); break;

    case MIOPEN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY: callGetter(&Rng::getBernoulliProb); break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

const std::string& OperationRng::signName() const
{
    static const std::string name = "OP_RNG";
    return name;
}

std::vector<Tensor*> OperationRng::getInTensors() const
{
    if(mSeed.index() == 0)
    {
        return {mOffset};
    }
    else
    {
        return {std::get<Tensor*>(mSeed), mOffset};
    }
}

std::vector<Tensor*> OperationRng::getOutTensors() const { return {mOutput}; }

OperationRngBuilder& OperationRngBuilder::setRng(Rng* rng)
{
    mOperationRng.mRng = checkPtr(rng);
    return *this;
}

OperationRngBuilder& OperationRngBuilder::setOutput(Tensor* output)
{
    mOperationRng.mOutput = checkPtr(output);
    return *this;
}

OperationRngBuilder& OperationRngBuilder::setSeed(int64_t seed) noexcept
{
    mOperationRng.mSeed = seed;
    return *this;
}

OperationRngBuilder& OperationRngBuilder::setSeed(Tensor* seed)
{
    bool valid = seed != nullptr;

    valid = valid && miopen::all_of(seed->GetLengths(), [](auto v) { return v == 1; }) &&
            miopen::all_of(seed->GetStrides(), [](auto v) { return v == 1; });

    if(valid)
    {
        mOperationRng.mSeed = seed;
        return *this;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

OperationRngBuilder& OperationRngBuilder::setOffset(Tensor* offset)
{
    bool valid = offset != nullptr;

    valid = valid && miopen::all_of(offset->GetLengths(), [](auto v) { return v == 1; }) &&
            miopen::all_of(offset->GetStrides(), [](auto v) { return v == 1; });

    if(valid)
    {
        mOperationRng.mOffset = offset;
        return *this;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

OperationRng OperationRngBuilder::build()
{
    if(mOperationRng.mRng == nullptr || mOperationRng.mOutput == nullptr ||
       mOperationRng.mOffset == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    return mOperationRng;
}

void BackendOperationRngDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                                 miopenBackendAttributeType_t attributeType,
                                                 int64_t elementCount,
                                                 void* arrayOfElements)
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    using TensorSetter = OperationRngBuilder& (OperationRngBuilder::*)(Tensor*);

    auto callTensorSetter = [=](TensorSetter setter, miopenBackendDescriptor_t& outApiDescriptor) {
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            miopenBackendDescriptor_t apiDescriptor =
                deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
            BackendDescriptor& backendDescriptor = deref(apiDescriptor);

            if(!backendDescriptor.isFinalized())
            {
                MIOPEN_THROW(miopenStatusBadParm);
            }

            BackendTensorDescriptor& tensorDescriptor =
                dynamic_cast<BackendTensorDescriptor&>(backendDescriptor);
            (mBuilder.*setter)(tensorDescriptor.getTensor());
            outApiDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
    };

    switch(attributeName)
    {
    case MIOPEN_ATTR_OPERATION_RNG_DESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            miopenBackendDescriptor_t apiDescriptor =
                deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
            BackendDescriptor& backendDescriptor = deref(apiDescriptor);

            if(!backendDescriptor.isFinalized())
            {
                MIOPEN_THROW(miopenStatusBadParm);
            }

            BackendRngDescriptor& rngDescriptor =
                dynamic_cast<BackendRngDescriptor&>(backendDescriptor);
            mBuilder.setRng(rngDescriptor.getRng());
            mRngDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_RNG_YDESC:
        callTensorSetter(&OperationRngBuilder::setOutput, mOutputDescriptor);
        break;

    case MIOPEN_ATTR_OPERATION_RNG_SEED:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            mBuilder.setSeed(*static_cast<int64_t*>(arrayOfElements));
        }
        else
        {
            callTensorSetter(&OperationRngBuilder::setSeed, mSeedDescriptor);
        }
        break;

    case MIOPEN_ATTR_OPERATION_RNG_OFFSET_DESC:
        callTensorSetter(&OperationRngBuilder::setOffset, mOffsetDescriptor);
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationRngDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mOperationRng = mBuilder.build();
    mFinalized    = true;
}

void BackendOperationRngDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
                                                 miopenBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements)
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    auto retrieveDescriptor = [=](miopenBackendDescriptor_t source) {
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = source;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
    };

    switch(attributeName)
    {
    case MIOPEN_ATTR_OPERATION_RNG_DESC: retrieveDescriptor(mRngDescriptor); break;

    case MIOPEN_ATTR_OPERATION_RNG_YDESC: retrieveDescriptor(mOutputDescriptor); break;

    case MIOPEN_ATTR_OPERATION_RNG_SEED:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int64_t*>(arrayOfElements) = std::get<int64_t>(mOperationRng.getSeed());
        }
        else if(mOperationRng.getSeed().index() == 1)
        {
            retrieveDescriptor(mSeedDescriptor);
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_RNG_OFFSET_DESC: retrieveDescriptor(mOffsetDescriptor); break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

OpNode* BackendOperationRngDescriptor::getOperation() { return &mOperationRng; }

} // namespace graphapi

} // namespace miopen
