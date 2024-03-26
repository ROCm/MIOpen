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

#include <miopen/errors.hpp>
#include <miopen/graphapi/graphapi.hpp>

#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace miopen {

namespace graphapi {

class Pointwise
{
public:
    Pointwise() noexcept = default;
    Pointwise(miopenPointwiseMode_t mode,
              miopenDataType_t mathPrecision,
              miopenNanPropagation_t nanPropagation = MIOPEN_NOT_PROPAGATE_NAN,
              double reluLowerClip                  = 0.0,
              double reluUpperClip                  = std::numeric_limits<double>::max(),
              double reluLowerClipSlope             = 0.0,
              double eluAlpha                       = 1.0,
              double softPlusBeta                   = 1.0,
              int64_t axis                          = -1)
        : mReluLowerClip(reluLowerClip),
          mReluUpperClip(reluUpperClip),
          mReluLowerClipSlope(reluLowerClipSlope),
          mEluAlpha(eluAlpha),
          mSoftPlusBeta(softPlusBeta),
          mAxis(axis),
          mMode(mode),
          mMathPrecision(mathPrecision),
          mNanPropagation(nanPropagation)
    {
    }

    miopenPointwiseMode_t getMode() const noexcept { return mMode; }
    miopenDataType_t getMathPrecision() const noexcept { return mMathPrecision; }
    miopenNanPropagation_t getNanPropagation() const noexcept { return mNanPropagation; }
    double getReluLowerClip() const noexcept { return mReluLowerClip; }
    double getReluUpperClip() const noexcept { return mReluUpperClip; }
    double getReluLowerClipSlope() const noexcept { return mReluLowerClipSlope; }
    double getEluAlpha() const noexcept { return mEluAlpha; }
    double getSoftPlusBeta() const noexcept { return mSoftPlusBeta; }
    int64_t getAxis() const noexcept { return mAxis; }

private:
    friend class PointwiseBuilder;

    double mReluLowerClip      = 0.0;
    double mReluUpperClip      = std::numeric_limits<double>::max();
    double mReluLowerClipSlope = 0.0;
    double mEluAlpha           = 1.0;
    double mSoftPlusBeta       = 1.0;
    int64_t mAxis              = -1;
    miopenPointwiseMode_t mMode;
    miopenDataType_t mMathPrecision;
    miopenNanPropagation_t mNanPropagation = MIOPEN_NOT_PROPAGATE_NAN;
};

class PointwiseBuilder
{
public:
    PointwiseBuilder& setMode(miopenPointwiseMode_t mode) noexcept
    {
        mPointwise.mMode = mode;
        mModeSet         = true;
        return *this;
    }
    PointwiseBuilder& setMathPrecision(miopenDataType_t mathPrecision) noexcept
    {
        mPointwise.mMathPrecision = mathPrecision;
        mMathPrecisionSet         = true;
        return *this;
    }
    PointwiseBuilder& setNanPropagation(miopenNanPropagation_t nanPropagation) noexcept
    {
        mPointwise.mNanPropagation = nanPropagation;
        return *this;
    }
    PointwiseBuilder& setReluLowerClip(double reluLowerClip) noexcept
    {
        mPointwise.mReluLowerClip = reluLowerClip;
        return *this;
    }
    PointwiseBuilder& setReluUpperClip(double reluUpperClip) noexcept
    {
        mPointwise.mReluUpperClip = reluUpperClip;
        return *this;
    }
    PointwiseBuilder& setReluLowerClipSlope(double reluLowerClipSlope) noexcept
    {
        mPointwise.mReluLowerClipSlope = reluLowerClipSlope;
        return *this;
    }
    PointwiseBuilder& setEluAlpha(double eluAlpha) noexcept
    {
        mPointwise.mEluAlpha = eluAlpha;
        return *this;
    }
    PointwiseBuilder& setSoftPlusBeta(double softPlusBeta) noexcept
    {
        mPointwise.mSoftPlusBeta = softPlusBeta;
        return *this;
    }
    PointwiseBuilder& setAxis(int64_t axis) noexcept
    {
        mPointwise.mAxis = axis;
        return *this;
    }

    Pointwise build() const
    {
        if(!mModeSet || !mMathPrecisionSet)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        return mPointwise;
    }

private:
    Pointwise mPointwise;
    bool mModeSet          = false;
    bool mMathPrecisionSet = false;
};

class BackendPointwiseDescriptor : public BackendDescriptor
{
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

private:
    PointwiseBuilder mBuilder;
    Pointwise mPointwise;

    using Setter = std::function<PointwiseBuilder&(PointwiseBuilder&, double value)>;

    void setFloatOrDouble(Setter setter,
                          miopenBackendAttributeType_t attributeType,
                          int64_t elementCount,
                          void* arrayOfElements)
    {
        if(attributeType == MIOPEN_TYPE_FLOAT && elementCount == 1)
        {
            std::invoke(setter, mBuilder, *static_cast<float*>(arrayOfElements));
        }
        else if(attributeType == MIOPEN_TYPE_DOUBLE && elementCount == 1)
        {
            std::invoke(setter, mBuilder, *static_cast<double*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
    }

    using Getter = std::function<double(const Pointwise&)>;

    void getFloatOrDouble(Getter getter,
                          miopenBackendAttributeType_t attributeType,
                          int64_t requestedElementCount,
                          int64_t* elementCount,
                          void* arrayOfElements)
    {
        if(attributeType == MIOPEN_TYPE_FLOAT && requestedElementCount == 1)
        {
            *elementCount                         = 1;
            *static_cast<float*>(arrayOfElements) = std::invoke(getter, mPointwise);
        }
        else if(attributeType == MIOPEN_TYPE_DOUBLE && requestedElementCount == 1)
        {
            *elementCount                          = 1;
            *static_cast<double*>(arrayOfElements) = std::invoke(getter, mPointwise);
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
    }
};

} // namespace graphapi

} // namespace miopen
