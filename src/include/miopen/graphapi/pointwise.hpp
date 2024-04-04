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

#include <miopen/graphapi/graphapi.hpp>

#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

namespace miopen {

namespace graphapi {

class Pointwise
{
public:
    using FpAttribute = std::variant<float, double>;

private:
    FpAttribute mReluLowerClip      = 0.0f;
    FpAttribute mReluUpperClip      = std::numeric_limits<float>::max();
    FpAttribute mReluLowerClipSlope = 0.0f;
    FpAttribute mEluAlpha           = 1.0f;
    FpAttribute mSoftPlusBeta       = 1.0f;
    int64_t mAxis                   = -1;
    miopenPointwiseMode_t mMode;
    miopenDataType_t mMathPrecision;
    miopenNanPropagation_t mNanPropagation = MIOPEN_NOT_PROPAGATE_NAN;

public:
    Pointwise() noexcept = default;
    Pointwise(miopenPointwiseMode_t mode,
              miopenDataType_t mathPrecision,
              miopenNanPropagation_t nanPropagation = MIOPEN_NOT_PROPAGATE_NAN,
              FpAttribute reluLowerClip             = 0.0f,
              FpAttribute reluUpperClip             = std::numeric_limits<float>::max(),
              FpAttribute reluLowerClipSlope        = 0.0f,
              FpAttribute eluAlpha                  = 1.0f,
              FpAttribute softPlusBeta              = 1.0f,
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
    FpAttribute getReluLowerClip() const noexcept { return mReluLowerClip; }
    FpAttribute getReluUpperClip() const noexcept { return mReluUpperClip; }
    FpAttribute getReluLowerClipSlope() const noexcept { return mReluLowerClipSlope; }
    FpAttribute getEluAlpha() const noexcept { return mEluAlpha; }
    FpAttribute getSoftPlusBeta() const noexcept { return mSoftPlusBeta; }
    int64_t getAxis() const noexcept { return mAxis; }

private:
    friend class PointwiseBuilder;
};

class PointwiseBuilder
{
private:
    Pointwise mPointwise;
    bool mModeSet          = false;
    bool mMathPrecisionSet = false;

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
    PointwiseBuilder& setReluLowerClip(Pointwise::FpAttribute reluLowerClip) noexcept
    {
        mPointwise.mReluLowerClip = reluLowerClip;
        return *this;
    }
    PointwiseBuilder& setReluUpperClip(Pointwise::FpAttribute reluUpperClip) noexcept
    {
        mPointwise.mReluUpperClip = reluUpperClip;
        return *this;
    }
    PointwiseBuilder& setReluLowerClipSlope(Pointwise::FpAttribute reluLowerClipSlope) noexcept
    {
        mPointwise.mReluLowerClipSlope = reluLowerClipSlope;
        return *this;
    }
    PointwiseBuilder& setEluAlpha(Pointwise::FpAttribute eluAlpha) noexcept
    {
        mPointwise.mEluAlpha = eluAlpha;
        return *this;
    }
    PointwiseBuilder& setSoftPlusBeta(Pointwise::FpAttribute softPlusBeta) noexcept
    {
        mPointwise.mSoftPlusBeta = softPlusBeta;
        return *this;
    }
    PointwiseBuilder& setAxis(int64_t axis) noexcept
    {
        mPointwise.mAxis = axis;
        return *this;
    }

    Pointwise build();
};

class BackendPointwiseDescriptor : public BackendDescriptor
{
private:
    PointwiseBuilder mBuilder;
    Pointwise mPointwise;

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

    const Pointwise* getPointwise() const { return &mPointwise; }
    Pointwise* getPointwise() { return &mPointwise; }
};

} // namespace graphapi

} // namespace miopen
