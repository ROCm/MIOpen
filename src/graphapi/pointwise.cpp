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
#include <miopen/graphapi/pointwise.hpp>

#include <algorithm>

namespace miopen {

namespace graphapi {

Pointwise PointwiseBuilder::build()
{
    if(!mModeSet || !mMathPrecisionSet)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return mPointwise;
}

void BackendPointwiseDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                              miopenBackendAttributeType_t attributeType,
                                              int64_t elementCount,
                                              void* arrayOfElements)
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    using Setter = PointwiseBuilder& (PointwiseBuilder::*)(Pointwise::FpAttribute value);

    auto setFloatOrDouble = [=](Setter setter) {
        if(attributeType == MIOPEN_TYPE_FLOAT && elementCount == 1)
        {
            (mBuilder.*setter)(*static_cast<float*>(arrayOfElements));
        }
        else if(attributeType == MIOPEN_TYPE_DOUBLE && elementCount == 1)
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
    case MIOPEN_ATTR_POINTWISE_MODE:
        if(attributeType == MIOPEN_TYPE_POINTWISE_MODE && elementCount == 1)
        {
            mBuilder.setMode(*static_cast<miopenPointwiseMode_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_MATH_PREC:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && elementCount == 1)
        {
            mBuilder.setMathPrecision(*static_cast<miopenDataType_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_NAN_PROPAGATION:
        if(attributeType == MIOPEN_TYPE_NAN_PROPOGATION && elementCount == 1)
        {
            mBuilder.setNanPropagation(*static_cast<miopenNanPropagation_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP:
        setFloatOrDouble(&PointwiseBuilder::setReluLowerClip);
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP:
        setFloatOrDouble(&PointwiseBuilder::setReluUpperClip);
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
        setFloatOrDouble(&PointwiseBuilder::setReluLowerClipSlope);
        break;

    case MIOPEN_ATTR_POINTWISE_ELU_ALPHA: setFloatOrDouble(&PointwiseBuilder::setEluAlpha); break;

    case MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA:
        setFloatOrDouble(&PointwiseBuilder::setSoftPlusBeta);
        break;

    case MIOPEN_ATTR_POINTWISE_SWISH_BETA: setFloatOrDouble(&PointwiseBuilder::setSwishBeta); break;

    case MIOPEN_ATTR_POINTWISE_AXIS:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            mBuilder.setAxis(*static_cast<int64_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendPointwiseDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mPointwise = mBuilder.build();
    mFinalized = true;
}

void BackendPointwiseDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
                                              miopenBackendAttributeType_t attributeType,
                                              int64_t requestedElementCount,
                                              int64_t* elementCount,
                                              void* arrayOfElements)
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    using Getter = Pointwise::FpAttribute (Pointwise::*)() const;

    auto getFloatOrDouble = [=](Getter getter) {
        if(attributeType == MIOPEN_TYPE_FLOAT && requestedElementCount == 1)
        {
            *elementCount                         = 1;
            *static_cast<float*>(arrayOfElements) = std::get<float>((mPointwise.*getter)());
        }
        else if(attributeType == MIOPEN_TYPE_DOUBLE && requestedElementCount == 1)
        {
            *elementCount                          = 1;
            *static_cast<double*>(arrayOfElements) = std::get<double>((mPointwise.*getter)());
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
    };

    switch(attributeName)
    {
    case MIOPEN_ATTR_POINTWISE_MODE:
        if(attributeType == MIOPEN_TYPE_POINTWISE_MODE && requestedElementCount == 1)
        {
            *elementCount                                         = 1;
            *static_cast<miopenPointwiseMode_t*>(arrayOfElements) = mPointwise.getMode();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_MATH_PREC:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && requestedElementCount == 1)
        {
            *elementCount                                    = 1;
            *static_cast<miopenDataType_t*>(arrayOfElements) = mPointwise.getMathPrecision();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_NAN_PROPAGATION:
        if(attributeType == MIOPEN_TYPE_NAN_PROPOGATION && requestedElementCount == 1)
        {
            *elementCount                                          = 1;
            *static_cast<miopenNanPropagation_t*>(arrayOfElements) = mPointwise.getNanPropagation();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP:
        getFloatOrDouble(&Pointwise::getReluLowerClip);
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP:
        getFloatOrDouble(&Pointwise::getReluUpperClip);
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
        getFloatOrDouble(&Pointwise::getReluLowerClipSlope);
        break;

    case MIOPEN_ATTR_POINTWISE_ELU_ALPHA: getFloatOrDouble(&Pointwise::getEluAlpha); break;

    case MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA: getFloatOrDouble(&Pointwise::getSoftPlusBeta); break;

    case MIOPEN_ATTR_POINTWISE_SWISH_BETA: getFloatOrDouble(&Pointwise::getSwishBeta); break;

    case MIOPEN_ATTR_POINTWISE_AXIS:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int64_t*>(arrayOfElements) = mPointwise.getAxis();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

std::vector<Tensor*> OperationPointwise::getInTensors() const
{
    switch(mPointwise->getMode())
    {
    /* 2-inputs operations
     * x input
     * b input
     * y output
     */
    case MIOPEN_POINTWISE_ADD:
    case MIOPEN_POINTWISE_ADD_SQUARE:
    case MIOPEN_POINTWISE_DIV:
    case MIOPEN_POINTWISE_MAX:
    case MIOPEN_POINTWISE_MIN:
    case MIOPEN_POINTWISE_MOD:
    case MIOPEN_POINTWISE_MUL:
    case MIOPEN_POINTWISE_POW:
    case MIOPEN_POINTWISE_SUB:
    case MIOPEN_POINTWISE_CMP_EQ:
    case MIOPEN_POINTWISE_CMP_NEQ:
    case MIOPEN_POINTWISE_CMP_GT:
    case MIOPEN_POINTWISE_CMP_GE:
    case MIOPEN_POINTWISE_CMP_LT:
    case MIOPEN_POINTWISE_CMP_LE:
    case MIOPEN_POINTWISE_LOGICAL_AND:
    case MIOPEN_POINTWISE_LOGICAL_OR: return {mX, mB};

    /* Single input operations
     * x input
     * y output
     */
    case MIOPEN_POINTWISE_ABS:
    case MIOPEN_POINTWISE_CEIL:
    case MIOPEN_POINTWISE_COS:
    case MIOPEN_POINTWISE_EXP:
    case MIOPEN_POINTWISE_FLOOR:
    case MIOPEN_POINTWISE_LOG:
    case MIOPEN_POINTWISE_NEG:
    case MIOPEN_POINTWISE_RSQRT:
    case MIOPEN_POINTWISE_SIN:
    case MIOPEN_POINTWISE_SQRT:
    case MIOPEN_POINTWISE_TAN:
    case MIOPEN_POINTWISE_IDENTITY:
    case MIOPEN_POINTWISE_RELU_FWD:
    case MIOPEN_POINTWISE_TANH_FWD:
    case MIOPEN_POINTWISE_SIGMOID_FWD:
    case MIOPEN_POINTWISE_ELU_FWD:
    case MIOPEN_POINTWISE_GELU_FWD:
    case MIOPEN_POINTWISE_SOFTPLUS_FWD:
    case MIOPEN_POINTWISE_SWISH_FWD:
    case MIOPEN_POINTWISE_GELU_APPROX_TANH_FWD:
    case MIOPEN_POINTWISE_LOGICAL_NOT:
    case MIOPEN_POINTWISE_RECIPROCAL: return {mX};

    /* 3-inputs operations
     * x input
     * b input
     * t input
     * y output
     */
    case MIOPEN_POINTWISE_BINARY_SELECT: return {mX, mB, mT};

    /* 2-inputs backward operations
     * y input
     * dy input
     * dx output
     */
    case MIOPEN_POINTWISE_RELU_BWD:
    case MIOPEN_POINTWISE_TANH_BWD:
    case MIOPEN_POINTWISE_SIGMOID_BWD:
    case MIOPEN_POINTWISE_ELU_BWD:
    case MIOPEN_POINTWISE_GELU_BWD:
    case MIOPEN_POINTWISE_SOFTPLUS_BWD:
    case MIOPEN_POINTWISE_SWISH_BWD:
    case MIOPEN_POINTWISE_GELU_APPROX_TANH_BWD: return {mY, mDy};

    /* TODO: Implement the remaining cases
     */
    case MIOPEN_POINTWISE_ERF:
    case MIOPEN_POINTWISE_GEN_INDEX: return {};

    default: MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

std::vector<Tensor*> OperationPointwise::getOutTensors() const
{
    switch(mPointwise->getMode())
    {
    /* 2-inputs operations
     * x input
     * b input
     * y output
     */
    case MIOPEN_POINTWISE_ADD:
    case MIOPEN_POINTWISE_ADD_SQUARE:
    case MIOPEN_POINTWISE_DIV:
    case MIOPEN_POINTWISE_MAX:
    case MIOPEN_POINTWISE_MIN:
    case MIOPEN_POINTWISE_MOD:
    case MIOPEN_POINTWISE_MUL:
    case MIOPEN_POINTWISE_POW:
    case MIOPEN_POINTWISE_SUB:
    case MIOPEN_POINTWISE_CMP_EQ:
    case MIOPEN_POINTWISE_CMP_NEQ:
    case MIOPEN_POINTWISE_CMP_GT:
    case MIOPEN_POINTWISE_CMP_GE:
    case MIOPEN_POINTWISE_CMP_LT:
    case MIOPEN_POINTWISE_CMP_LE:
    case MIOPEN_POINTWISE_LOGICAL_AND:
    case MIOPEN_POINTWISE_LOGICAL_OR:
    /* Single input operations
     * x input
     * y output
     */
    case MIOPEN_POINTWISE_ABS:
    case MIOPEN_POINTWISE_CEIL:
    case MIOPEN_POINTWISE_COS:
    case MIOPEN_POINTWISE_EXP:
    case MIOPEN_POINTWISE_FLOOR:
    case MIOPEN_POINTWISE_LOG:
    case MIOPEN_POINTWISE_NEG:
    case MIOPEN_POINTWISE_RSQRT:
    case MIOPEN_POINTWISE_SIN:
    case MIOPEN_POINTWISE_SQRT:
    case MIOPEN_POINTWISE_TAN:
    case MIOPEN_POINTWISE_IDENTITY:
    case MIOPEN_POINTWISE_RELU_FWD:
    case MIOPEN_POINTWISE_TANH_FWD:
    case MIOPEN_POINTWISE_SIGMOID_FWD:
    case MIOPEN_POINTWISE_ELU_FWD:
    case MIOPEN_POINTWISE_GELU_FWD:
    case MIOPEN_POINTWISE_SOFTPLUS_FWD:
    case MIOPEN_POINTWISE_SWISH_FWD:
    case MIOPEN_POINTWISE_GELU_APPROX_TANH_FWD:
    case MIOPEN_POINTWISE_LOGICAL_NOT:
    case MIOPEN_POINTWISE_RECIPROCAL:
        /* 3-inputs operations
         * x input
         * b input
         * t input
         * y output
         */
    case MIOPEN_POINTWISE_BINARY_SELECT: return {mY};

    /* 2-inputs backward operations
     * y input
     * dy input
     * dx output
     */
    case MIOPEN_POINTWISE_RELU_BWD:
    case MIOPEN_POINTWISE_TANH_BWD:
    case MIOPEN_POINTWISE_SIGMOID_BWD:
    case MIOPEN_POINTWISE_ELU_BWD:
    case MIOPEN_POINTWISE_GELU_BWD:
    case MIOPEN_POINTWISE_SOFTPLUS_BWD:
    case MIOPEN_POINTWISE_SWISH_BWD:
    case MIOPEN_POINTWISE_GELU_APPROX_TANH_BWD: return {mDx};

    /* TODO: Implement the remaining cases
     */
    case MIOPEN_POINTWISE_ERF:
    case MIOPEN_POINTWISE_GEN_INDEX: return {};

    default: MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

namespace {

template <typename Ptr>
void assignPtr(Ptr src, Ptr& dst)
{
    if(src != nullptr)
    {
        dst = src;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace

OperationPointwiseBuilder& OperationPointwiseBuilder::setPointwise(Pointwise* pointwise)
{
    assignPtr(pointwise, mOperationPointwise.mPointwise);
    return *this;
}

OperationPointwiseBuilder& OperationPointwiseBuilder::setX(Tensor* x)
{
    assignPtr(x, mOperationPointwise.mX);
    return *this;
}

OperationPointwiseBuilder& OperationPointwiseBuilder::setB(Tensor* b)
{
    assignPtr(b, mOperationPointwise.mB);
    return *this;
}

OperationPointwiseBuilder& OperationPointwiseBuilder::setY(Tensor* y)
{
    assignPtr(y, mOperationPointwise.mY);
    return *this;
}

OperationPointwiseBuilder& OperationPointwiseBuilder::setT(Tensor* t)
{
    assignPtr(t, mOperationPointwise.mT);
    return *this;
}

OperationPointwiseBuilder& OperationPointwiseBuilder::setDx(Tensor* dX)
{
    assignPtr(dX, mOperationPointwise.mDx);
    return *this;
}

OperationPointwiseBuilder& OperationPointwiseBuilder::setDy(Tensor* dY)
{
    assignPtr(dY, mOperationPointwise.mDy);
    return *this;
}

OperationPointwiseBuilder& OperationPointwiseBuilder::setAlpha1(OperationPointwise::Alpha alpha1)
{
    mOperationPointwise.mAlpha1 = alpha1;
    return *this;
}

OperationPointwiseBuilder& OperationPointwiseBuilder::setAlpha2(OperationPointwise::Alpha alpha2)
{
    mOperationPointwise.mAlpha2 = alpha2;
    mAlpha2Set                  = true;
    return *this;
}

namespace {

template <typename Range1, typename Range2, typename Range3>
bool checkDimsWithPossibleBroadcasting(Range1 input1, Range2 input2, Range3 output)
{
    auto input1it   = input1.cbegin();
    auto input1Last = input1.cend();
    auto input2it   = input2.cbegin();
    auto input2Last = input2.cend();
    auto outputit   = output.cbegin();
    auto outputLast = output.cend();

    bool OK = true;

    for(; OK && input1it != input1Last && input2it != input2Last && outputit != outputLast;
        ++input1it, ++input2it, ++outputit)
    {
        OK = (*input1it == *input2it && *input1it == *outputit) ||
             (*input1it == 1 && *input2it > 1 && *input2it == *outputit) ||
             (*input2it == 1 && *input1it > 1 && *input1it == *outputit);
    }
    OK = OK && input1it == input1Last && input2it == input2Last && outputit == outputLast;

    return OK;
}

} // namespace

OperationPointwise OperationPointwiseBuilder::build()
{
    if(mOperationPointwise.mPointwise == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    switch(mOperationPointwise.mPointwise->getMode())
    {
    /* 2-inputs operations
     * x input
     * b input
     * y output
     */
    case MIOPEN_POINTWISE_ADD:
    case MIOPEN_POINTWISE_ADD_SQUARE:
    case MIOPEN_POINTWISE_DIV:
    case MIOPEN_POINTWISE_MAX:
    case MIOPEN_POINTWISE_MIN:
    case MIOPEN_POINTWISE_MOD:
    case MIOPEN_POINTWISE_MUL:
    case MIOPEN_POINTWISE_POW:
    case MIOPEN_POINTWISE_SUB:
    case MIOPEN_POINTWISE_CMP_EQ:
    case MIOPEN_POINTWISE_CMP_NEQ:
    case MIOPEN_POINTWISE_CMP_GT:
    case MIOPEN_POINTWISE_CMP_GE:
    case MIOPEN_POINTWISE_CMP_LT:
    case MIOPEN_POINTWISE_CMP_LE:
    case MIOPEN_POINTWISE_LOGICAL_AND:
    case MIOPEN_POINTWISE_LOGICAL_OR:
        if(mOperationPointwise.mX == nullptr || mOperationPointwise.mB == nullptr ||
           mOperationPointwise.mY == nullptr || mOperationPointwise.mT != nullptr ||
           mOperationPointwise.mDx != nullptr || mOperationPointwise.mDy != nullptr ||
           !checkDimsWithPossibleBroadcasting(mOperationPointwise.mX->getDimensions(),
                                              mOperationPointwise.mB->getDimensions(),
                                              mOperationPointwise.mY->getDimensions()))
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    /* Single input operations
     * x input
     * y output
     */
    case MIOPEN_POINTWISE_ABS:
    case MIOPEN_POINTWISE_CEIL:
    case MIOPEN_POINTWISE_COS:
    case MIOPEN_POINTWISE_EXP:
    case MIOPEN_POINTWISE_FLOOR:
    case MIOPEN_POINTWISE_LOG:
    case MIOPEN_POINTWISE_NEG:
    case MIOPEN_POINTWISE_RSQRT:
    case MIOPEN_POINTWISE_SIN:
    case MIOPEN_POINTWISE_SQRT:
    case MIOPEN_POINTWISE_TAN:
    case MIOPEN_POINTWISE_IDENTITY:
    case MIOPEN_POINTWISE_RELU_FWD:
    case MIOPEN_POINTWISE_TANH_FWD:
    case MIOPEN_POINTWISE_SIGMOID_FWD:
    case MIOPEN_POINTWISE_ELU_FWD:
    case MIOPEN_POINTWISE_GELU_FWD:
    case MIOPEN_POINTWISE_SOFTPLUS_FWD:
    case MIOPEN_POINTWISE_SWISH_FWD:
    case MIOPEN_POINTWISE_GELU_APPROX_TANH_FWD:
    case MIOPEN_POINTWISE_LOGICAL_NOT:
    case MIOPEN_POINTWISE_RECIPROCAL:
        if(mOperationPointwise.mX == nullptr || mOperationPointwise.mY == nullptr ||
           mOperationPointwise.mB != nullptr || mOperationPointwise.mT != nullptr ||
           mOperationPointwise.mDx != nullptr || mOperationPointwise.mDy != nullptr || mAlpha2Set ||
           !std::equal(mOperationPointwise.mX->getDimensions().cbegin(),
                       mOperationPointwise.mX->getDimensions().cend(),
                       mOperationPointwise.mY->getDimensions().cbegin(),
                       mOperationPointwise.mY->getDimensions().cend()))
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    /* 3-inputs operations
     * x input
     * b input
     * t input
     * y output
     */
    case MIOPEN_POINTWISE_BINARY_SELECT:
        if(mOperationPointwise.mX == nullptr || mOperationPointwise.mB == nullptr ||
           mOperationPointwise.mT == nullptr || mOperationPointwise.mY == nullptr ||
           mOperationPointwise.mDx != nullptr || mOperationPointwise.mDy != nullptr ||
           !std::equal(mOperationPointwise.mX->getDimensions().cbegin(),
                       mOperationPointwise.mX->getDimensions().cend(),
                       mOperationPointwise.mB->getDimensions().cbegin(),
                       mOperationPointwise.mB->getDimensions().cend()) ||
           !std::equal(mOperationPointwise.mX->getDimensions().cbegin(),
                       mOperationPointwise.mX->getDimensions().cend(),
                       mOperationPointwise.mT->getDimensions().cbegin(),
                       mOperationPointwise.mT->getDimensions().cend()) ||
           !std::equal(mOperationPointwise.mX->getDimensions().cbegin(),
                       mOperationPointwise.mX->getDimensions().cend(),
                       mOperationPointwise.mY->getDimensions().cbegin(),
                       mOperationPointwise.mY->getDimensions().cend()))
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    /* 2-inputs backward operations
     * y input
     * dy input
     * dx output
     */
    case MIOPEN_POINTWISE_RELU_BWD:
    case MIOPEN_POINTWISE_TANH_BWD:
    case MIOPEN_POINTWISE_SIGMOID_BWD:
    case MIOPEN_POINTWISE_ELU_BWD:
    case MIOPEN_POINTWISE_GELU_BWD:
    case MIOPEN_POINTWISE_SOFTPLUS_BWD:
    case MIOPEN_POINTWISE_SWISH_BWD:
    case MIOPEN_POINTWISE_GELU_APPROX_TANH_BWD:
        if(mOperationPointwise.mY == nullptr || mOperationPointwise.mDy == nullptr ||
           mOperationPointwise.mDx == nullptr || mOperationPointwise.mX != nullptr ||
           mOperationPointwise.mB != nullptr || mOperationPointwise.mT != nullptr ||
           !checkDimsWithPossibleBroadcasting(mOperationPointwise.mY->getDimensions(),
                                              mOperationPointwise.mDy->getDimensions(),
                                              mOperationPointwise.mDx->getDimensions()))
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    /* TODO: Implement the remaining cases
     */
    case MIOPEN_POINTWISE_ERF:
    case MIOPEN_POINTWISE_GEN_INDEX: MIOPEN_THROW(miopenStatusNotImplemented);

    default: MIOPEN_THROW(miopenStatusNotImplemented);
    }

    return mOperationPointwise;
}

} // namespace graphapi

} // namespace miopen
