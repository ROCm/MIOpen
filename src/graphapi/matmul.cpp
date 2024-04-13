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
#include <miopen/graphapi/matmul.hpp>

namespace miopen {
namespace graphapi {

Matmul MatmulBuilder::build() const
{
    if(!mComputeTypeSet)
        MIOPEN_THROW(miopenStatusBadParm);
    return mMatmul;
}

void BackendMatmulDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                           miopenBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           void* arrayOfElements)
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_MATMUL_COMP_TYPE:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && elementCount == 1)
        {
            mBuilder.setComputeType(*static_cast<miopenDataType_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendMatmulDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mMatmul    = mBuilder.build();
    mFinalized = true;
}

void BackendMatmulDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
                                           miopenBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements)
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_MATMUL_COMP_TYPE:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && requestedElementCount == 1)
        {
            *elementCount                                    = 1;
            *static_cast<miopenDataType_t*>(arrayOfElements) = mMatmul.getComputeType();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace graphapi
} // namespace miopen
