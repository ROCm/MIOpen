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

#include <miopen/graphapi/variant_pack.hpp>

namespace miopen {

namespace graphapi {

void BackendVariantPackDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount >= 0)
        {
            mBuilder.setTensorIds(
                std::vector<int64_t>(static_cast<int64_t*>(arrayOfElements),
                                     static_cast<int64_t*>(arrayOfElements) + elementCount));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS:
        if(attributeType == MIOPEN_TYPE_VOID_PTR && elementCount >= 0)
        {
            // Don't use braced-list syntax here, it creates a 2-element vector
            mBuilder.setDataPointers(
                std::vector<void*>(static_cast<void**>(arrayOfElements),
                                   static_cast<void**>(arrayOfElements) + elementCount));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_VARIANT_PACK_INTERMEDIATES: MIOPEN_THROW(miopenStatusNotImplemented);

    case MIOPEN_ATTR_VARIANT_PACK_WORKSPACE:
        if(attributeType == MIOPEN_TYPE_VOID_PTR && elementCount == 1)
        {
            mBuilder.setWorkspace(*static_cast<void**>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendVariantPackDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mVariantPack = std::move(mBuilder).build();
    mFinalized   = true;
}

void BackendVariantPackDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount >= 0)
        {
            *elementCount = mVariantPack.mTensorIds.size();
            std::copy_n(mVariantPack.mTensorIds.cbegin(),
                        std::min(*elementCount, requestedElementCount),
                        static_cast<int64_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS:
        if(attributeType == MIOPEN_TYPE_VOID_PTR && requestedElementCount >= 0)
        {
            *elementCount = mVariantPack.mDataPointers.size();
            std::copy_n(mVariantPack.mDataPointers.cbegin(),
                        std::min(*elementCount, requestedElementCount),
                        static_cast<void**>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_VARIANT_PACK_INTERMEDIATES: MIOPEN_THROW(miopenStatusNotImplemented);

    case MIOPEN_ATTR_VARIANT_PACK_WORKSPACE:
        if(attributeType == MIOPEN_TYPE_VOID_PTR && requestedElementCount == 1)
        {
            *elementCount                         = 1;
            *static_cast<void**>(arrayOfElements) = mVariantPack.getWorkspace();
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
