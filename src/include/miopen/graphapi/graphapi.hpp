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
#include <miopen/config.hpp>
#include <miopen/object.hpp>

#include <cstdint>

namespace miopen {

namespace graphapi {

#ifdef _WIN32
// WORKAROUND: building on Windows is failing due to conflicting definitions of std::min()
// between the MSVC standard library and HIP Clang wrappers for int64_t data type.
constexpr std::int64_t minimum(std::int64_t a, std::int64_t b) { return a < b ? a : b; }
#else
#define minimum std::min
#endif

class OpNode;

class MIOPEN_INTERNALS_EXPORT BackendDescriptor : public miopenBackendDescriptor
{
public:
    virtual ~BackendDescriptor();
    virtual void setAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              void* arrayOfElements) = 0;
    virtual void finalize()                          = 0;
    virtual void getAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) = 0;
    virtual void execute(miopenHandle_t handle, miopenBackendDescriptor_t variantPack);
    virtual OpNode* getOperation();

    bool isFinalized() const noexcept { return mFinalized; };

protected:
    bool mFinalized = false;
};
} // namespace graphapi
} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenBackendDescriptor, miopen::graphapi::BackendDescriptor)
