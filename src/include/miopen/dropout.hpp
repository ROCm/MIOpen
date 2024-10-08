/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_DROPOUT_HPP_
#define GUARD_MIOPEN_DROPOUT_HPP_

#include <miopen/common.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

#define MAX_PRNG_STATE (256 * 64)
#define MAX_WORKITEM_NUM (256 * 4096)

struct xorwowStates
{
    // Xorshift values (160 bits)
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
    unsigned int v;

    // Weyl sequence value
    unsigned int d;
};

using prngStates = xorwowStates;

namespace miopen {

struct MIOPEN_INTERNALS_EXPORT DropoutDescriptor : miopenDropoutDescriptor
{
    DropoutDescriptor();

    float dropout;
    Data_t pstates;
    size_t stateSizeInBytes;
    unsigned long long seed;
    bool use_mask;
    bool state_evo;
    miopenRNGType_t rng_mode;

    miopenDataType_t dataType_;

    void InitPRNGState(Handle& handle,
                       Data_t prng_states,
                       size_t prng_stateSizeInBytes,
                       unsigned long long prng_seed) const;

    void Dropout(const Handle& handle,
                 const TensorDescriptor& noise_shape,
                 const TensorDescriptor& xDesc,
                 ConstData_t x,
                 const TensorDescriptor& yDesc,
                 Data_t y,
                 Data_t reserveSpace,
                 size_t reserveSpaceSizeInBytes,
                 size_t in_offset    = 0,
                 size_t out_offset   = 0,
                 size_t rsvsp_offset = 0,
                 bool is_backward    = false) const;
};

std::ostream& operator<<(std::ostream& stream, const DropoutDescriptor& x);

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenDropoutDescriptor, miopen::DropoutDescriptor);

#endif // GUARD_MIOPEN_DROPOUT_HPP_
