/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef CK_REDUCTION_ENUMS_HPP
#define CK_REDUCTION_ENUMS_HPP

namespace ck {

enum class ReduceTensorOp_t
{
    ADD   = 0,
    MUL   = 1,
    MIN   = 2,
    MAX   = 3,
    AMAX  = 4,
    AVG   = 5,
    NORM1 = 6,
    NORM2 = 7,
    // MUL_NO_ZEROS = 8,
};

enum class NanPropagation_t
{
    NOT_PROPAGATE_NAN = 0,
    PROPAGATE_NAN     = 1,
};

enum class ReduceTensorIndices_t
{
    NO_INDICES        = 0,
    FLATTENED_INDICES = 1,
};

enum class IndicesType_t
{
    INDICES_32BIT = 0,
    INDICES_64BIT = 1,
    INDICES_16BIT = 2,
    INDICES_8BIT  = 3,
};

}; // end of namespace ck

#endif
