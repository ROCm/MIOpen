/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/solution.hpp>

namespace miopen {

std::size_t Solution::GetSize() const
{
    MIOPEN_THROW(miopenStatusNotImplemented);
    return 0;
}

void Solution::Save(char* data) const
{
    std::ignore = data;

    MIOPEN_THROW(miopenStatusNotImplemented);
}

void Solution::Load(const char* data, std::size_t size)
{
    std::ignore = data;
    std::ignore = size;

    MIOPEN_THROW(miopenStatusNotImplemented);
}

void Solution::Run(const Handle& handle,
                   const std::vector<RunInput>& inputs,
                   void* workspace,
                   size_t workspaceSize)
{
    std::ignore = handle;
    std::ignore = inputs;
    std::ignore = workspace;
    std::ignore = workspaceSize;

    MIOPEN_THROW(miopenStatusNotImplemented);
}

} // namespace miopen
