/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GCN_ASM_UTILS_H
#define GCN_ASM_UTILS_H

#include <miopen/config.h>
#include <miopen/target_properties.hpp>
#include <sstream>
#include <string>

/// Since 3.8.20403, ".amdhsa_reserve_xnack_mask 0" is not working without
/// explicit "-mno-xnack" option.
#define WORKAROUND_SWDEV_255735 1

bool ValidateGcnAssembler();
#if !MIOPEN_USE_COMGR
std::string AmdgcnAssemble(const std::string& source,
                           const std::string& params,
                           const miopen::TargetProperties& target);
#endif

template <typename TValue>
void GenerateClangDefsym(std::ostream& stream, const std::string& name, TValue value)
{
    GenerateClangDefsym<const std::string&>(stream, name, std::to_string(value));
}

template <>
void GenerateClangDefsym<const std::string&>(std::ostream& stream,
                                             const std::string& name,
                                             const std::string& value);

#endif // GCN_ASM_UTILS_H
