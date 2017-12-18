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

#include <string>
#include <vector>
#include <sstream>

std::string GetGcnAssemblerPath();
bool ValidateGcnAssembler();
void AmdgcnAssemble(std::string& source, const std::string& params);
bool GcnAssemblerHasBug34765();

template <typename TValue>
void GenerateClangDefsym(std::ostream& stream, const std::string& name, TValue value)
{
    GenerateClangDefsym<const std::string&>(stream, name, std::to_string(value));
}

template <>
void GenerateClangDefsym<const std::string&>(std::ostream& stream,
                                             const std::string& name,
                                             const std::string& value);

/// @param dir 1: fwd, 0: bwd wrt data. Use 0 for WrW.
/// Encodes key with default strides (u1v1)
std::string MakeLutKey(int w, int h, int c, int n, int k, int dir, int CUs = -1);
/// Allows for any strides.
std::string MakeLutKey(int w, int h, int c, int n, int k, int u, int v, int dir, int CUs = -1);

#endif // GCN_ASM_UTILS_H
