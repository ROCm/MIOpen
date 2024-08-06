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
#ifndef GUARD_COMGR_HPP
#define GUARD_COMGR_HPP

#include <miopen/config.h>
#if MIOPEN_USE_COMGR

#include <miopen/target_properties.hpp>
#include <string>
#include <vector>

namespace miopen {
namespace comgr {

void BuildOcl(const std::string& name,
              std::string_view text,
              const std::string& options,
              const miopen::TargetProperties& target,
              std::vector<char>& binary);

void BuildAsm(const std::string& name,
              std::string_view text,
              const std::string& options,
              const miopen::TargetProperties& target,
              std::vector<char>& binary);

} // namespace comgr
} // namespace miopen

#endif // MIOPEN_USE_COMGR

#if MIOPEN_USE_HIPRTC

namespace miopen {
namespace hiprtc {

void BuildHip(const std::string& name,
              std::string_view text,
              const std::string& options,
              const miopen::TargetProperties& target,
              std::vector<char>& binary);

} // namespace hiprtc
} // namespace miopen

#endif // MIOPEN_USE_HIPRTC

#endif // GUARD_COMGR_HPP
