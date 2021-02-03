/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef GUARD_ROCM_FEATURES_HPP_
#define GUARD_ROCM_FEATURES_HPP_

#include <miopen/config.h>

/// Older HIP runtimes return hipDeviceProp_t.gcnArchName with codenames
/// of GPUs instead of valid names, e.g. "Vega 20" instead of "gfx906".
/// To be removed as soon as support for ROCm 3.x is discontinued.
#define ROCM_FEATURE_HIP_GCNARCHNAME_RETURNS_CODENAME (HIP_PACKAGE_VERSION_FLAT < 4000000000ULL)

/// Workaround for https://github.com/AMDComputeLibraries/MLOpen/issues/1711:
/// Since ROCM 2.4 rc1, OCL returns "gfx906+sram-ecc" on a gfx906 machine.
/// See also rejected SWDEV-188028. Fixed since ROCm 4.0 or even sooner.
/// To be removed as soon as support for ROCm 3.x is discontinued.
#define WORKAROUND_MLOPEN_ISSUE_1711 (HIP_PACKAGE_VERSION_FLAT < 4000000000ULL)

#endif // GUARD_ROCM_FEATURES_HPP_
