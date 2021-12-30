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

/// Fix for SWDEV-255735. Since 3.8.20403, ".amdhsa_reserve_xnack_mask 0"
/// is not working without explicit "-mno-xnack" option.
#define ROCM_FEATURE_ASM_REQUIRES_NO_XNACK_OPTION (HIP_PACKAGE_VERSION_FLAT >= 3008020403ULL)

/// Older HIP runtimes return hipDeviceProp_t.gcnArchName with codenames
/// of GPUs instead of valid names, e.g. "Vega 20" instead of "gfx906".
/// To be removed as soon as support for ROCm 3.x is discontinued.
#define ROCM_FEATURE_HIP_GCNARCHNAME_RETURNS_CODENAME (HIP_PACKAGE_VERSION_FLAT < 4000000000ULL)

/// Workaround for https://github.com/AMDComputeLibraries/MLOpen/issues/1711:
/// Since ROCM 2.4 rc1, OCL returns "gfx906+sram-ecc" on a gfx906 machine.
/// See also rejected SWDEV-188028. Fixed since ROCm 4.0 or even sooner.
/// To be removed as soon as support for ROCm 3.x is discontinued.
#define WORKAROUND_MLOPEN_ISSUE_1711 (HIP_PACKAGE_VERSION_FLAT < 4000000000ULL)

/// W/A for MIOpenGEMM issues with ROCm 4.1 and newer ROCm
/// versions. The issue is highly likely related to the
/// issues in the OpenCL compiler or in MIOpenGEMM itself.
/// MIOpenGEMM is used only for OCL BE and deprecated.
/// Related ticket: http://ontrack-internal.amd.com/browse/SWDEV-276757
///
/// Some failing cases:
/// test_immed_conv2d --float --cmode conv --pmode default --group-count 1
///  --input 1, 3, 224, 224 --weights 1, 3, 11, 11
///   --pads_strides_dilations 1 1 1 1 1 1 --trans_output_pads 0 0
///  --input 1, 3, 224, 224 --weights 1, 3, 7, 7
///   --pads_strides_dilations 3 3 2 2 1 1 --trans_output_pads 0 0
/// test_immed_conv3d --float --cmode conv --pmode default --group-count 1
///  --input 1, 4, 4, 161, 700 --weights 1, 4, 3, 11, 11
///   --pads_strides_dilations 3 3 3 2 2 2 4 4 4 --trans_output_pads 0 0 0
///
/// W/A is in effect only when MIOpenGEMM is used (OCL BE) and disables
/// GEMM for the failing configs. When this happens, Naive solvers
/// are used as backup on the Immediate Mode Fallback path.
#define WORKAROUND_MIOPENGEMM_SINCE_ROCM41 \
    (MIOPEN_USE_MIOPENGEMM && (HIP_PACKAGE_VERSION_FLAT >= 4001000000ULL))

#define ROCM_FEATURE_TARGETID_OFF (HIP_PACKAGE_VERSION_FLAT < 4001000000ULL)

/// Return type of llvm.amdgcn.buffer.atomic.fadd.f32 can't be detected.
/// With COMGR: llvm sends SIGABRT when return type is wrong.
/// SIGABRT means that llvm instance can't be used anymore, therefore
/// custom handling of this signal should not be used.
/// Repetitive use of llvm instance after that would lead to UB.
/// Without COMGR: at least 3.10 compiler doesn't care of return type of this atomic.
/// Therefore auto-detection delivers wrong information we should not rely on.
#define ROCM_FEATURE_LLVM_AMDGCN_BUFFER_ATOMIC_FADD_F32_RETURNS_FLOAT \
    (HIP_PACKAGE_VERSION_FLAT >= 4001021072ULL)

#endif // GUARD_ROCM_FEATURES_HPP_
