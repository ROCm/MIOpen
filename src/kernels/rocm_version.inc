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

.ifnotdef ROCM_METADATA_VERSION
.error "Undefined ROCM_METADATA_VERSION"
.end
.endif

.if !((ROCM_METADATA_VERSION == 4) || (ROCM_METADATA_VERSION == 5))
.error "Unsupported ROCM_METADATA_VERSION"
.end
.endif

.if ROCM_METADATA_VERSION == 5
    // The ".option.machine..." symbols pre-defined by the Assember
    // in code-object v2 mode are not exist in CO v3 mode, but we have
    // ".amdgcn.gfx..." symbols instead. Let's define the ".option.machine..."
    // symbols for backward compatibility of CO v2 assembly sources
    // with the code-object-v3 assembly mode.
    .ifdef .option.machine_version_major
        .error ".option.machine_version_major already defined"
        .end
    .endif
    .ifdef .option.machine_version_minor
        .error ".option.machine_version_minor already defined"
        .end
    .endif
    .ifdef .option.machine_version_stepping
        .error ".option.machine_version_stepping already defined"
        .end
    .endif
    .set .option.machine_version_major, .amdgcn.gfx_generation_number
    .set .option.machine_version_minor, .amdgcn.gfx_generation_minor
    .set .option.machine_version_stepping, .amdgcn.gfx_generation_stepping
.endif
