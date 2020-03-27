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

#pragma once

#include <miopen/db_path.hpp>
#include <miopen/handle.hpp>

#include <string>

class rocm_meta_version
{
    int val = Unknown;

    public:
    static constexpr int Unknown = 0, // Unset env.vars read as 0.
        AMDHSA_COv2              = 1, // V2 metadata, https://llvm.org/docs/AMDGPUUsage.html
        AMDHSA_COv2_COv3         = 2, // E.g. ROCm 2.10 supports both.
        AMDHSA_COv3              = 3, // V3 metadata, https://llvm.org/docs/AMDGPUUsage.html
        Default                  = AMDHSA_COv2; // Used when auto-detection fails.

    private:
    static constexpr int End = 4, Begin = Unknown;

    public:
    rocm_meta_version(int v) : val(v) {}
    int getValue() const { return val; }
    bool IsValid() const { return Begin <= val && val < End; }
    bool IsUnknown() const { return val == Unknown; }
    bool IsV2() const { return AMDHSA_COv2 <= val && val <= AMDHSA_COv2_COv3; }
    bool IsV2orV3() const { return AMDHSA_COv2 <= val && val <= AMDHSA_COv3; }
    bool IsV3() const { return AMDHSA_COv2_COv3 <= val && val <= AMDHSA_COv3; }
    bool UseV3() const;
};

namespace miopen {

struct ExecutionContext
{
    // Operation modes & environment
    bool do_search               = false;
    bool save_srch_req           = false;
    bool use_asm_kernels         = false;
    bool use_opencl_convolutions = true;
    bool use_binaries            = true;
    rocm_meta_version rmv        = rocm_meta_version::Default;
    bool disable_search_enforce  = false;
    // Skip perf-db reads and use the default performance configuration. This is used, for example,
    // to optimize the getWorkspaceSize() calls for speed. This specific optimization is correct
    // because Solvers shall be written so that the required workspace size does not depend on the
    // performance config.
    bool disable_perfdb_access = false;

    inline Handle& GetStream() const { return *stream; }
    inline void SetStream(Handle* stream_) { stream = stream_; }

    ExecutionContext() = default;

    void DetectRocm();

    std::string GetPerfDbPath() const
    {
        // clang-format off
        return GetSystemDbPath()
#if MIOPEN_ENABLE_SQLITE
            + "/miopen.db";
#else
            + "/"
            + GetStream().GetDbBasename()
            + ".cd.pdb.txt";
#endif
        // clang-format on
    }

    std::string GetUserPerfDbPath() const
    {
        // clang-format off
        return GetUserDbPath()
#if MIOPEN_ENABLE_SQLITE
             + "/miopen.udb";
#else
             + "/"
             + GetStream().GetDbBasename()
             + "."
             + GetUserDbSuffix()
             + ".cd.updb.txt";
#endif
        // clang-format on
    }

    private:
    Handle* stream = nullptr;
};
} // namespace miopen
