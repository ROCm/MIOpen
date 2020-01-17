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
#include <miopen/problem_description.hpp>
#include <miopen/miopen.h>

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
struct ConvolutionDescriptor;
struct Handle;
struct TensorDescriptor;

struct ConvolutionUserBuffers
{
    union
    {
        struct Fwd
        {
            ConstData_t x;
            ConstData_t w;
            Data_t y;
        } fwd;
        struct Bwd
        {
            Data_t dx;
            ConstData_t w;
            ConstData_t dy;
        } bwd;
        struct WrW
        {
            ConstData_t dx;
            Data_t dw;
            ConstData_t dy;
        } wrw;
    } io;
    Data_t workSpace;
    size_t workSpaceSize;
    ConstData_t bias;
    ConvolutionUserBuffers(Data_t w, size_t s, ConstData_t b = nullptr)
        : io({{nullptr, nullptr, nullptr}}), workSpace(w), workSpaceSize(s), bias(b)
    {
    }
    ConvolutionUserBuffers() : ConvolutionUserBuffers(nullptr, 0, nullptr) {}
    void SetFwd(ConstData_t x, ConstData_t w, Data_t y)
    {
        io.fwd.x = x;
        io.fwd.y = y;
        io.fwd.w = w;
    }
    void SetBwd(Data_t dx, ConstData_t w, ConstData_t dy)
    {
        io.bwd.dx = dx;
        io.bwd.dy = dy;
        io.bwd.w  = w;
    }
    void SetWrW(ConstData_t dx, Data_t dw, ConstData_t dy)
    {
        io.wrw.dx = dx;
        io.wrw.dy = dy;
        io.wrw.dw = dw;
    }
};

/// A leftover of the legacy design, houses problem config,
/// environmental context (e.g. HW/SW platform) and solver-specific state.
///
/// TODO: These three entities should be made separate.
struct ConvolutionContext : ProblemDescription
{
    // Solution-specific
    std::string general_compile_options;
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

    inline Handle& GetStream() const { return *_stream; }
    inline void SetStream(Handle* stream) { _stream = stream; }

    ConvolutionContext() = default;
    ConvolutionContext(const TensorDescriptor& in,
                       const TensorDescriptor& weights,
                       const TensorDescriptor& out,
                       const ConvolutionDescriptor& conv,
                       int dir,
                       int bias_ = 0)
        : ProblemDescription(in, weights, out, conv, dir, bias_)
    {
    }
    ConvolutionContext(const ProblemDescription& problem) : ProblemDescription(problem) {}

    void DetectRocm();
    void SetupFloats();

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
    Handle* _stream = nullptr;

    public:
    inline void SetBufs(const ConvolutionUserBuffers& bufs) { _bufs = bufs; }
    inline const ConvolutionUserBuffers& GetBufs() const { return _bufs; }

    private:
    ConvolutionUserBuffers _bufs;
};

} // namespace miopen
