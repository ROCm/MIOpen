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
/**********************************************************************
Copyright (c)2017 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef MLO_INTERNAL_H_
#define MLO_INTERNAL_H_

// Header Files
#ifndef NOMINMAX
#define NOMINMAX // stupid windows.h confused with min() macros in std namespace
#endif

#include <miopen/config.h>

#if MIOPEN_BACKEND_OPENCL
#include <miopen/oclkernel.hpp>
#include <miopen/clhelper.hpp>
#include <miopen/ocldeviceinfo.hpp>
#endif
#include <miopen/db_path.hpp>
#include <miopen/handle.hpp>
#include <miopen/problem_description.hpp>

#if MIOPEN_BACKEND_OPENCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifdef __APPLE__
#include <mach/mach_time.h> // for mach_absolute_time() and friends
#endif

#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <map>
#include <string>
#include <limits>
#include <algorithm> // std::find  and std::min std::maxx

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <ctime>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#include <cstdint>
#include <tuple>

using mlo_kernel_info = std::tuple<const std::string,
                                   const std::string,
                                   const std::string,
                                   const std::vector<size_t>,
                                   const std::vector<size_t>>;

inline int mloLg2(int v)
{
    auto ret = static_cast<int>(std::ceil(std::log(v) / std::log(2)));
    return (ret);
}

inline int AlignUp(int val, unsigned step)
{
    assert(val >= 0);
    return static_cast<int>(((static_cast<unsigned>(val) + step - 1) / step) * step);
}

enum class rocm_meta_version
{
    Unknown,
    AMDHSA_1_0,           // 1.0, see https://llvm.org/docs/AMDGPUUsage.html#code-object-metadata
    Default = AMDHSA_1_0, // Assumption for HIP backend. To be updated together with ROCm release.
};

namespace miopen {

struct TensorDescriptor;

template <class TInstalled, class TUser, bool merge_records>
class MultiFileDb;

class ReadonlyRamDb;
class Db;

template <class TInnerDb>
class DbTimer;

struct ConvolutionContext;

template <class TInstance>
class StaticContainer
{
    public:
    inline static TInstance& Instance()
    {
        static TInstance data{};
        return data;
    }
};

using PerfDb = DbTimer<MultiFileDb<Db, Db, true>>;
PerfDb GetDb(const ConvolutionContext& ctx);

template <class TTo>
size_t setTopDescFromMLDesc(int spatial_dims, TTo& to, const TensorDescriptor& tensor)
{
    return SetDescFromMLDesc(spatial_dims, to, tensor, &TTo::setTopDescr);
}

template <class TTo>
size_t setBotDescFromMLDesc(int spatial_dims, TTo& to, const TensorDescriptor& tensor)
{
    return SetDescFromMLDesc(spatial_dims, to, tensor, &TTo::setBotDescr);
}

template <class TTo>
size_t setTopDfDescFromMLDesc(int spatial_dims, TTo& to, const TensorDescriptor& tensor)
{
    return SetDescFromMLDesc(spatial_dims, to, tensor, &TTo::setTopDfDescr);
}

template <class TTo>
size_t setBotDfDescFromMLDesc(int spatial_dims, TTo& to, const TensorDescriptor& tensor)
{
    return SetDescFromMLDesc(spatial_dims, to, tensor, &TTo::setBotDfDescr);
}

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
    bool do_search              = false;
    bool save_srch_req          = false;
    bool use_asm_kernels        = false;
    bool use_binaries           = true;
    rocm_meta_version rmv       = rocm_meta_version::Default;
    bool disable_search_enforce = false;
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
             + "/"
             + GetStream().GetDbBasename()
             + ".cd.pdb.txt";
        // clang-format on
    }

    std::string GetUserPerfDbPath() const
    {
        // clang-format off
        return GetUserDbPath()
             + "/"
             + GetStream().GetDbBasename()
			 + "."
			 + GetUserDbSuffix()
             + ".cd.updb.txt";
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

namespace solver {
struct ConvSolution;

} // namespace solver

} // namespace miopen

template <class T>
auto mloConstruct(T& x) -> decltype(x.mloConstruct(), void())
{
    x.setupFloats();
    x.mloConstruct();
}

template <class T>
auto FindFirstSolution(T& x) -> decltype(x.FindSolution())
{
    x.detectRocm();
    x.setupFloats();
    return x.FindSolution();
}

template <class T, class U>
auto FindFirstSolution(T& x, U& solvers) -> decltype(x.FindSolution(solvers))
{
    x.detectRocm();
    x.setupFloats();
    return x.FindSolution(solvers);
}

template <class T>
auto FindAllSolutions(T& x) -> decltype(x.FindAllSolutions())
{
    x.detectRocm();
    x.setupFloats();
    return x.FindAllSolutions();
}

std::vector<miopen::solver::ConvSolution>
FindAllDirectSolutions(const miopen::ConvolutionContext& ctx);

std::vector<std::pair<std::string, size_t>>
AllDirectForwardBackwardDataWorkspaceSize(const miopen::ConvolutionContext& ctx);
std::vector<std::pair<std::string, size_t>>
AllDirectBwdWrW2DWorkspaceSize(const miopen::ConvolutionContext& ctx);

std::vector<miopen::solver::ConvSolution>
FindAllImplicitGemmSolutions(const miopen::ConvolutionContext& ctx);

std::vector<miopen::solver::ConvSolution>
FindAllWinogradSolutions(const miopen::ConvolutionContext& ctx);
miopen::solver::ConvSolution FindWinogradSolution(const miopen::ConvolutionContext& ctx);

std::vector<miopen::solver::ConvSolution>
FindWinogradWrWAllSolutions(const miopen::ConvolutionContext& ctx);

std::vector<miopen::solver::ConvSolution>
FindAllBwdWrW2DSolutions(const miopen::ConvolutionContext& ctx);

std::vector<miopen::solver::ConvSolution>
FindAllFwdSCGemmSolutions(const miopen::ConvolutionContext& ctx);

/*
 * returns parameter values that are compiled in legacy kernels for kernels using them as
 * arguments.
 */
inline void GetCompiledInParameters(const miopen::ConvolutionContext& ctx,
                                    int* const N,
                                    int* const C,
                                    int* const H,
                                    int* const W,
                                    int* const K,
                                    int* const n_groups)
{
    assert(N && C && H && W && K && n_groups);
    *N        = ctx.batch_sz;
    *C        = ctx.n_inputs;
    *H        = ctx.in_height;
    *W        = ctx.in_width;
    *K        = ctx.n_outputs;
    *n_groups = ctx.GetStream().GetMaxComputeUnits();
}

inline void GetCompiledInParameters(const miopen::ConvolutionContext& ctx,
                                    int* const N,
                                    int* const C,
                                    int* const H,
                                    int* const W,
                                    int* const K,
                                    int* const n_groups,
                                    int* const out_H,
                                    int* const out_W)
{
    GetCompiledInParameters(ctx, N, C, H, W, K, n_groups);
    assert(out_H && out_W);
    *out_H = ctx.out_height;
    *out_W = ctx.out_width;
}

inline void GetCompiledInParameters(const miopen::ConvolutionContext& ctx,
                                    int* const N,
                                    int* const C,
                                    int* const H,
                                    int* const W,
                                    int* const K,
                                    int* const n_groups,
                                    int* const out_H,
                                    int* const out_W,
                                    int* const filter_size_H,
                                    int* const filter_size_W,
                                    int* const pad_H,
                                    int* const pad_W)
{
    GetCompiledInParameters(ctx, N, C, H, W, K, n_groups, out_H, out_W);
    assert(filter_size_H && filter_size_W && pad_H && pad_W);
    *filter_size_H = ctx.kernel_size_h;
    *filter_size_W = ctx.kernel_size_w;
    *pad_H         = ctx.direction.IsForward() ? ctx.pad_h : ctx.GetBackwardPadH();
    *pad_W         = ctx.direction.IsForward() ? ctx.pad_w : ctx.GetBackwardPadW();
}

struct mlo_construct_base
{
    mlo_construct_base(int dir, bool do_bias = false)
    {
        _search_params.direction.Set(dir);
        _search_params.bias              = (do_bias) ? 1 : 0;
        _search_params.pad_w             = 1;
        _search_params.pad_h             = 1;
        _search_params.kernel_size_w     = 3;
        _search_params.kernel_size_h     = 3;
        _search_params.kernel_stride_w   = 1;
        _search_params.kernel_stride_h   = 1;
        _search_params.kernel_dilation_w = 1;
        _search_params.kernel_dilation_h = 1;
        _search_params.deconvolution     = 0;
        _search_params.bot_sz            = 0; // bytes
        _search_params.top_sz            = 0; // bytes
        _search_params.weights_sz        = 0; // bytes
        _search_params.bias_sz           = 0; // bytes
        _search_params.group_counts      = 1;
    }

    mlo_construct_base(const miopen::TensorDescriptor& in,
                       const miopen::TensorDescriptor& weights,
                       const miopen::TensorDescriptor& out,
                       const miopen::ConvolutionDescriptor& conv,
                       int dir,
                       bool do_bias = false)
        : _search_params(in, weights, out, conv, dir, (do_bias) ? 1 : 0)
    {
        _search_params.deconvolution = 0;
    }

    void detectRocm() { _search_params.DetectRocm(); }
    void setupFloats() { _search_params.SetupFloats(); }

    miopen::PerfDb GetDb() const;

    /*
     * get common compiler options
     */
    inline const std::string& getGeneralCompOptions() const
    {
        return (_search_params.general_compile_options);
    }

    /*
     * return direction: true - forward, false - backward
     */
    inline bool isForwardDirection() const
    {
        if(!_search_params.direction.IsKnown())
            MIOPEN_THROW("!_search_params.direction.IsKnown()");
        return _search_params.direction.IsForward(); // convolutions: backward data OR wrw otherwise
    }

    /*
     * set library stream
     */
    inline void setStream(miopen::Handle* stream) { _search_params.SetStream(stream); }

    // MD: Hack to get the key outside of mlo_internal
    int mloBuildConf_Key(std::string& conf_key) const
    {
        return _search_params.mloBuildConf_Key(conf_key);
    }

    std::string db_path() const
    {
        return _db_path != nullptr ? _db_path : _search_params.GetPerfDbPath();
    }

    protected:
    miopen::ConvolutionContext _search_params;

    const char* _db_path = nullptr;
};

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1
#define MLO_POOLING_OP_STC 2
#define MLO_POOLING_OP_AVE_INCLUSIVE 3

/// \todo Move this into respective Solution objects. --atamazov
struct mlo_construct_activ_lrn_pooling_common : mlo_construct_base
{
    std::string _comp_options;
    std::string _kernel_file;
    std::string _kernel_name;
    std::vector<size_t> _l_wk;
    std::vector<size_t> _g_wk;

    mlo_construct_activ_lrn_pooling_common(int dir) : mlo_construct_base(dir) {}

    /*
     * returns kernel file name without location
     */
    inline std::string getKernelFile() const { return (_kernel_file); }
    /*
     * retuns kerner/shader name
     */
    inline std::string getKernelName() const { return (_kernel_name); }
    /*
     * return set of compile options
     */
    inline const std::string& getCompilerOptions() const { return (_comp_options); }
    /*
     *  return a local working configuration
     */
    inline const std::vector<size_t>& getLocalWkSize() const { return (_l_wk); }
    /*
     * return a global working configuration
     */
    inline const std::vector<size_t>& getGlobalWkSize() const { return (_g_wk); }

    int _grp_tile0      = 8; // total number ALUs per group
    int _grp_tile1      = 8; // total number ALUs per group
    int _out_pix_tile0  = 2; // # of generated pixels per output per wk-item  (ALU)
    int _out_pix_tile1  = 4; // # of generated pixels per output per wk-item  (ALU)
    size_t _workspce_sz = 0;

    /*
     * get workspace size
     */
    inline size_t getWorkSpaceSzBytes() const { return (_workspce_sz); }

    void setupFloats();

    inline void setBufs(const miopen::ConvolutionUserBuffers& bufs)
    {
        _search_params.SetBufs(bufs);
    }
    /*
     * set top tensor
     */
    void setTopDescr(const std::string& layout,
                     miopenDataType_t data_type,
                     int batch,
                     int channels,
                     int depth,
                     int height,
                     int width,
                     int batch_stride,
                     int channel_stride,
                     int stride,
                     int w_stride)
    {
        _search_params.setTopDescr(layout,
                                   data_type,
                                   batch,
                                   channels,
                                   depth,
                                   height,
                                   width,
                                   batch_stride,
                                   channel_stride,
                                   stride,
                                   w_stride);
    }

    /*
     *  set bot tensor
     */
    void setBotDescr(const std::string& layout,
                     miopenDataType_t data_type,
                     int batch,
                     int channels,
                     int depth,
                     int height,
                     int width,
                     int batch_stride,
                     int channel_stride,
                     int stride,
                     int w_stride)
    {
        _search_params.setBotDescr(layout,
                                   data_type,
                                   batch,
                                   channels,
                                   depth,
                                   height,
                                   width,
                                   batch_stride,
                                   channel_stride,
                                   stride,
                                   w_stride);
    }

    /*
     * set top df tensor
     */
    void setTopDfDescr(const std::string& layout,
                       miopenDataType_t data_type,
                       int batch,
                       int channels,
                       int depth,
                       int height,
                       int width,
                       int batch_stride,
                       int channel_stride,
                       int stride,
                       int w_stride)
    {
        _search_params.setTopDfDescr(layout,
                                     data_type,
                                     batch,
                                     channels,
                                     depth,
                                     height,
                                     width,
                                     batch_stride,
                                     channel_stride,
                                     stride,
                                     w_stride);

        int data_len = miopen::GetTypeSize(data_type);
        size_t size  = (layout == "NCHW")
                          ? batch * channels * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;

        _out_df_width          = width;
        _out_df_height         = height;
        _out_df_batch_stride   = batch_stride;
        _out_df_channel_stride = channel_stride;
        _out_df_stride         = stride;
        _top_df_sz             = size;
        _out_df_layout         = layout;
        _out_df_data_type      = miopen::GetDataTypeName(data_type);
    }

    /*
     *  set bot df tensor
     */
    void setBotDfDescr(const std::string& layout,
                       miopenDataType_t data_type,
                       int batch,
                       int channels,
                       int depth,
                       int height,
                       int width,
                       int batch_stride,
                       int channel_stride,
                       int stride,
                       int w_stride)
    {
        _search_params.setBotDfDescr(layout,
                                     data_type,
                                     batch,
                                     channels,
                                     depth,
                                     height,
                                     width,
                                     batch_stride,
                                     channel_stride,
                                     stride,
                                     w_stride);

        int data_len = miopen::GetTypeSize(data_type);
        size_t size  = (layout == "NCHW")
                          ? batch * channels * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;

        _in_df_width          = width;
        _in_df_height         = height;
        _in_df_batch_stride   = batch_stride;
        _in_df_channel_stride = channel_stride;
        _in_df_stride         = stride;
        _bot_df_sz            = size;
        _in_df_layout         = layout;
        _in_df_data_type      = miopen::GetDataTypeName(data_type);
    }

    size_t setTopDescFromMLDesc(const miopen::TensorDescriptor& tensor)
    {
        return miopen::setTopDescFromMLDesc(_search_params.spatial_dims, *this, tensor);
    }

    size_t setBotDescFromMLDesc(const miopen::TensorDescriptor& tensor)
    {
        return miopen::setBotDescFromMLDesc(_search_params.spatial_dims, *this, tensor);
    }

    size_t setTopDfDescFromMLDesc(const miopen::TensorDescriptor& tensor)
    {
        return miopen::setTopDfDescFromMLDesc(_search_params.spatial_dims, *this, tensor);
    }

    size_t setBotDfDescFromMLDesc(const miopen::TensorDescriptor& tensor)
    {
        return miopen::setBotDfDescFromMLDesc(_search_params.spatial_dims, *this, tensor);
    }

    /*
     *  indicate the need for backward pass
     */
    inline void doBackward(bool do_bwd) { _do_backward = do_bwd; }
    /*
     * need for backward pass?
     */
    inline bool doBackward() const { return (_do_backward); }

    protected:
    bool _do_backward = false;
    int _hw_wave_sz   = 0;

    // cl_queue
    std::size_t _bot_df_sz = 0; /// \todo Written but not read - remove?
    std::size_t _top_df_sz = 0; /// \todo Written but not read - remove?

    int _in_df_width          = 0;
    int _in_df_height         = 0;
    int _in_df_batch_stride   = 0;
    int _in_df_channel_stride = 0;
    int _in_df_stride         = 0;
    std::string _in_df_layout;
    std::string _in_df_data_type;

    int _out_df_width          = 0;
    int _out_df_height         = 0;
    int _out_df_batch_stride   = 0;
    int _out_df_channel_stride = 0;
    int _out_df_stride         = 0;
    std::string _out_df_layout;
    std::string _out_df_data_type;
};

struct mlo_construct_pooling2D : mlo_construct_activ_lrn_pooling_common
{
    mlo_construct_pooling2D(int dir) : mlo_construct_activ_lrn_pooling_common(dir)
    {
        _pooling_method = MLO_POOLING_OP_MAX;
        _index_type     = miopenIndexUint8;
        _NAN_option     = 0;
    }

    inline void setPoolingDescr(int pooling_method           = MLO_POOLING_OP_MAX,
                                miopenIndexType_t index_type = miopenIndexUint8,
                                int windowHeight             = 3,
                                int windowWidth              = 3,
                                int padding_h                = 0,
                                int padding_w                = 0,
                                int stride_h                 = 2,
                                int stride_w                 = 2,
                                int NAN_opt                  = 0)
    {
        _pooling_method                = pooling_method;
        _index_type                    = index_type;
        _search_params.pad_h           = padding_h;
        _search_params.pad_w           = padding_w;
        _search_params.kernel_size_h   = windowHeight;
        _search_params.kernel_size_w   = windowWidth;
        _search_params.kernel_stride_h = stride_h;
        _search_params.kernel_stride_w = stride_w;
        _NAN_option                    = NAN_opt;
    }

    inline void getPoolingDescr(int& /*pooling_method*/,
                                miopenIndexType_t& index_type,
                                int& windowHeight,
                                int& windowWidth,
                                int& padding_h,
                                int& padding_w,
                                int& stride_h,
                                int& stride_w,
                                int& NAN_opt) const
    {
        index_type   = _index_type;
        padding_h    = _search_params.pad_h;
        padding_w    = _search_params.pad_w;
        windowHeight = _search_params.kernel_size_h;
        windowWidth  = _search_params.kernel_size_w;
        stride_h     = _search_params.kernel_stride_h;
        stride_w     = _search_params.kernel_stride_w;
        NAN_opt      = _NAN_option;
    }

    inline int getPoolingMethod() const { return (_pooling_method); }
    int mloConstruct();

    protected:
    int _pooling_method;
    miopenIndexType_t _index_type;
    int _NAN_option;
    int mloConstructFwd();
    int mloConstructBwd();
};

#define MLO_LRN_WITHIN_CHANNEL 0
#define MLO_LRN_ACROSS_CHANNELS 1

struct mlo_construct_norm : mlo_construct_activ_lrn_pooling_common
{
    mlo_construct_norm(int dir) : mlo_construct_activ_lrn_pooling_common(dir) {}

    inline void setNormDescr(
        int norm_region, int norm_area, double normAlpha, double normBeta, double normK = 1.)
    {
        _norm_region = norm_region;
        _norm_area   = norm_area;
        _normAlpha   = normAlpha;
        _normBeta    = normBeta;
        _normK       = normK;
    }

    inline void getNormDescr(int& norm_region,
                             int& norm_area,
                             double& normAlpha,
                             double& normBeta,
                             double& normK,
                             double& alphaoverarea) const
    {
        norm_region   = _norm_region;
        norm_area     = _norm_area;
        normAlpha     = _normAlpha;
        normBeta      = _normBeta;
        normK         = _normK;
        alphaoverarea = (_norm_region == MLO_LRN_ACROSS_CHANNELS)
                            ? _normAlpha / _norm_area
                            : _normAlpha / (_norm_area * _norm_area);
    }

    void mloConstruct();

    protected:
    int mloConstructFwd();
    int mloConstructBwd();
    int _norm_region  = 0;
    int _norm_area    = 0;
    double _normAlpha = 0.0;
    double _normBeta  = 0.0;
    double _normK     = 0.0;
};

struct mlo_construct_neuron : mlo_construct_activ_lrn_pooling_common
{
    mlo_construct_neuron(int dir) : mlo_construct_activ_lrn_pooling_common(dir)
    {
        _neuron_type = 0;
        _gamma       = 0;
        _beta        = 1;
        _alpha       = 0;
    }

    inline void setNeuronDescr(int neuron_type, double gamma, double beta, double alpha)
    {
        _neuron_type = neuron_type;
        _gamma       = gamma;
        _beta        = beta;
        _alpha       = alpha;
    }

    inline void getNeuronDescr(int& neuron_type, double& gamma, double& beta, double& alpha) const
    {
        neuron_type = _neuron_type;
        gamma       = _gamma;
        beta        = _beta;
        alpha       = _alpha;
    }

    void mloConstruct();

    protected:
    int mloConstructFwd();
    int mloConstructBwd();
    int _neuron_type;
    double _gamma;
    double _beta;
    double _alpha;
};
#endif
