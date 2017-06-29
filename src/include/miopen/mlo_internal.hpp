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
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifdef __APPLE__
#include <mach/mach_time.h> // for mach_absolute_time() and friends
#endif

#include <algorithm> // std::find  and std::min std::maxx
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <string>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
// #include <BaseTsd.h>
#include <direct.h>
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
//#ifndef getcwd
// #define getcwd _getcwd
//#endif
typedef unsigned int uint;

#ifndef getcwd
#define getcwd _getcwd
#endif

#else // !WIN32 so Linux and APPLE
#include <climits>
#include <cstdbool>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
using __int64 = long long;
#ifndef fopen_s
#define fopen_s(file, fileName, mode) ((*(file)) = fopen((fileName), (mode))) == NULL
#endif

#endif

using mlo_kernel_info = std::tuple<const std::string,
                                   const std::string,
                                   const std::string,
                                   const std::vector<size_t>,
                                   const std::vector<size_t>>;

#if MIOPEN_BACKEND_OPENCL
#include <miopen/clhelper.hpp>
#include <miopen/ocldeviceinfo.hpp>
#include <miopen/oclkernel.hpp>
#endif
#include <miopen/tensor.hpp>

class mlo_construct_direct2D
{
    public:
    mlo_construct_direct2D(int dir, bool do_bias = false)
    {
        _direction   = dir;
        _do_backward = false;

        //#if !(defined(__APPLE__) || defined(__MACOSX))
        //	_gen_comp_options = std::string(" -cl-std=CL2.0 ");
        //#endif
        _in_tile0 = (_in_width < 12) ? 8 : 16;  //(_in_width < 12) ? 8 : (_in_width < 24 ||
                                                //(_in_width > 32 && _in_width < 48)) ? 16 : 32; //
                                                // size of input data per ALU plane
        _in_tile1 = (_in_height < 12) ? 8 : 16; // (_in_height < 12) ? 8 : (_in_height < 24 ||
                                                // (_in_height > 32 && _in_height < 48)) ? 16 : 32;
                                                // // size of input data per ALU plane

        _grp_tile0 = _in_tile0;
        _grp_tile1 = _in_tile1;

        _out_pix_tile0 = 2; // size of ouptput tile per wk-item (ALU))
        _out_pix_tile1 = 4; //

        _n_out_pix_tiles = 2; // # output pixel tiles per wk-item (ALU)
        _n_in_data_tiles = 4; // # of blocks of different inputs in LDS

        _n_stacks       = 1; // # of diff stacks (part of batch).
        _bias           = (do_bias) ? 1 : 0;
        _pad0           = 1;
        _pad1           = 1;
        _kernel_size0   = 3;
        _kernel_size1   = 3;
        _kernel_stride0 = 1;
        _kernel_stride1 = 1;
        _stream         = nullptr;
        _bot_sz         = 0; // bytes
        _top_sz         = 0; // bytes
        _weights_sz     = 0; // bytes
        _bias_sz        = 0; // bytes

        _workspce_sz = 0;

        _small         = true;
        _copy_input    = false;
        _new_in_width  = 0;
        _new_in_height = 0;
        _new_in_sz     = 0;
    }

    virtual ~mlo_construct_direct2D() = default;
    /*
    * major interface
    * it has to be called only after
    * convolutional parmeters, input, output and weight tesnor have been set
    *
    * constructs compiler option
    *
    * selects kernel file and name
    * covers genrinc forward convolution:
    * arbitrary combination of kerenl sizes, strides
    */

    virtual int mloConstruct();

    /*
    * makes a unique key that represent the current kernel c0onfiguration
    */
    int mloMakeKernelHash(std::string& hash) const;

    /*
    * ontains major configuration parameres:
    * grp_tile1, grp_tile0 - group work size vertically and horizontally
    * in_tile1, in_tile0 - vertical and horizotal size of input data block processed by the group
    * out_pix_tile1, out_pix_tile0 - vertical and horizontal size of output tile process by a single
    * wk-item
    * n_out_pix_tiles - number of output maps processed by a simgle wk-item. that's wk-item
    * processes a stack of n_out_pix_tiles tiles out_pix_tile1 x out_pix_tile0.
    * n_in_data_tiles - number of different input maps kept in LDS per one batch (or stack).
    * n_stacks - number of batches processed by the group
    */
    inline void getConfigParameters(int& grp_tile1,
                                    int& grp_tile0,
                                    int& in_tile1,
                                    int& in_tile0,
                                    int& out_pix_tile1,
                                    int& out_pix_tile0,
                                    int& n_out_pix_tiles,
                                    int& n_in_data_tiles,
                                    int& n_stacks) const
    {

        grp_tile0       = _grp_tile0;
        grp_tile1       = _grp_tile1;
        in_tile0        = _in_tile0;
        in_tile1        = _in_tile1;
        out_pix_tile0   = _out_pix_tile0;
        out_pix_tile1   = _out_pix_tile1;
        n_out_pix_tiles = _n_out_pix_tiles;
        n_in_data_tiles = _n_in_data_tiles;
        n_stacks        = _n_stacks;
    }

    inline void setConfigParameters(int grp_tile1,
                                    int grp_tile0,
                                    int in_tile1,
                                    int in_tile0,
                                    int out_pix_tile1,
                                    int out_pix_tile0,
                                    int n_out_pix_tiles,
                                    int n_in_data_tiles,
                                    int n_stacks)
    {

        _grp_tile0       = grp_tile0;
        _grp_tile1       = grp_tile1;
        _in_tile0        = in_tile0;
        _in_tile1        = in_tile1;
        _out_pix_tile0   = out_pix_tile0;
        _out_pix_tile1   = out_pix_tile1;
        _n_out_pix_tiles = n_out_pix_tiles;
        _n_in_data_tiles = n_in_data_tiles;
        _n_stacks        = n_stacks;
    }

    /*
    * returns parameter values that are compiled in legacy kernels for kernels using them as
    * arguments.
    */
    inline void getCompiledInParameters(int* N, int* C, int* H, int* W, int* K, int* n_groups)
    {
        assert(N && C && H && W && K && n_groups);

        *N        = _batch_sz;
        *C        = _n_inputs;
        *H        = _in_height;
        *W        = _in_width;
        *K        = _n_outputs;
        *n_groups = _stream->GetMaxComputeUnits();
    }

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

    /*
    * get common compiler options
    */
    inline const std::string& getGeneralCompOptions() const { return (_gen_comp_options); }

    /*
    * get info for all kernels of the layer
    * std::string _kernel_name;
    * std::string _kernel_file;
    * std::string _comp_options;
    * std::vector<size_t> _g_wk;
    * std::vector<size_t> _l_wk;
    */

    inline const std::vector<mlo_kernel_info>& getKernelsInfo() const
    {
        return (_mlo_kernels_info);
    }

    /*
    * return direction: true - forward, false - backward
    */
    inline bool isForwardDirection() const { return (_direction == 1); }

    /*
    * get workspace size
    */
    inline size_t getWorkSpaceSzBytes() const { return (_workspce_sz); }
    /*
    *  is bias incuded
    */

    inline bool doBias() const { return (_bias == 1); }

    /*
    * set a number of iteration for thwe wall clock performance meaturement
    */

    inline void setTimerIter(int n_timer_iter) { _n_timer_iter = n_timer_iter; }

    /*
    * set library stream
    */
    inline void setStream(miopen::Handle* stream) { _stream = stream; }

    /*
    * set ocl Kernels path
    */
    inline void setKernelPath(const std::string& kernel_path) { _kernel_path = kernel_path; }

    /*
    * set convolutional parameters
    */
    inline void setConvDescr(int u_padding,
                             int v_padding,
                             int u_stride,
                             int v_stride,
                             int /*u_upstride*/,
                             int /*v_upstride*/
                             )
    {
        _pad1           = u_padding;
        _pad0           = v_padding;
        _kernel_stride0 = u_stride;
        _kernel_stride1 = v_stride;
    }

    /*
    * set weights tensor
    */
    inline void setWeightsDescr(const std::string& layout,
                                const std::string& data_type,
                                int batch,
                                int depth,
                                int height,
                                int width,
                                int batch_stride,
                                int channel_stride,
                                int stride,
                                int w_stride)
    {
        _kernel_size0 = width;
        _kernel_size1 = height;
        int data_len  = (data_type == "FP32" ? 4 : 8);
        size_t size   = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;
        _weights_sz = size;
    }

    /*
    * set output tensor
    */
    inline void setOutputDescr(const std::string& layout,
                               const std::string& data_type,
                               int batch,
                               int depth,
                               int height,
                               int width,
                               int batch_stride,
                               int channel_stride,
                               int stride,
                               int w_stride)
    {
        _batch_sz    = batch;
        int data_len = (data_type == "FP32" ? 4 : 8);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;
        if(_direction)
        {

            _out_width          = width;
            _out_height         = height;
            _n_outputs          = depth;
            _out_batch_stride   = batch_stride;
            _out_channel_stride = channel_stride;
            _out_stride         = stride;
            _top_sz             = size;
            _out_layout         = layout;
            _out_data_type      = data_type;
        }
        else
        {
            _in_width          = width;
            _in_height         = height;
            _n_inputs          = depth;
            _in_batch_stride   = batch_stride;
            _in_channel_stride = channel_stride;
            _in_stride         = stride;
            _bot_sz            = size;
            _in_layout         = layout;
            _in_data_type      = data_type;
            //			_tens_layout = layout;
            //			_tens_data_format = data_type;
        }
    }

    /*
    *  set input tensor
    */

    inline void setInputDescr(const std::string& layout,
                              const std::string& data_type,
                              int batch,
                              int depth,
                              int height,
                              int width,
                              int batch_stride,
                              int channel_stride,
                              int stride,
                              int w_stride)
    {
        _batch_sz    = batch;
        int data_len = (data_type == "FP32" ? 4 : 8);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;
        if(_direction)
        {

            _in_width          = width;
            _in_height         = height;
            _n_inputs          = depth;
            _in_batch_stride   = batch_stride;
            _in_channel_stride = channel_stride;
            _in_stride         = stride;
            _bot_sz            = size;
            _in_layout         = layout;
            _in_data_type      = data_type;
            //			_tens_layout = layout;
            //			_tens_data_format = data_type;
        }
        else
        {
            _out_width          = width;
            _out_height         = height;
            _n_outputs          = depth;
            _out_batch_stride   = batch_stride;
            _out_channel_stride = channel_stride;
            _out_stride         = stride;
            _top_sz             = size;
            _out_layout         = layout;
            _out_data_type      = data_type;
        }

        _bias_sz = (_bias) ? _n_outputs * data_len : 0;
    }

    /*
    * set top tensor
    */
    void setTopDescr(const std::string& layout,
                     const std::string& data_type,
                     int batch,
                     int depth,
                     int height,
                     int width,
                     int batch_stride,
                     int channel_stride,
                     int stride,
                     int w_stride)
    {
        _batch_sz    = batch;
        int data_len = (data_type == "FP32" ? 4 : 8);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;

        _out_width          = width;
        _out_height         = height;
        _n_outputs          = depth;
        _out_batch_stride   = batch_stride;
        _out_channel_stride = channel_stride;
        _out_stride         = stride;
        _top_sz             = size;
        _out_layout         = layout;
        _out_data_type      = data_type;
        _bias_sz            = (_bias) ? _n_outputs * data_len : 0;
    }

    /*
    *  set bot tensor
    */

    void setBotDescr(const std::string& layout,
                     const std::string& data_type,
                     int batch,
                     int depth,
                     int height,
                     int width,
                     int batch_stride,
                     int channel_stride,
                     int stride,
                     int w_stride)
    {
        _batch_sz    = batch;
        int data_len = (data_type == "FP32" ? 4 : 8);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;

        _in_width          = width;
        _in_height         = height;
        _n_inputs          = depth;
        _in_batch_stride   = batch_stride;
        _in_channel_stride = channel_stride;
        _in_stride         = stride;
        _bot_sz            = size;
        _in_layout         = layout;
        _in_data_type      = data_type;
        //			_tens_layout = layout;
        //			_tens_data_format = data_type;
    }

    /*
    * set top df tensor
    */
    void setTopDfDescr(const std::string& layout,
                       const std::string& data_type,
                       int batch,
                       int depth,
                       int height,
                       int width,
                       int batch_stride,
                       int channel_stride,
                       int stride,
                       int w_stride)
    {
        _batch_sz    = batch;
        int data_len = (data_type == "FP32" ? 4 : 8);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;

        _out_df_width          = width;
        _out_df_height         = height;
        _n_outputs             = depth;
        _out_df_batch_stride   = batch_stride;
        _out_df_channel_stride = channel_stride;
        _out_df_stride         = stride;
        _top_df_sz             = size;
        _out_df_layout         = layout;
        _out_df_data_type      = data_type;
    }

    /*
    *  set bot df tensor
    */

    void setBotDfDescr(const std::string& layout,
                       const std::string& data_type,
                       int batch,
                       int depth,
                       int height,
                       int width,
                       int batch_stride,
                       int channel_stride,
                       int stride,
                       int w_stride)
    {
        _batch_sz    = batch;
        int data_len = (data_type == "FP32" ? 4 : 8);
        size_t size  = (layout == "NCHW")
                          ? batch * depth * height * width * data_len
                          : batch * batch_stride * channel_stride * stride * w_stride * data_len;

        _in_df_width          = width;
        _in_df_height         = height;
        _n_inputs             = depth;
        _in_df_batch_stride   = batch_stride;
        _in_df_channel_stride = channel_stride;
        _in_df_stride         = stride;
        _bot_df_sz            = size;
        _in_df_layout         = layout;
        _in_df_data_type      = data_type;
    }

    /*
    *  indicate the need for backward pass
    */
    inline void doBackward(bool do_bwd) { _do_backward = do_bwd; }
    /*
    * need for backward pass?
    */
    inline bool doBackward() const { return (_do_backward); }

    /*
    *  allow the search for the best possible solution
    */
    inline void doSearch(bool do_search) { _search = do_search; }
    /*
    * is search set?
    */
    inline bool doSearch() const { return (_search); }

    /*
    * allow to save the missing configuraion in the search request file for an offline search
    */
    inline void saveSearchRequest(bool save_req) { _save_srch_req = save_req; }
    /*
    * set common compiler options
    */
    inline void setGeneralCompOptions(const std::string& options) { _gen_comp_options += options; }

    // MD: Hack to get the key outside of mlo_internal
    int mloBuildConf_Key(std::string& conf_key) const;

    inline bool doCopyInput() const { return (_copy_input); }

    // MD: Where is this being used?
    void getNewInputDescr(std::string& layout,
                          std::string& data_type,
                          int& batch,
                          int& depth,
                          int& height,
                          int& width,
                          int& batch_stride,
                          int& channel_stride,
                          int& stride,
                          int& w_stride);
    // TEMP
    int mloConstructSP2D();

    size_t setInputDescFromMLDesc(const miopen::TensorDescriptor& input_tensor);
    size_t setOutputDescFromMLDesc(const miopen::TensorDescriptor& output_tensor);
    size_t setWeightDescFromMLDesc(const miopen::TensorDescriptor& weight_tensor);

    bool mloIsCompilerWorkarounds() const;
    bool mloIsFastBinaryWinograd3x3Fwd() const;
    int mloConstructDirect2D_11x11(bool n_passes = false);

    protected:
    bool mloGetConfig();
    int mloSearchDirect2D();
    int mloConstructDirect2DFwd();

    enum rocm_meta_version
    {
        V1,
        V2,
        V3
    };
    bool mloIsAmdOpenclRocm(rocm_meta_version& rmv) const;

    bool mloIsCorrectBinaryWinograd3x3Fwd() const;
    int mloConstructBinaryWinograd3x3Fwd(rocm_meta_version rmv);

    bool mloIsCorrectAsmDirect3x3U() const;
    bool mloIsFastAsmDirect3x3U() const;
    int mloConstructAsmDirect3x3U(rocm_meta_version rmv);

    bool mloIsCorrectAsmDirect5x10u2v2f1() const;
    bool mloIsFastAsmDirect5x10u2v2f1() const;
    int mloConstructAsmDirect5x10u2v2f1(rocm_meta_version rmv);

    bool mloIsCorrectAsmDirect5x10u2v2b1() const;
    bool mloIsFastAsmDirect5x10u2v2b1() const;
    int mloConstructAsmDirect5x10u2v2b1(rocm_meta_version rmv);

    bool mloIsCorrectAsmDirect7x7c3h224w224k64u2v2p3q3f1(rocm_meta_version rmv) const;
    bool mloIsFastAsmDirect7x7c3h224w224k64u2v2p3q3f1() const;
    int mloConstructAsmDirect7x7c3h224w224k64u2v2p3q3f1();

    int mloConstructDirect2DFwdC();
    int mloConstructDirect2D1x1();
    int mloConstructDirect2D3x3();
    int mloConstructDirect2DFwdGen();

    int mloConstructBwd() { return (0); }
    int mloConstructFwd() { return (0); }

    int mloSetConf(const std::string& conf_val);

    //	int mloBuildConf_Key(std::string & conf_key) const;

    int mloSelectDefaultConfig(std::string& conf_val);
    int mloAddConfigReq(const std::string& conf_key) const;
    int mloRemoveConfigReq(const std::string& conf_key) const;
    int mloReadConfigDB(std::map<std::string, std::string>& conf_db) const;
    int mloWriteConfigDB(const std::map<std::string, std::string>& conf_db) const;
    int mloAddConfig(std::string& conf_key, std::string& conf_val) const;
    bool mloSearchConfigInDB(std::string& conf_key, std::string& conf_val) const;

    int mloMeasuredLoop(miopen::Handle* profile_h,
                        Data_t bot_ocl_buf,
                        Data_t top_ocl_buf,
                        Data_t wei_ocl_buf,
                        Data_t bias_ocl_buf,
                        double& processing_time);

    protected:
    int _direction;
    int _pad0;
    int _pad1;
    int _kernel_size0;
    int _kernel_size1;
    int _kernel_stride0;
    int _kernel_stride1;
    int _n_outputs = 0;
    int _n_inputs  = 0;
    int _batch_sz  = 0;

    int _out_width          = 0;
    int _out_height         = 0;
    int _out_batch_stride   = 0;
    int _out_channel_stride = 0;
    int _out_stride         = 0;
    std::string _out_layout;
    std::string _out_data_type;

    int _in_width          = 0;
    int _in_height         = 0;
    int _in_batch_stride   = 0;
    int _in_channel_stride = 0;
    int _in_stride         = 0;
    std::string _in_layout;
    std::string _in_data_type;

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

    // SP

    bool _small;
    bool _copy_input;
    int _new_in_height;
    int _new_in_width;
    int _new_in_batch_stride   = 0;
    int _new_in_channel_stride = 0;
    int _new_in_stride         = 0;
    size_t _new_in_sz;
    bool _do_backward;

    // FIX IT
    //	int _weights_height;
    //	int _weights_stride;
    std::string _weights_layout;
    std::string _weight_data_type;
    //
    //	std::string _tens_layout;
    //	std::string _tens_data_format;

    int _in_tile0        = 0; // size of in-tile in local memory
    int _in_tile1        = 0; // size of in-tile in local memory
    int _grp_tile0       = 0; // total number ALUs per group
    int _grp_tile1       = 0; // total number ALUs per group
    int _out_pix_tile0   = 0; // # of generated pixels per output per wk-item  (ALU)
    int _out_pix_tile1   = 0; // # of generated pixels per output per wk-item  (ALU)
    int _n_out_pix_tiles = 0; // # output pixel tiles per wk-item (ALU)
    int _n_in_data_tiles = 0; // # of blocks of different inputs in LDS
    int _n_stacks        = 0; // # of diff stacks (part of batch).
    int _bias;                // bias calculated inside conv (forward)
    std::string _comp_options;
    std::string _kernel_file;
    std::string _kernel_name;
    std::vector<size_t> _l_wk;
    std::vector<size_t> _g_wk;

    // more than 1 kerenls per stage
    std::vector<mlo_kernel_info> _mlo_kernels_info;

    bool _gen           = false; // genral case
    int _n_timer_iter   = 0;
    int _quiet          = 0;
    bool _search        = false;
    bool _save_srch_req = false;
    std::string _gen_comp_options;
    std::string _kernel_path;
    // local memory size per group
    size_t _dev_local_mem_sz = 0;
    // wave size
    int _hw_wave_sz = 0;
    // cl_queue
    miopen::Handle* _stream;
    size_t _bot_sz;        // bytes
    size_t _top_sz;        // bytes
    size_t _bot_df_sz = 0; // bytes
    size_t _top_df_sz = 0; // bytes
    size_t _weights_sz;    // bytes
    size_t _bias_sz;       // bytes

    size_t _workspce_sz;

    unsigned int _n_groups{};
};

/*
* backward with regard to weights construction
*/

class mlo_construct_BwdWrW2D : public mlo_construct_direct2D
{
    public:
    mlo_construct_BwdWrW2D(int dir, bool do_bias = false) : mlo_construct_direct2D(dir, do_bias) {}

    int mloConstruct() override;
    bool mloIsCompilerWorkarounds() const;
    int mloMultiStep();

    protected:
    int mloConstruct2(bool n_stages = false);
    int mloConstruct53(bool n_stages = false);
    int mloConstruct1x1(bool n_stages = false);
    int mloConstruct1x1Mmap();
    //	int mloConstruct3x3();

    struct PerfParamsAsmDirect3x3WrW
    {
        int limit_wave_cnt;
        int chunk_size;  // 16 or 8. Lower values increase register pressure
        int c_per_wave;  // should be (64 / chunk_size)
        int k_per_wave;  // 1, 2, 4, 8 and chunk_size * k_per_wave <= 64. Higher values increase
                         // register preasure
        int n_per_group; // 1..8 and n_per_group <= batch_size
        int pipe_lines_depth; // 1..8 and pipe_lines_depth <= img_h. Higher values increase register
                              // pressure
        int reverse_inout;    // 0 or 1
        PerfParamsAsmDirect3x3WrW()
            : limit_wave_cnt(0),
              chunk_size(16),
              c_per_wave(4),
              k_per_wave(4),
              n_per_group(1),
              pipe_lines_depth(2),
              reverse_inout(0)
        {
        }
    };
    PerfParamsAsmDirect3x3WrW mloComputePerfParamsAsmDirect3x3WrW() const;
    bool mloIsCorrectAsmDirect3x3WrW() const;
    bool mloIsFastAsmDirect3x3WrW() const;
    int mloConstructAsmDirect3x3WrW();
};

/*
* winograd algorithm
*/

class mlo_construct_winograd : public mlo_construct_direct2D
{
    public:
    mlo_construct_winograd(int dir, bool do_bias = false) : mlo_construct_direct2D(dir, do_bias) {}

    int mloConstruct() override;
};

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1
#define MLO_POOLING_OP_STC 2

class mlo_construct_pooling2D : public mlo_construct_direct2D
{
    public:
    mlo_construct_pooling2D(int dir) : mlo_construct_direct2D(dir)
    {
        _pooling_method = MLO_POOLING_OP_MAX;
        _NAN_option     = 0;
    }

    inline void setPoolingDescr(int pooling_method = MLO_POOLING_OP_MAX,
                                int windowHeight   = 3,
                                int windowWidth    = 3,
                                int padding_v      = 0,
                                int padding_h      = 0,
                                int stride_v       = 2,
                                int stride_h       = 2,
                                int NAN_opt        = 0)
    {
        _pooling_method = pooling_method;
        _pad1           = padding_v;
        _pad0           = padding_h;
        _kernel_size1   = windowHeight;
        _kernel_size0   = windowWidth;
        _kernel_stride1 = stride_v;
        _kernel_stride0 = stride_h;
        _NAN_option     = NAN_opt;
    }

    inline void getPoolingDescr(int& /*pooling_method*/,
                                int& windowHeight,
                                int& windowWidth,
                                int& padding_v,
                                int& padding_h,
                                int& stride_v,
                                int& stride_h,
                                int& NAN_opt) const
    {
        padding_v    = _pad1;
        padding_h    = _pad0;
        windowHeight = _kernel_size1;
        windowWidth  = _kernel_size0;
        stride_v     = _kernel_stride1;
        stride_h     = _kernel_stride0;
        NAN_opt      = _NAN_option;
    }

    inline int getPoolingMethod() const { return (_pooling_method); }
    int mloConstruct() override;

    protected:
    int _pooling_method;
    int _NAN_option;
    int mloConstructFwd();
    int mloConstructBwd();
};

#define MLO_LRN_WITHIN_CHANNEL 0
#define MLO_LRN_ACROSS_CHANNELS 1

class mlo_construct_norm : public mlo_construct_direct2D
{
    public:
    mlo_construct_norm(int dir) : mlo_construct_direct2D(dir) {}

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

    int mloConstruct() override;

    protected:
    int mloConstructFwd();
    int mloConstructBwd();
    int _norm_region  = 0;
    int _norm_area    = 0;
    double _normAlpha = 0.0;
    double _normBeta  = 0.0;
    double _normK     = 0.0;
};

#define MLO_NEURON_PASTHRU 0                       // x
#define MLO_NEURON_LOGISTIC MLO_NEURON_PASTHRU + 1 //	1 / (1 + e^-x)	//Sigmoid
#define MLO_NEURON_TANH MLO_NEURON_LOGISTIC + 1    //	a * tanh( b * x)
#define MLO_NEURON_RELU MLO_NEURON_TANH + 1        //	max(0, x)
#define MLO_NEURON_BRELU MLO_NEURON_RELU + 1       //	min(a, max(0, x))
#define MLO_NEURON_SOFTRELU \
    MLO_NEURON_BRELU + 1                       //	log(1 + e^x)   // bonomial normal log likelihood
#define MLO_NEURON_ABS MLO_NEURON_SOFTRELU + 1 //	abs(x)
#define MLO_NEURON_SQUARE MLO_NEURON_ABS + 1   //	x^2
#define MLO_NEURON_SQR MLO_NEURON_SQUARE + 1   //	sqr(x)
#define MLO_NEURON_LINEAR MLO_NEURON_SQR + 1   //	a + b * x
#define MLO_NEURON_POWER MLO_NEURON_LINEAR + 1 // (a + b * x ) ^power
#define MLO_NEURON_TOTAL MLO_NEURON_POWER + 1

class mlo_construct_neuron : public mlo_construct_direct2D
{
    public:
    mlo_construct_neuron(int dir) : mlo_construct_direct2D(dir)
    {
        _neuron_type = 0;
        _power       = 0;
        _scale       = 1;
        _shift       = 0;
    }

    inline void setNeuronDescr(int neuron_type, double power, double scale, double shift)
    {
        _neuron_type = neuron_type;
        _power       = power;
        _scale       = scale;
        _shift       = shift;
    }

    inline void getNeuronDescr(int& neuron_type, double& power, double& scale, double& shift) const
    {
        neuron_type = _neuron_type;
        power       = _power;
        scale       = _scale;
        shift       = _shift;
    }

    int mloConstruct() override;

    protected:
    int mloConstructFwd();
    int mloConstructBwd();
    int _neuron_type;
    double _power;
    double _scale;
    double _shift;
};

#endif
