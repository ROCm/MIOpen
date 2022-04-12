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
Copyright (c)2016 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

?	Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
?	Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or
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
// to share code with between CPU and GPU

#define MIOPEN
#include <miopen/mlo_internal.hpp>
#include <miopen/mlo_utils.hpp>
#include <miopen/logger.hpp>

// get the previous (less or equal to v) power of 2
int prePow2(int v)
{
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return (v + 1) >> 1;
}

static std::string get_pooling_index_type_name(miopenIndexType_t index_type)
{
    switch(index_type)
    {
    case miopenIndexUint8: {
        return "uchar";
    }
    case miopenIndexUint16: {
        return "ushort";
    }
    case miopenIndexUint32: {
        return "uint";
    }
    case miopenIndexUint64: {
        return "ulong";
    }
    }

    MIOPEN_THROW("not belong to any case");
}

static std::string get_pooling_index_type_max_name(miopenIndexType_t index_type)
{
    switch(index_type)
    {
    case miopenIndexUint8: {
        return "UCHAR_MAX";
    }
    case miopenIndexUint16: {
        return "USHRT_MAX";
    }
    case miopenIndexUint32: {
        return "UINT_MAX";
    }
    case miopenIndexUint64: {
        return "ULONG_MAX";
    }
    }

    MIOPEN_THROW("not belong to any case");
}

int mlo_construct_pooling2D::mloConstruct()
{
    int ret = 0;

    if(isForwardDirection())
    {

        ret = mloConstructFwd();
    }
    else
    {
        ret = mloConstructBwd();
    }

    return (ret);
}

int mlo_construct_pooling2D::mloConstructFwd()
{
    int ret = 0;

    _out_pix_tile0 = 1;
    _out_pix_tile1 = _search_params.out_height <= 8 ? 1 : _search_params.out_height <= 32 ? 4 : 8;
    if(_search_params.out_height > 16 && _search_params.out_height % 32 > 16)
        _out_pix_tile1 =
            std::min(16, std::max(1, prePow2(_out_pix_tile1 * _search_params.kernel_stride_h)));

    _grp_tile0 =
        _search_params.out_width <= 8 ? 8 : (_search_params.out_width % 32 <= 16 ? 16 : 32);
    _grp_tile1 = _search_params.out_height <= 8
                     ? 8
                     : _search_params.out_height < 16
                           ? 16
                           : _search_params.out_height <= 32
                                 ? 32
                                 : _search_params.out_height <= 64 ? 64 : 128;
    _grp_tile1 /= _out_pix_tile1;
    while(_grp_tile0 * _grp_tile1 > 256 && _grp_tile0 > 1)
        _grp_tile0 >>= 1;

    _comp_options =
        std::string(" -DMLO_POOLING_OP_ID=") +
        std::to_string(static_cast<long long>(_pooling_method)) +
        std::string(" -DMLO_POOLING_KERNEL_SZ1=") +
        std::to_string(static_cast<long long>(_search_params.kernel_size_h)) +
        std::string(" -DMLO_POOLING_STRIDE1=") +
        std::to_string(static_cast<long long>(_search_params.kernel_stride_h)) +
        std::string(" -DMLO_POOLING_KERNEL_SZ0=") +
        std::to_string(static_cast<long long>(_search_params.kernel_size_w)) +
        std::string(" -DMLO_POOLING_STRIDE0=") +
        std::to_string(static_cast<long long>(_search_params.kernel_stride_w)) +
        std::string(" -DMLO_POOLING_N_HORIZ_OUT_PIX=") +
        std::to_string(static_cast<long long>(_out_pix_tile0)) +
        std::string(" -DMLO_POOLING_N_VERT_OUT_PIX=") +
        std::to_string(static_cast<long long>(_out_pix_tile1)) +
        std::string(" -DMLO_POOLING_GROUP_SZ0=") +
        std::to_string(static_cast<long long>(_grp_tile0)) +
        std::string(" -DMLO_POOLING_GROUP_SZ1=") +
        std::to_string(static_cast<long long>(_grp_tile1)) +
        std::string(_do_backward ? " -DMLO_POOLING_SAVE_INDEX" : "") +
        std::string(" -DMLO_POOLING_INDEX_TYPE=") + get_pooling_index_type_name(_index_type) +
        std::string(" -DMLO_POOLING_INDEX_MAX=") + get_pooling_index_type_max_name(_index_type) +
        std::string(_do_backward
                        ? (_wsp_index == miopenPoolingWorkspaceIndexImage ? " -DUSE_IMG_INDEX=1"
                                                                          : " -DUSE_IMG_INDEX=0")
                        : "") +
        getGeneralCompOptions();

    int g_wk_width  = ((_search_params.out_width + _grp_tile0 * _out_pix_tile0 - 1) /
                      (_grp_tile0 * _out_pix_tile0));
    int g_wk_height = ((_search_params.out_height + _grp_tile1 * _out_pix_tile1 - 1) /
                       (_grp_tile1 * _out_pix_tile1));

    _l_wk.clear();
    _l_wk.push_back(_grp_tile0);
    _l_wk.push_back(_grp_tile1);
    _l_wk.push_back(1);

    _g_wk.clear();
    _g_wk.push_back(g_wk_width * _grp_tile0);
    _g_wk.push_back(g_wk_height * _grp_tile1);
    _g_wk.push_back(_search_params.n_outputs * _search_params.batch_sz);

    _kernel_file = "MIOpenPooling.cl";

    _kernel_name = "mloPoolingG";

    return (ret);
}

int mlo_construct_pooling2D::mloConstructBwd()
{
    int ret = 0;

    _grp_tile0 = 8;
    _grp_tile1 = 8;

    _out_pix_tile0 = 1;
    _out_pix_tile1 = 1;

    if(_pooling_method == MLO_POOLING_OP_MAX)
    {
        _grp_tile0 =
            _search_params.in_width <= 8
                ? 8
                : (_search_params.in_width <= 16
                       ? 4
                       : (_search_params.in_width <= 24
                              ? 8
                              : (_search_params.in_width <= 32
                                     ? 32
                                     : (_search_params.in_width <= 64
                                            ? 8
                                            : (_search_params.in_width <= 96
                                                   ? 16
                                                   : (_search_params.in_width <= 128 ? 16
                                                                                     : 32))))));
        _grp_tile1 =
            _search_params.in_width <= 8
                ? 8
                : (_search_params.in_width <= 16
                       ? 16
                       : (_search_params.in_width <= 24
                              ? 8
                              : (_search_params.in_width <= 32
                                     ? 4
                                     : (_search_params.in_width <= 64
                                            ? 8
                                            : (_search_params.in_width <= 96
                                                   ? 4
                                                   : (_search_params.in_width <= 128 ? 16 : 4))))));

        _out_pix_tile0 = _search_params.in_width > 8 && _search_params.in_width <= 24 ? 4 : 1;
        _out_pix_tile1 =
            _search_params.in_width <= 24
                ? 1
                : (_search_params.in_width > 64 && _search_params.in_width <= 96 ? 4 : 8);
    }

    _comp_options =
        std::string(" -DMLO_POOLING_OP_ID=") +
        std::to_string(static_cast<long long>(_pooling_method)) +
        std::string(" -DMLO_POOLING_KERNEL_SZ1=") +
        std::to_string(static_cast<long long>(_search_params.kernel_size_h)) +
        std::string(" -DMLO_POOLING_STRIDE1=") +
        std::to_string(static_cast<long long>(_search_params.kernel_stride_h)) +
        std::string(" -DMLO_POOLING_KERNEL_SZ0=") +
        std::to_string(static_cast<long long>(_search_params.kernel_size_w)) +
        std::string(" -DMLO_POOLING_STRIDE0=") +
        std::to_string(static_cast<long long>(_search_params.kernel_stride_w)) +
        std::string(" -DMLO_POOLBWD_N_HORIZ_OUT_PIX=") +
        std::to_string(static_cast<long long>(_out_pix_tile0)) +
        std::string(" -DMLO_POOLBWD_N_VERT_OUT_PIX=") +
        std::to_string(static_cast<long long>(_out_pix_tile1)) +
        std::string(" -DMLO_POOLBWD_GROUP_SZ0=") +
        std::to_string(static_cast<long long>(_grp_tile0)) +
        std::string(" -DMLO_POOLBWD_GROUP_SZ1=") +
        std::to_string(static_cast<long long>(_grp_tile1)) +
        std::string(" -DMLO_POOLING_INDEX_TYPE=") + get_pooling_index_type_name(_index_type) +
        std::string(" -DMLO_POOLING_INDEX_MAX=") + get_pooling_index_type_max_name(_index_type) +
        std::string(_wsp_index == miopenPoolingWorkspaceIndexImage ? " -DUSE_IMG_INDEX=1"
                                                                   : " -DUSE_IMG_INDEX=0") +
        getGeneralCompOptions();

    int g_wk_width  = ((_search_params.in_width + _grp_tile0 * _out_pix_tile0 - 1) /
                      (_grp_tile0 * _out_pix_tile0));
    int g_wk_height = ((_search_params.in_height + _grp_tile1 * _out_pix_tile1 - 1) /
                       (_grp_tile1 * _out_pix_tile1));

    _l_wk.clear();
    _l_wk.push_back(_grp_tile0);
    _l_wk.push_back(_grp_tile1);
    _l_wk.push_back(1);

    _g_wk.clear();
    _g_wk.push_back(g_wk_width * _grp_tile0);
    _g_wk.push_back(g_wk_height * _grp_tile1);
    _g_wk.push_back(_search_params.n_inputs * _search_params.batch_sz);

    _kernel_file = "MIOpenPoolingBwd.cl";
    if(_pooling_method == MLO_POOLING_OP_MAX)
    {
        _kernel_name = "mloPoolingMaxBwd";
    }
    else if(_pooling_method == MLO_POOLING_OP_AVE ||
            _pooling_method == MLO_POOLING_OP_AVE_INCLUSIVE)
    {
        _kernel_name = "mloPoolingAveBwd";
    }
    else
    {
        MIOPEN_LOG_E("Layer: %s. Error: unknowm method"); /// FIXME %s?
        ret = -1;
    }
    return (ret);
}
