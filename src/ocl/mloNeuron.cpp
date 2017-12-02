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

//#define MIOPEN
#include <miopen/mlo_internal.hpp>
#include <miopen/mlo_utils.hpp>

void mlo_construct_neuron::mloConstruct()
{
    _hw_wave_sz = 64;

    const int read_unit = 4;

    size_t map_size = _search_params.in_width * _search_params.in_height * _search_params.n_inputs *
                      _search_params.batch_sz;
    size_t map_size_aligned = (map_size + read_unit - 1) / read_unit;
    int N_PIXS_OFF          = map_size - (map_size / read_unit) * read_unit;

    size_t glbl_wk = map_size_aligned;

    _grp_tile0 =
        std::min(static_cast<int>((glbl_wk + _hw_wave_sz - 1) / _hw_wave_sz) * _hw_wave_sz, 256);
    _grp_tile1 = 1;

    _comp_options =
        std::string(" -DMLO_NRN_GROUP_SZ0=") + std::to_string(static_cast<long long>(_grp_tile0)) +
        std::string(" -DMLO_NRN_GROUP_SZ1=") + std::to_string(static_cast<long long>(_grp_tile1)) +
        std::string(" -DMLO_NRN_OP_ID=") + std::to_string(static_cast<long long>(_neuron_type)) +
        std::string(" -DMLO_N_PIXS_OFF=") + std::to_string(static_cast<long long>(N_PIXS_OFF)) +
        std::string(" -DMLO_MAP_SZ=") + std::to_string(static_cast<long long>(map_size)) +
        std::string(" -DMLO_MAP_SZ_ALIGNED=") +
        std::to_string(static_cast<long long>(map_size_aligned)) +
        std::string(" -DMLO_READ_UNIT=") + std::to_string(static_cast<long long>(read_unit)) +
        getGeneralCompOptions();

    _l_wk.clear();
    _l_wk.push_back(_grp_tile0);
    _l_wk.push_back(_grp_tile1);
    _l_wk.push_back(1);

    _g_wk.clear();
    _g_wk.push_back(glbl_wk);
    _g_wk.push_back(1);
    _g_wk.push_back(1);

    _kernel_file = "MIOpenNeuron.cl";
    _kernel_name = (isForwardDirection()) ? "MIOpenNeuronFwd" : "MIOpenNeuronBwd";
}

int mlo_construct_neuron::mloConstructFwd()
{
    int ret = 0;
    return (ret);
}

int mlo_construct_neuron::mloConstructBwd()
{
    int ret = 0;
    return (ret);
}
