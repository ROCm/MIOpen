/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/mlo_internal.hpp>

void mlo_construct_neuron::mloConstruct()
{
    _hw_wave_sz = 64;

    const size_t read_unit = 4;

    size_t map_size = static_cast<size_t>(_search_params.problem.in_width) *
                      _search_params.problem.in_height * _search_params.problem.n_inputs *
                      _search_params.problem.batch_sz;
    size_t map_size_aligned = (map_size + read_unit - 1) / read_unit;
    size_t N_PIXS_OFF       = map_size - (map_size / read_unit) * read_unit;

    size_t glbl_wk = map_size_aligned;

    _grp_tile0 =
        std::min(static_cast<int>((glbl_wk + _hw_wave_sz - 1) / _hw_wave_sz) * _hw_wave_sz, 256);
    _grp_tile1 = 1;

    _comp_options =
        std::string(" -DMIOPEN_NRN_GROUP_SZ0=") +
        std::to_string(static_cast<long long>(_grp_tile0)) +
        std::string(" -DMIOPEN_NRN_GROUP_SZ1=") +
        std::to_string(static_cast<long long>(_grp_tile1)) + std::string(" -DMIOPEN_NRN_OP_ID=") +
        std::to_string(static_cast<long long>(_neuron_type)) +
        std::string(" -DMIOPEN_N_PIXS_OFF=") + std::to_string(static_cast<long long>(N_PIXS_OFF)) +
        std::string(" -DMIOPEN_MAP_SZ=") + std::to_string(static_cast<long long>(map_size)) +
        std::string(" -DMIOPEN_MAP_SZ_ALIGNED=") +
        std::to_string(static_cast<long long>(map_size_aligned)) +
        std::string(" -DMIOPEN_READ_UNIT=") + std::to_string(static_cast<long long>(read_unit)) +
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
