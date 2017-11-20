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

#ifndef GUARD_MIOPEN_LEGACY_EXHAUSTIVE_SEARCH_HPP
#define GUARD_MIOPEN_LEGACY_EXHAUSTIVE_SEARCH_HPP

#include <miopen/config.h>
#include <miopen/serializable.hpp>
#include <iostream>

namespace miopen {
namespace solver {

struct LegacyPerformanceConfig : Serializable<LegacyPerformanceConfig>
{
    int grp_tile1       = 0;
    int grp_tile0       = 0;
    int in_tile1        = 0;
    int in_tile0        = 0;
    int out_pix_tile1   = 0;
    int out_pix_tile0   = 0;
    int n_out_pix_tiles = 0;
    int n_in_data_tiles = 0;
    int n_stacks        = 0;

    template <class Solution>
    void CopyTo(Solution& iud) const
    {
        iud.grp_tile0       = grp_tile0;
        iud.grp_tile1       = grp_tile1;
        iud.in_tile0        = in_tile0;
        iud.in_tile1        = in_tile1;
        iud.out_pix_tile0   = out_pix_tile0;
        iud.out_pix_tile1   = out_pix_tile1;
        iud.n_out_pix_tiles = n_out_pix_tiles;
        iud.n_in_data_tiles = n_in_data_tiles;
        iud.n_stacks        = n_stacks;
    }

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.grp_tile1, "temp.grp_tile1");
        f(self.grp_tile0, "temp.grp_tile0");
        f(self.in_tile1, "temp.in_tile1");
        f(self.in_tile0, "temp.in_tile0");
        f(self.out_pix_tile1, "temp.out_pix_tile1");
        f(self.out_pix_tile0, "temp.out_pix_tile0");
        f(self.n_out_pix_tiles, "temp.n_out_pix_tiles");
        f(self.n_in_data_tiles, "temp.n_in_data_tiles");
        f(self.n_stacks, "temp.n_stacks");
    }

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    bool LegacyDeserialize(const std::string& from);
#endif
};
} // namespace solver
} // namespace miopen

#endif
