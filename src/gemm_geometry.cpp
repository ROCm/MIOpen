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
#include <miopen/float_equal.hpp>
#include <miopen/gemm_geometry.hpp>

#if MIOPEN_USE_MIOPENGEMM
namespace miopen {

// so that MIOpen works whether or not recent MIOpenGEMM changes pulled:
// convert size_t and ulong kernel function parameters to unsigned.
namespace tempfix {
void set_offsets_to_uint(std::string& clstr)
{

    for(char x : {'a', 'b', 'c'})
    {
        std::string replacement = "const unsigned " + std::string(1, x) + "_offset,";
        for(auto inttype : {"size_t", "ulong"})
        {
            std::string cmpstr =
                "const " + std::string(inttype) + ' ' + std::string(1, x) + "_offset,";
            auto pos = clstr.find(cmpstr);
            if(pos != std::string::npos)
            {
                clstr.replace(pos, cmpstr.size(), replacement);
                break;
            }
        }
    }
}
} // namespace tempfix

void GemmGeometry::EnableBetaKernel(bool enable) { beta_kern_req = enable; }

void GemmGeometry::FindSolution(
    float time, Handle& handle, ConstData_t a, ConstData_t b, Data_t c, bool enforce_determinism)
{

#if MIOPEN_BACKEND_OPENCL
    // jn : print search results to terminal
    bool miopengemm_verbose = false;

    // jn : print warning messages when the returned kernel(s) might be sub-optimal
    bool miopengemm_warnings = false;

    // jn : find with no workspace
    MIOpenGEMM::Solution soln = MIOpenGEMM::find(time,
                                                 handle.GetStream(),
                                                 a,
                                                 b,
                                                 c,
                                                 enforce_determinism,
                                                 tgg,
                                                 miopengemm_verbose,
                                                 miopengemm_warnings);
#else
    (void)time;
    (void)a;
    (void)b;
    (void)c;
    (void)enforce_determinism;
    (void)tgg;
    (void)alpha;
    (void)beta;
    MIOpenGEMM::Solution soln = MIOpenGEMM::get_default(tgg);
#endif

    // jn : the main kernel is at the back of the solution vector
    std::string kernel_clstring = soln.v_tgks.back().kernstr;
    tempfix::set_offsets_to_uint(kernel_clstring);

    std::string kernel_name    = soln.v_tgks.back().fname;
    std::string network_config = tgg.get_networkconfig_string();
    size_t local_work_size     = soln.v_tgks.back().local_work_size;
    size_t global_work_size    = soln.v_tgks.back().global_work_size;

    std::vector<size_t> vld{local_work_size, 1, 1};
    std::vector<size_t> vgd{global_work_size, 1, 1};

    handle.AddKernel(algorithm_name, network_config, kernel_clstring, kernel_name, vld, vgd, "");

    if(soln.v_tgks.size() == 2)
    {
        beta_kern_returned = true;
    }

    // jn : case where the beta kernel is part of the solution
    if(soln.v_tgks.size() == 2 && !miopen::float_equal(beta, 1))
    {
        std::string beta_program_name = soln.v_tgks[0].kernstr;
        tempfix::set_offsets_to_uint(beta_program_name);

        std::string beta_kernel_name = soln.v_tgks[0].fname;
        local_work_size              = soln.v_tgks[0].local_work_size;
        global_work_size             = soln.v_tgks[0].global_work_size;

        EnableBetaKernel(true);

        vld[0] = local_work_size;
        vgd[0] = global_work_size;

        handle.AddKernel(
            algorithm_name + "_beta",
            network_config, // jn : different network_configs require different beta kernels
            beta_program_name,
            beta_kernel_name,
            vld,
            vgd,
            "");
    }
    handle.geo_map[std::make_pair(algorithm_name, network_config)] =
        std::make_unique<GemmGeometry>(*this);
}

void GemmGeometry::RunGemm(Handle& handle,
                           ConstData_t a,
                           ConstData_t b,
                           Data_t c,
                           int a_offset,
                           int b_offset,
                           int c_offset)
{
    std::string network_config = tgg.get_networkconfig_string();

    if(beta_kern_req)
    {
        handle.GetKernel(algorithm_name + "_beta", network_config)(c, c_offset, beta);
        handle.GetKernel(algorithm_name,
                         network_config)(a, a_offset, b, b_offset, c, c_offset, alpha);
    }
    else
    {

        /* jn : the case where a beta kernel
         * was returned, but beta = 1 so it is not
         * needed. Notice: no beta in function sig  */
        if(beta_kern_returned)
        {
            handle.GetKernel(algorithm_name,
                             network_config)(a, a_offset, b, b_offset, c, c_offset, alpha);
        }
        else
        {
            handle.GetKernel(algorithm_name,
                             network_config)(a, a_offset, b, b_offset, c, c_offset, alpha, beta);
        }
    }
}

} // namespace miopen
#endif // MIOPEN_USE_MIOPENGEMM
