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
#include <regex>

#if MIOPEN_USE_MIOPENGEMM
namespace miopen {

// so that MIOpen works whether or not recent MIOpenGEMM changes pulled:
// convert size_t and ulong kernel function parameters to unsigned.
namespace tempfix {
void set_offsets_to_uint(std::string& clstr)
{
    auto get_target = [](std::string inttype, char x) {
        std::stringstream ss;
        ss << "const " << inttype << ' ' << std::string(1, x) << "_offset";
        return std::regex(ss.str());
    };

    for(char x : {'a', 'b', 'c'})
    {
        std::string replacement = "const unsigned " + std::string(1, x) + "_offset";
        for(auto inttype : {"size_t", "ulong"})
        {
            clstr = std::regex_replace(clstr, get_target(inttype, x), replacement);
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

    // debug
    std::cout << __func__ << ": 1st kernel: " << kernel_name << std::endl;

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

        // debug
        std::cout << __func__ << ": 2nd kernel: " << beta_kernel_name << std::endl;
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
    std::cout << __func__ << ": alpha " << alpha << ", beta " << beta << std::endl;

    std::string network_config = tgg.get_networkconfig_string();

    if(beta_kern_req)
    {
        handle.GetKernel(algorithm_name + "_beta", network_config)(c, c_offset, beta);
        handle.GetKernel(algorithm_name,
                         network_config)(a, a_offset, b, b_offset, c, c_offset, alpha);

        // debug
        {
            const auto& kernel = handle.GetKernel(algorithm_name + "_beta", network_config);
            std::cout << __func__ << ": 1st kernel: " << kernel.GetName() << std::endl;
        }
        {
            const auto kernel = handle.GetKernel(algorithm_name, network_config);
            std::cout << __func__ << ": 2nd kernel: " << kernel.GetName() << std::endl;
        }
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

        // debug
        {
            const auto kernel = handle.GetKernel(algorithm_name, network_config);
            std::cout << __func__ << ": 1st kernel: " << kernel.GetName() << std::endl;
        }
    }
}

} // namespace miopen

namespace miopen {

void GemmGeometry::RunGemmSimple(Handle& handle,
                                 ConstData_t a,
                                 ConstData_t b,
                                 Data_t c,
                                 int a_offset,
                                 int b_offset,
                                 int c_offset) const
{
    std::string network_config = tgg.get_networkconfig_string();

#if 1 // debug
    {
        std::cout << __func__ << ": alpha " << alpha << ", beta " << beta << std::endl;

        auto const& kernels = handle.GetKernels(algorithm_name, network_config);

        for(const auto& k : kernels)
        {
            std::cout << __func__ << ": kernel name: " << k.GetName() << std::endl;
        }
    }
#endif

    if(handle.GetKernels(algorithm_name, network_config).empty())
    {
        // TODO: a_offset, b_offse, c_offset are not passed in to FindSolution, may result in memory
        // out-of-bound
        FindSolutionTmp(.003, handle, a, b, c, false);
    }

    RunGemmTmp(handle, a, b, c, a_offset, b_offset, c_offset);
}

void GemmGeometry::FindSolutionTmp(float time,
                                   Handle& handle,
                                   ConstData_t a,
                                   ConstData_t b,
                                   Data_t c,
                                   bool enforce_determinism) const
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

    std::cout << __func__ << ": alpha " << alpha << ", beta " << beta << std::endl;

    // chao : this could be kernel for
    //   kernel_0: c = alpha * a * b + c, (if beta == 1) or
    //   kernel_1: c = alpha * a * b + beta * c, (if beta != 1) or
    //   kernel_2: c = beta * c (needs to work together with kernel_0)
    handle.AddKernel(algorithm_name, network_config, kernel_clstring, kernel_name, vld, vgd, "", 0);

    //
    {
        std::cout << __func__ << ": after added 1st kernel: " << kernel_name << std::endl;

        const auto& kernels = handle.GetKernels(algorithm_name, network_config);

        for(const auto& k : kernels)
        {
            std::cout << __func__ << ": kernel name: " << k.GetName() << std::endl;
        }
    }

    // jn : case where the beta kernel is part of the solution
    if(soln.v_tgks.size() == 2 && !miopen::float_equal(beta, 1))
    {
        std::string beta_program_name = soln.v_tgks[0].kernstr;
        tempfix::set_offsets_to_uint(beta_program_name);

        std::string beta_kernel_name = soln.v_tgks[0].fname;
        local_work_size              = soln.v_tgks[0].local_work_size;
        global_work_size             = soln.v_tgks[0].global_work_size;

        vld[0] = local_work_size;
        vgd[0] = global_work_size;

        // chao : this is the kernel for
        //   kernel_0: c = alpha * a * b + c (needs to work with kernel_2)
        handle.AddKernel(
            algorithm_name,
            network_config, // jn : different network_configs require different beta kernels
            beta_program_name,
            beta_kernel_name,
            vld,
            vgd,
            "",
            1);

        // debug
        {
            std::cout << __func__ << ": after added 2nd kernel: " << beta_kernel_name << std::endl;

            const auto& kernels = handle.GetKernels(algorithm_name, network_config);

            for(const auto& k : kernels)
            {
                std::cout << __func__ << ": kernel name: " << k.GetName() << std::endl;
            }
        }
    }
}

void GemmGeometry::RunGemmTmp(Handle& handle,
                              ConstData_t a,
                              ConstData_t b,
                              Data_t c,
                              int a_offset,
                              int b_offset,
                              int c_offset) const
{
    std::string network_config = tgg.get_networkconfig_string();

    const auto& kernels = handle.GetKernels(algorithm_name, network_config);

#if 1 // debug
    {
        std::cout << __func__ << ": alpha " << alpha << ", beta " << beta << std::endl;

        for(const auto& k : kernels)
        {
            std::cout << __func__ << ": kernel name: " << k.GetName() << std::endl;
        }
    }
#endif

    if(kernels.size() == 2)
    {
        // c = beta * c
        std::cout << __func__ << ": " << kernels[1].GetName() << ": c = beta * c" << std::endl;
        kernels[1](c, c_offset, beta);

        // c = alpha * a * b + c
        std::cout << __func__ << ": " << kernels[0].GetName() << ": c = alpha* a * b + c"
                  << std::endl;
        kernels[0](a, a_offset, b, b_offset, c, c_offset, alpha);
    }
    else if(kernels.size() == 1)
    {
        std::string kernel_name = kernels[0].GetName();
        if(kernel_name == "miog_alphaab")
        {
            // c = alpha * a * b + c
            std::cout << __func__ << ": " << kernels[0].GetName() << ": c = alpha* a * b + c"
                      << std::endl;
            kernels[0](a, a_offset, b, b_offset, c, c_offset, alpha);
        }
        else if(kernel_name == "miog_betac_alphaab")
        {
            // c = alpha * a * b + beta * c
            std::cout << __func__ << ": " << kernels[0].GetName() << ": c = alpha* a * b + beta * c"
                      << std::endl;
            kernels[0](a, a_offset, b, b_offset, c, c_offset, alpha, beta);
        }
        else
        {
            MIOPEN_THROW("wrong MIOpenGEMM kernel");
        }
    }
    else
    {
        MIOPEN_THROW("unable to get correct MIOpenGEMM kenerls");
    }
}

} // namespace miopen

namespace miopen {

// TODO: doesn't support offset to A, B, C yet
void FindMiopengemmSolution(Handle& handle,
                            const MIOpenGEMM::Geometry& mgg,
                            ConstData_t A,
                            ConstData_t B,
                            Data_t C,
                            float time,
                            bool enforce_determinism)
{
#if MIOPEN_BACKEND_OPENCL
    // jn : print search results to terminal
    bool miopengemm_verbose = false;

    // jn : print warning messages when the returned kernel(s) might be sub-optimal
    bool miopengemm_warnings = false;

    // jn : find with no workspace
    MIOpenGEMM::Solution soln = MIOpenGEMM::find(time,
                                                 handle.GetStream(),
                                                 A,
                                                 B,
                                                 C,
                                                 enforce_determinism,
                                                 mgg,
                                                 miopengemm_verbose,
                                                 miopengemm_warnings);
#else
    (void)A;
    (void)B;
    (void)C;
    (void)time;
    (void)enforce_determinism;
    MIOpenGEMM::Solution soln = MIOpenGEMM::get_default(mgg);
#endif
    // jn : the main kernel is at the back of the solution vector
    std::string kernel_clstring = soln.v_tgks.back().kernstr;
    tempfix::set_offsets_to_uint(kernel_clstring);

    std::string kernel_name = soln.v_tgks.back().fname;
    size_t local_work_size  = soln.v_tgks.back().local_work_size;
    size_t global_work_size = soln.v_tgks.back().global_work_size;

    std::vector<size_t> vld{local_work_size, 1, 1};
    std::vector<size_t> vgd{global_work_size, 1, 1};

    //
    const std::string algorithm_name = "MIOpenGEMM";
    const std::string network_config = mgg.get_networkconfig_string();

    // chao : there are 2 possible kernel paths for C = alpha * A * B + beta * C in MIOpenGEMM
    // library
    //   1) kernel_0 : C = alpha * A * B + beta * C
    //   2) kernel_1 : C *= beta
    //      kernel_2 : C += alpha * A * B

    // this kernel could be kernel_0, kernel_1
    handle.AddKernel(algorithm_name, network_config, kernel_clstring, kernel_name, vld, vgd, "", 0);

#if 0
    {
        std::cout << __func__ << ": after added 1st kernel: " << kernel_name << std::endl;

        const auto& kernels = handle.GetKernels(algorithm_name, network_config);

        for(const auto& k : kernels)
        {
            std::cout << __func__ << ": kernel name: " << k.GetName() << std::endl;
        }
    }
#endif

    if(soln.v_tgks.size() == 2)
    // if(soln.v_tgks.size() == 2 && !miopen::float_equal(beta, 1))
    {
        std::string beta_program_name = soln.v_tgks[0].kernstr;
        tempfix::set_offsets_to_uint(beta_program_name);

        std::string beta_kernel_name = soln.v_tgks[0].fname;
        local_work_size              = soln.v_tgks[0].local_work_size;
        global_work_size             = soln.v_tgks[0].global_work_size;

        vld[0] = local_work_size;
        vgd[0] = global_work_size;

        // chao : this is kernel_2: C += alpha * A * B (needs to work with kernel_1)
        handle.AddKernel(
            algorithm_name, network_config, beta_program_name, beta_kernel_name, vld, vgd, "", 1);

#if 0
        {
            std::cout << __func__ << ": after added 2nd kernel: " << beta_kernel_name << std::endl;

            const auto& kernels = handle.GetKernels(algorithm_name, network_config);

            for(const auto& k : kernels)
            {
                std::cout << __func__ << ": kernel name: " << k.GetName() << std::endl;
            }
        }
#endif
    }

#if 1
    {
        const auto& kernels = handle.GetKernels(algorithm_name, network_config);

        if(kernels.size() == 2)
        {
            // C *= beta
            assert(kernels[1].GetName() == "miog_betac");

            // C += alpha * A * B
            assert(kernels[0].GetName() == "miog_alphaab");
        }
        else if(kernels.size() == 1)
        {
            // C = alpha * A * B + beta * C
            assert(kernels[0].GetName() == "miog_betac_alphaab");
        }
        else
        {
            MIOPEN_THROW("unable to get correct MIOpenGEMM kenerls");
        }
    }
#endif
}

void RunMiopengemmSolution(Handle& handle,
                           const MIOpenGEMM::Geometry& mgg,
                           float alpha,
                           ConstData_t A,
                           int a_offset,
                           ConstData_t B,
                           int b_offset,
                           float beta,
                           Data_t C,
                           int c_offset)
{
    const std::string algorithm_name = "MIOpenGEMM";
    const std::string network_config = mgg.get_networkconfig_string();

    const auto& kernels = handle.GetKernels(algorithm_name, network_config);

#if 0
    {
        for(const auto& k : kernels)
        {
            std::cout << __func__ << ": kernel name: " << k.GetName() << std::endl;
        }
    }
#endif

    if(kernels.size() == 2)
    {
        // C *= beta
        assert(kernels[1].GetName() == "miog_betac");
        kernels[1](C, c_offset, beta);

        // C += alpha * A * B
        assert(kernels[0].GetName() == "miog_alphaab");
        kernels[0](A, a_offset, B, b_offset, C, c_offset, alpha);
    }
    else if(kernels.size() == 1)
    {
        // C = alpha * A * B + beta * C
        assert(kernels[0].GetName() == "miog_betac_alphaab");
        kernels[0](A, a_offset, B, b_offset, C, c_offset, alpha, beta);
    }
    else
    {
        MIOPEN_THROW("unable to get correct MIOpenGEMM kenerls");
    }
}

} // namespae miopen
#endif // MIOPEN_USE_MIOPENGEMM
