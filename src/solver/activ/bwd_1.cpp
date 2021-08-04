/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/activ/solvers.hpp>

#include <miopen/activ/invoke_params.hpp>
#include <miopen/activ/problem_description.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>

#include <cassert>

namespace miopen {

namespace solver {

namespace activ {

bool ActivBwdSolver1::IsApplicable(const ExecutionContext& context,
                                   const miopen::activ::ProblemDescription& problem) const
{
    if(problem.GetDirection() != miopen::activ::Direction::Backward)
        return false;

    const auto x_elem_sz = problem.GetXDesc().GetElementSize();
    const auto y_elem_sz = problem.GetYDesc().GetElementSize();

    if(x_elem_sz != y_elem_sz)
        return false;

    // Todo: probably fix "the rest" logic here
    return !ActivFwdSolver1{}.IsApplicable(context, problem);
}

ConvSolution ActivBwdSolver1::GetSolution(const ExecutionContext&,
                                          const miopen::activ::ProblemDescription& problem) const
{
    const auto& xDesc  = problem.GetXDesc();
    const auto& yDesc  = problem.GetXDesc();
    const auto& dxDesc = problem.GetDXDesc();
    const auto& dyDesc = problem.GetDXDesc();

    int ndOut       = 1;
    int cdOut       = 1;
    int hdOut       = 1;
    int wdOut       = 1;
    int ndOutStride = 0;
    int cdOutStride = 0;
    int hdOutStride = 0;
    int wdOutStride = 0;

    if(dyDesc.GetSize() == 4)
    {
        std::tie(ndOut, cdOut, hdOut, wdOut)                         = tien<4>(dyDesc.GetLengths());
        std::tie(ndOutStride, cdOutStride, hdOutStride, wdOutStride) = tien<4>(dyDesc.GetStrides());
    }
    else if(dyDesc.GetSize() < 4 && dyDesc.GetSize() > 0)
    {
        auto tensor_size = dyDesc.GetSize();
        switch(tensor_size)
        {
        case 1:
            std::tie(wdOut)       = tien<1>(dyDesc.GetLengths());
            std::tie(wdOutStride) = tien<1>(dyDesc.GetStrides());
            ndOutStride           = wdOut * wdOutStride;
            cdOutStride           = wdOut * wdOutStride;
            hdOutStride           = wdOut * wdOutStride;
            break;
        case 2:
            std::tie(hdOut, wdOut)             = tien<2>(dyDesc.GetLengths());
            std::tie(hdOutStride, wdOutStride) = tien<2>(dyDesc.GetStrides());
            ndOutStride                        = hdOut * hdOutStride;
            cdOutStride                        = hdOut * hdOutStride;
            break;
        case 3:
            std::tie(cdOut, hdOut, wdOut)                   = tien<3>(dyDesc.GetLengths());
            std::tie(cdOutStride, hdOutStride, wdOutStride) = tien<3>(dyDesc.GetStrides());
            ndOutStride                                     = cdOut * cdOutStride;
            break;
        default: assert(false);
        }
    }
    else
    {
        MIOPEN_THROW("activation does not support tensor size larger than 3 or smaller than 1");
    }

    int nOut       = 1;
    int cOut       = 1;
    int hOut       = 1;
    int wOut       = 1;
    int nOutStride = 0;
    int cOutStride = 0;
    int hOutStride = 0;
    int wOutStride = 0;

    if(yDesc.GetSize() == 4)
    {
        std::tie(nOut, cOut, hOut, wOut)                         = tien<4>(yDesc.GetLengths());
        std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tien<4>(yDesc.GetStrides());
    }
    else if(yDesc.GetSize() < 4 && yDesc.GetSize() > 0)
    {
        auto tensor_size = yDesc.GetSize();
        switch(tensor_size)
        {
        case 1:
            std::tie(wOut)       = tien<1>(yDesc.GetLengths());
            std::tie(wOutStride) = tien<1>(yDesc.GetStrides());
            nOutStride           = wOut * wOutStride;
            cOutStride           = wOut * wOutStride;
            hOutStride           = wOut * wOutStride;
            break;
        case 2:
            std::tie(hOut, wOut)             = tien<2>(yDesc.GetLengths());
            std::tie(hOutStride, wOutStride) = tien<2>(yDesc.GetStrides());
            nOutStride                       = hOut * hOutStride;
            cOutStride                       = hOut * hOutStride;
            break;
        case 3:
            std::tie(cOut, hOut, wOut)                   = tien<3>(yDesc.GetLengths());
            std::tie(cOutStride, hOutStride, wOutStride) = tien<3>(yDesc.GetStrides());
            nOutStride                                   = cOut * cOutStride;
            break;
        default: assert(false);
        }
    }
    else
    {
        MIOPEN_THROW("Activation does not support tensor dimensions larger than 3 or "
                     "smaller than 1");
    }

    int ndIn       = 1;
    int cdIn       = 1;
    int hdIn       = 1;
    int wdIn       = 1;
    int ndInStride = 0;
    int cdInStride = 0;
    int hdInStride = 0;
    int wdInStride = 0;

    if(dxDesc.GetSize() == 4)
    {
        std::tie(ndIn, cdIn, hdIn, wdIn)                         = tien<4>(dxDesc.GetLengths());
        std::tie(ndInStride, cdInStride, hdInStride, wdInStride) = tien<4>(dxDesc.GetStrides());
    }
    else if(dxDesc.GetSize() < 4 && dxDesc.GetSize() > 0)
    {
        auto tensor_size = dxDesc.GetSize();
        switch(tensor_size)
        {
        case 1:
            std::tie(wdIn)       = tien<1>(dxDesc.GetLengths());
            std::tie(wdInStride) = tien<1>(dxDesc.GetStrides());
            ndInStride           = wdIn * wdInStride;
            cdInStride           = wdIn * wdInStride;
            hdInStride           = wdIn * wdInStride;
            break;
        case 2:
            std::tie(hdIn, wdIn)             = tien<2>(dxDesc.GetLengths());
            std::tie(hdInStride, wdInStride) = tien<2>(dxDesc.GetStrides());
            ndInStride                       = hdIn * hdInStride;
            cdInStride                       = hdIn * hdInStride;
            break;
        case 3:
            std::tie(cdIn, hdIn, wdIn)                   = tien<3>(dxDesc.GetLengths());
            std::tie(cdInStride, hdInStride, wdInStride) = tien<3>(dxDesc.GetStrides());
            ndInStride                                   = cdIn * cdInStride;
            break;
        default: assert(false);
        }
    }
    else
    {
        MIOPEN_THROW("Activation does not support tensor dimensions larger than 3 or "
                     "smaller than 1");
    }

    int nIn       = 1;
    int cIn       = 1;
    int hIn       = 1;
    int wIn       = 1;
    int nInStride = 0;
    int cInStride = 0;
    int hInStride = 0;
    int wInStride = 0;

    if(xDesc.GetSize() == 4)
    {
        std::tie(nIn, cIn, hIn, wIn)                         = tien<4>(xDesc.GetLengths());
        std::tie(nInStride, cInStride, hInStride, wInStride) = tien<4>(xDesc.GetStrides());
    }
    else if(xDesc.GetSize() < 4 && xDesc.GetSize() > 0)
    {
        auto tensor_size = xDesc.GetSize();
        switch(tensor_size)
        {
        case 1:
            std::tie(wIn)       = tien<1>(xDesc.GetLengths());
            std::tie(wInStride) = tien<1>(xDesc.GetStrides());
            nInStride           = wIn * wInStride;
            cInStride           = wIn * wInStride;
            hInStride           = wIn * wInStride;
            break;
        case 2:
            std::tie(hIn, wIn)             = tien<2>(xDesc.GetLengths());
            std::tie(hInStride, wInStride) = tien<2>(xDesc.GetStrides());
            nInStride                      = hIn * hInStride;
            cInStride                      = hIn * hInStride;
            break;
        case 3:
            std::tie(cIn, hIn, wIn)                   = tien<3>(xDesc.GetLengths());
            std::tie(cInStride, hInStride, wInStride) = tien<3>(xDesc.GetStrides());
            nInStride                                 = cIn * cInStride;
            break;
        default: assert(false);
        }
    }
    else
    {
        MIOPEN_THROW("Activation does not support tensor dimensions larger than 3 or "
                     "smaller than 1");
    }

    int activ_mode = problem.GetActivDesc().GetMode();

    constexpr const auto hw_wave_sz  = 64;
    constexpr const size_t read_unit = 4;

    const size_t map_size       = wIn * hIn * nIn * cIn;
    const auto map_size_aligned = (map_size + read_unit - 1) / read_unit;
    const auto N_PIXS_OFF       = map_size - (map_size / read_unit) * read_unit;

    const auto glbl_wk = map_size_aligned;

    const auto grp_tile0 =
        std::min(static_cast<int>((glbl_wk + hw_wave_sz - 1) / hw_wave_sz) * hw_wave_sz, 256);
    const auto grp_tile1 = 1;

    auto compiler_options = KernelBuildParameters{
        {"MIOPEN_N_IN", nIn},
        {"MIOPEN_C_IN", cIn},
        {"MIOPEN_H_IN", hIn},
        {"MIOPEN_W_IN", wIn},
        {"MIOPEN_N_IN_STRIDE", nInStride},
        {"MIOPEN_C_IN_STRIDE", cInStride},
        {"MIOPEN_H_IN_STRIDE", hInStride},
        {"MIOPEN_W_IN_STRIDE", wInStride},
        {"MIOPEN_N_OUT", nOut},
        {"MIOPEN_C_OUT", cOut},
        {"MIOPEN_H_OUT", hOut},
        {"MIOPEN_W_OUT", wOut},
        {"MIOPEN_N_OUT_STRIDE", nOutStride},
        {"MIOPEN_C_OUT_STRIDE", cOutStride},
        {"MIOPEN_H_OUT_STRIDE", hOutStride},
        {"MIOPEN_W_OUT_STRIDE", wOutStride},
        {"MIOPEN_N_DIN", ndIn},
        {"MIOPEN_C_DIN", cdIn},
        {"MIOPEN_H_DIN", hdIn},
        {"MIOPEN_W_DIN", wdIn},
        {"MIOPEN_N_DIN_STRIDE", ndInStride},
        {"MIOPEN_C_DIN_STRIDE", cdInStride},
        {"MIOPEN_H_DIN_STRIDE", hdInStride},
        {"MIOPEN_W_DIN_STRIDE", wdInStride},
        {"MIOPEN_N_DOUT", ndOut},
        {"MIOPEN_C_DOUT", cdOut},
        {"MIOPEN_H_DOUT", hdOut},
        {"MIOPEN_W_DOUT", wdOut},
        {"MIOPEN_N_DOUT_STRIDE", ndOutStride},
        {"MIOPEN_C_DOUT_STRIDE", cdOutStride},
        {"MIOPEN_H_DOUT_STRIDE", hdOutStride},
        {"MIOPEN_W_DOUT_STRIDE", wdOutStride},
        {"MIOPEN_IN_BLOCK_SZ", cIn * hIn * wIn},
        {"MIOPEN_OUT_BLOCK_SZ", cOut * hOut * wOut},
        {"MIOPEN_DIN_BLOCK_SZ", cdIn * hdIn * wdIn},
        {"MIOPEN_DOUT_BLOCK_SZ", cdOut * hdOut * wdOut},
        {"MIOPEN_NRN_GROUP_SZ0", grp_tile0},
        {"MIOPEN_NRN_GROUP_SZ1", grp_tile1},
        {"MIOPEN_NRN_OP_ID", static_cast<int>(activ_mode)},
        {"MIOPEN_N_PIXS_OFF", N_PIXS_OFF},
        {"MIOPEN_MAP_SZ", map_size},
        {"MIOPEN_MAP_SZ_ALIGNED", map_size_aligned},
        {"MIOPEN_READ_UNIT", read_unit},
    };

    if(problem.GetXDesc().GetType() == miopenFloat && problem.GetYDesc().GetType() == miopenFloat)
    {
        compiler_options.Define("MIOPEN_USE_FP32", 1);
        compiler_options.Define("MIOPEN_USE_FP16", 0);
    }
    else if(problem.GetXDesc().GetType() == miopenHalf &&
            problem.GetYDesc().GetType() == miopenHalf)
    {
        compiler_options.Define("MIOPEN_USE_FP32", 0);
        compiler_options.Define("MIOPEN_USE_FP16", 1);
    }
    else
    {
        MIOPEN_LOG_E("Unsupported data types configuration: "
                     << miopen::GetDataTypeName(problem.GetXDesc().GetType()) << "x"
                     << miopen::GetDataTypeName(problem.GetYDesc().GetType()));
        return {miopenStatusNotImplemented};
    }

    auto solution = ConvSolution{};

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenNeuron.cl";
        kernel.kernel_name = "MIOpenNeuronBwd";

        kernel.l_wk.push_back(grp_tile0);
        kernel.l_wk.push_back(grp_tile1);
        kernel.l_wk.push_back(1);

        kernel.g_wk.push_back(glbl_wk);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        kernel.comp_options = compiler_options.GenerateFor(kbp::OpenCL{});

        solution.construction_params.emplace_back(std::move(kernel));
    }

    solution.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& invoke_params) {
            const auto kernel  = handle.Run(kernels.front());
            const auto& params = invoke_params.CastTo<miopen::activ::BwdInvokeParams>();

            visit_float(xDesc.GetType(), [&](auto as_float) {
                auto f_activ_alpha = as_float(params.alpha);
                auto f_activ_beta  = as_float(params.beta);
                auto f_activ_gamma = as_float(params.gamma);
                auto f_diff_scale  = f_activ_beta * f_activ_gamma;

                kernel(params.dx,
                       params.dy,
                       params.x,
                       params.y,
                       as_float(f_diff_scale),
                       as_float(f_activ_gamma),
                       as_float(f_activ_beta),
                       as_float(f_activ_alpha),
                       static_cast<long long>(params.dx_offset),
                       static_cast<long long>(params.dy_offset),
                       static_cast<long long>(params.x_offset),
                       static_cast<long long>(params.y_offset));
            });
        };
    };

    return solution;
}

} // namespace activ

} // namespace solver

} // namespace miopen
