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

// clang-format off
#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/tensors.hpp>
#include <boost/any.hpp>
#include <miopen/conv/invokers/cellfft.hpp>
#include "../cellfft/include/cellfft_get_kernel.hpp"

static void get_solution_cellfft( miopen::solver::ConvSolution& solution, const miopen::cellfft::cellfft_param_t& p, const std::string& file_name )
{
    solution.construction_params.push_back(miopen::cellfft::get_kernel_cgemm(p,file_name));
    if(p.dir!=2)
    {
        if((p.grid_x|p.grid_y)>1){
            solution.construction_params.push_back(miopen::cellfft::get_kernel_r2c_grid(p,file_name));
            solution.construction_params.push_back(miopen::cellfft::get_kernel_r2c_b(p,file_name));
            solution.construction_params.push_back(miopen::cellfft::get_kernel_c2r_grid(p,file_name,0));
        } else {
            solution.construction_params.push_back(miopen::cellfft::get_kernel_r2c_a(p,file_name,0));
            solution.construction_params.push_back(miopen::cellfft::get_kernel_r2c_b(p,file_name));
            solution.construction_params.push_back(miopen::cellfft::get_kernel_c2r(p,file_name,0));
        }
    }
    else
    {
        if((p.grid_x|p.grid_y)>1){
            solution.construction_params.push_back(miopen::cellfft::get_kernel_r2c_xgrad_a(p,file_name));
            solution.construction_params.push_back(miopen::cellfft::get_kernel_r2c_xgrad_b(p,file_name));
        } else {
            solution.construction_params.push_back(miopen::cellfft::get_kernel_r2c_a(p,file_name,1));
            solution.construction_params.push_back(miopen::cellfft::get_kernel_r2c_b(p,file_name));
        }
        solution.construction_params.push_back(miopen::cellfft::get_kernel_c2r_grad(p,file_name));
    }
}

namespace miopen {
namespace solver {
bool ConvCellfft::IsApplicable( const ConvolutionContext& ctx ) const
{
    const auto name=ctx.GetStream().GetDeviceName();
    if(name!="gfx900"||name!="gfx906") return false;
    if((ctx.kernel_stride_w|ctx.kernel_stride_h|ctx.kernel_dilation_w|ctx.kernel_dilation_h|ctx.group_counts)!=1)
        return false;
    return (ctx.Is2d()&&ctx.IsFp32()&&(ctx.in_layout=="NCHW")&&(ctx.bias==0));
}
size_t ConvCellfft::GetWorkspaceSize( const ConvolutionContext& ctx ) const
{
    if(!ctx.direction.IsBackwardWrW()){
        return cellfft::get_auxbuf_size(ctx);
    } else {
        return cellfft::get_auxbuf_size_grad(ctx);
    }
}
ConvSolution ConvCellfft::GetSolution( const ConvolutionContext& ctx ) const
{
    ConvSolution solution;
    cellfft::cellfft_param_t params{};
    const std::string file_name="cellfft_"+ctx.GetStream().GetDeviceName()+".hsaco";
    if(!ctx.direction.IsBackwardWrW()){
        cellfft::build_cellfft_params( params, ctx );
    } else {
        cellfft::build_cellfft_params_grad( params, ctx );
    }
    solution.workspce_sz=cellfft::get_auxbuf_size(params);
    get_solution_cellfft( solution, params, file_name );
    solution.invoker_factory=conv::MakeCellfftInvokerFactory( params, 1.f );
    return solution;
}
} // namespace solver
} // namespace miopen
// clang-format on
