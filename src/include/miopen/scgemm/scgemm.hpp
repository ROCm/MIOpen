/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_SCGEMM_HPP_
#define GUARD_MIOPEN_SCGEMM_HPP_

#include <cstdint>
#include <cstdio>
#include <string>
#include <memory>
#include <vector>
#include <miopen/scgemm/scgemm_op.hpp>
#include <miopen/scgemm/tensorshape.hpp>

namespace miopen {
namespace scgemm {

using scgemm_params_t = std::shared_ptr<char>;

enum scgemm_status_t
{
    scgemm_success                        = 0,
    scgemm_error_invalid_device           = 1,
    scgemm_error_invalid_value            = 2,
    scgemm_error_invalid_binary           = 3,
    scgemm_error_host_memory_alloc_failed = 4
};

enum scgemm_conv_routine_t
{
    sconv_064x016 = 0,
    sconv_064x032 = 1,
    sconv_064x064 = 2,
    sconv_128x032 = 3,
    sconv_128x064 = 4,
    sconv_128x128 = 5,
    sconv_256x032 = 6,
    sconv_256x064 = 7
};

enum scgemm_gemm_routine_t
{
    sgemm_064x016 = 0,
    sgemm_064x032 = 1,
    sgemm_064x064 = 2,
    sgemm_128x064 = 3,
    sgemm_128x128 = 4,
    sgemm_256x032 = 5,
    sgemm_256x064 = 6
};

enum scgemm_op_t
{
    scgemm_fconv = 0,
    scgemm_fgemm = 1,
};

/*
 *! @brief Create 4D tensorshape
 *
 * @param nx  Data width dimension size, supported range [1, 4096] (input)
 * @param ny  Data height dimension size, supported range [1, 4096] (intput)
 * @param bt  Batch size (input)
 * @param nc  Number of channels (input)
 * return     tensorshape
*/
tensorshape_t scgemm_create_tensorshape_4d(uint32_t nx, uint32_t ny, uint32_t bt, uint32_t nc);

/*
 *! @brief Create 5D tensorshape
 *
 * @param nx  Data width dimension size, supported range [1, 512] (input)
 * @param ny  Data height dimension size, supported range [1, 512] (intput)
 * @param nz  Data depth dimension size, supported range [1, 512] (intput)
 * @param bt  Batch size (input)
 * @param nc  Number of channels (input)
 * return     tensorshape
*/
tensorshape_t
scgemm_create_tensorshape_5d(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t bt, uint32_t nc);

/*
 *! @brief Create 4D tensorshape for filter
 *
 * @param nx  Filter width dimension size, supported range [1, 16] (input)
 * @param ny  Filter height dimension size, supported range [1, 16] (intput)
 * @param inc Number of input channels (input)
 * @param onc Number of output channels (input)
 * return     tensorshape
*/
tensorshape_t
scgemm_create_tensorshape_filter_4d(uint32_t nx, uint32_t ny, uint32_t inc, uint32_t onc);

/*
 *! @brief Create 5D tensorshape for filter
 *
 * @param nx  Filter width dimension size, supported range [1, 16] (input)
 * @param ny  Filter height dimension size, supported range [1, 16] (intput)
 * @param nz  Filter depth dimension size, supported range [1, 16] (intput)
 * @param inc Number of input channels (input)
 * @param onc Number of channels (input)
 * return     tensorshape
*/
tensorshape_t scgemm_create_tensorshape_filter_5d(
    uint32_t nx, uint32_t ny, uint32_t nz, uint32_t inc, uint32_t onc);

size_t scgemm_get_tensor_size(const tensorshape_t& shape, uint32_t ng, uint32_t prec);

/*
 *! @brief Returns the extra buffer size required for a particular routine.
 *
 * @param routine    Routine type of the Static Compiled Gemm (input)
 * @param Sd         Tensorshape of input data (intput)
 * @param Sf         Tensorshape of filter (intput)
 * @param strides    Array containing the size of stride (input)
 * @param dilations  Array containing the size of dilation, dilation is starting from 0. (input)
 * return            auxbuf size in number of bytes
*/
size_t scgemm_get_fconv_auxnb(scgemm_conv_routine_t routine,
                              const tensorshape_t& Sd,
                              const tensorshape_t& Sf,
                              const std::vector<uint32_t>& strides,
                              const std::vector<uint32_t>& dilations);

/*
 *! @brief Returns the extra buffer size required for a particular routine.
 *
 * @param routine    Routine type of the Static Compiled Gemm (input)
 * @param S          Tensorshape of input data (intput)
 * return            auxbuf size in number of bytes
*/
size_t scgemm_get_fgemm_auxnb(scgemm_gemm_routine_t routine, const tensorshape_t& S);

/*
 *! @brief Get the type of op through specific kernel name.
 *
 * @param kernel_name  Name of the static compiled gemm kernel (output)
 * return              Type of op
*/
scgemm_op_t scgemm_get_op_type(const std::string& kernel_name);

/*
 *! @brief Get the type of routine through specific kernel name.
 *
 * @param kernel_name  Name of the static compiled gemm kernel (output)
 * return              Type of routine
*/
scgemm_conv_routine_t scgemm_get_conv_routine(const std::string& kernel_name);

/*
 *! @brief Get the type of routine through specific kernel name.
 *
 * @param kernel_name  Name of the static compiled gemm kernel (output)
 * return              Type of routine
*/
scgemm_gemm_routine_t scgemm_get_gemm_routine(const std::string& kernel_name);

/*
 *! @brief Initialize the parameter for a particular problem
 *
 * @param kernel_name  Name of the static compiled gemm kernel (output)
 * @param grids        3D array containing the size of grid. (output)
 * @param blocks       3D array containing the size of block. (output)
 * @param routine      Type of static compiled gemm routine. (input)
 * @param Sa           Tensorshape of input data (intput)
 * @param Sb           Tensorshape of filter (intput)
 * @param strides      Array containing the size of stride (input)
 * @param dilations    Array containing the size of dilation, dilation is start from 0. (input)
 * @param mask         Reserve (default = 0) (input)
 * @param ng           Number of group (input)
 * return              Parameter
*/
scgemm_params_t scgemm_create_fconv_params(std::string& kernel_name,
                                           std::vector<uint32_t>& grids,
                                           std::vector<uint32_t>& blocks,
                                           scgemm_conv_routine_t routine,
                                           const tensorshape_t& Sa,
                                           const tensorshape_t& Sb,
                                           const std::vector<uint32_t>& strides,
                                           const std::vector<uint32_t>& dilations,
                                           uint32_t mask,
                                           uint32_t ng);

/*
 *! @brief Initialize the parameter for a particular problem
 *
 * @param kernel_name  Name of the static compiled gemm kernel (output)
 * @param grids        3D array containing the size of grid. (output)
 * @param blocks       3D array containing the size of block. (output)
 * @param routine      Type of static compiled gemm routine. (input)
 * @param Sa           Tensorshape of input data (intput)
 * @param Sb           Tensorshape of filter (intput)
 * @param mask         Reserve (default = 0) (input)
 * @param ng           Number of group (input)
 * return              Parameter
*/
scgemm_params_t scgemm_create_fgemm_params(std::string& kernel_name,
                                           std::vector<uint32_t>& grids,
                                           std::vector<uint32_t>& blocks,
                                           scgemm_gemm_routine_t routine,
                                           const tensorshape_t& Sa,
                                           const tensorshape_t& Sb,
                                           uint32_t mask,
                                           uint32_t ng);

/// These fucintions are used to present how to fill in the kernel arguments for a
/// Static Compiled GEMM - GEMM/FCONV Kernel.
/// Left the code here for reference.
//
// using scgemm_kernel_args_t = std::unique_ptr<char>;
//
/*
 *! @brief Generate the kernel arguments.
 *
 * @param args_size Number bytes of the kernel argument size (output)
 * @param src       Input tensor (input)
 * @param wei       Filter tensor (input)
 * @param bias      Bias tensor (not suppoted, default = nullptr) (input)
 * @param out       Output tensor (input)
 * @param auxbuf    Buffer of aux information (intput)
 * @param params    Static Compiled Gemm parameters (input)
 * @param coef      Reserve (default = 1.0)
 * return           Kernel arguments
*/
/*
scgemm_kernel_args_t scgemm_compiled_fgemm_args(size_t& args_size,
                                                const void* src,
                                                const void* wei,
                                                const void* bias,
                                                void* out,
                                                void* auxbuf,
                                                scgemm_params_t& params,
                                                float coef);
*/
/*
 *! @brief Generate the kernel arguments.
 *
 * @param args_size Number bytes of the kernel argument size (output)
 * @param src       Input tensor (input)
 * @param wei       Filter tensor (input)
 * @param bias      Bias tensor (not suppoted, default = nullptr) (input)
 * @param out       Output tensor (input)
 * @param auxbuf    Buffer of aux information (intput)
 * @param params    Static Compiled Gemm parameters (input)
 * @param coef      Reserve (default = 1.0)
 * return           Kernel arguments
*/
/*
scgemm_kernel_args_t scgemm_compiled_fconv_args(size_t& args_size,
                                                const void* src,
                                                const void* wei,
                                                const void* bias,
                                                void* out,
                                                void* auxbuf,
                                                scgemm_params_t& params,
                                                float coef);
*/
} // namespace scgemm
} // namespace miopen

#endif
