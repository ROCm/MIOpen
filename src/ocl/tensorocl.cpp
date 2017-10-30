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
#include <cassert>
#include <algorithm>
#include <miopen/errors.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include <numeric>

#define MIO_TENSOROCL_DEBUG 0

namespace miopen {

void SetTensor(Handle& handle,
               const TensorDescriptor& yDesc,
               Data_t y,
               const void* alpha,
               const size_t yOffset)
{

    if(y == nullptr || alpha == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    size_t global_threads = yDesc.GetElementSize();
    size_t local_threads  = 256;
    const std::vector<size_t> vld{local_threads, 1, 1};
    const std::vector<size_t> vgd{global_threads, 1, 1};

    std::string program_name = "MIOpenTensorScaleKernel.cl";
    switch(yDesc.GetType())
    {
    case miopenFloat:
    case miopenHalf:
    {
        float miopen_alpha = *(static_cast<const float*>(alpha));
        std::string parms =
            " -DMIOPEN_TYPE=" + GetDataType(yDesc.GetType()) + " -DMIOPEN_ALPHA_TYPE=float";

        handle.GetKernel("SetTensor", "", program_name, "SetTensor", vld, vgd, parms)(
            y, miopen_alpha, global_threads, long(yOffset));
    }
    break;
    }
}

void ScaleTensor(Handle& handle,
                 const TensorDescriptor& yDesc,
                 Data_t y,
                 const void* alpha,
                 const size_t yOffset)
{

    if(y == nullptr || alpha == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    size_t global_threads = yDesc.GetElementSize();
    size_t local_threads  = 256;
    const std::vector<size_t> vld{local_threads, 1, 1};
    const std::vector<size_t> vgd{global_threads, 1, 1};

    std::string program_name = "MIOpenTensorScaleKernel.cl";
    switch(yDesc.GetType())
    {
    case miopenFloat:
    case miopenHalf:
    {
        float miopen_alpha = *(static_cast<const float*>(alpha));
        std::string parms =
            " -DMIOPEN_TYPE=" + GetDataType(yDesc.GetType()) + " -DMIOPEN_ALPHA_TYPE=float";

        handle.GetKernel("ScaleTensor", "", program_name, "ScaleTensor", vld, vgd, parms)(
            y, miopen_alpha, global_threads, long(yOffset));
    }
    break;
    }
}

// Free Tensor Functions
static void CreateBitmapAndGrid(unsigned int& bitmap,
                                std::vector<std::size_t>& a_lens,
                                std::vector<std::size_t>& c_lens,
                                int& num_wg,
                                int& work,
                                int d)
{
    for(int i = d; i >= 0; i--)
    {
        if(a_lens[i] != 1)
        {
            bitmap |= (1 << (a_lens.size() - (i + 1)));
            num_wg *= a_lens[i];
        }
        else
        {
            work *= c_lens[i];
        }
    }
}

static bool IsBitmapLeadingOnes(unsigned int& bitmap, int n_size, int first_not_one)
{
    bool leading_ones = true;

    for(int i = first_not_one; i >= 0; i--)
    {
        bool is_one = (bitmap & (1 << (n_size - 1 - i)));
        leading_ones &= is_one;
    }
    return leading_ones;
}

void OpTensor(Handle& handle,
              miopenTensorOp_t tensorOp,
              const void* alpha0,
              const TensorDescriptor& aTensorDesc,
              ConstData_t ATensor,
              const void* alpha1,
              const TensorDescriptor& bTensorDesc,
              ConstData_t BTensor,
              const void* beta,
              const TensorDescriptor& cTensorDesc,
              Data_t CTensor,
              const size_t Aoffset,
              const size_t Boffset,
              const size_t Coffset)
{

    if(ATensor == nullptr || BTensor == nullptr || CTensor == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(aTensorDesc != cTensorDesc)
    {
        MIOPEN_THROW("A and C Tensors do not match");
    }

    if(bTensorDesc.GetType() != cTensorDesc.GetType())
    {
        MIOPEN_THROW("Datatypes for B and C tensors do not match !");
    }

    auto blens = bTensorDesc.GetLengths();
    auto clens = cTensorDesc.GetLengths();
    auto dims  = clens.size();

    if(clens.size() > 5)
    {
        MIOPEN_THROW("Tensor dimension larger than 5: " + std::to_string(clens.size()));
    }

    if(blens.size() != clens.size())
    {
        MIOPEN_THROW("Number of dims in B and C Tensors do not match: " +
                     std::to_string(blens.size()) + ", " + std::to_string(clens.size()));
    }

    for(auto i = 0; i < clens.size(); i++)
    {
        if(blens[i] != 1 && blens[i] != clens[i])
        {
            MIOPEN_THROW("BTensor dim != 1 && BTensor dim != CTensor dim: " + std::to_string(i));
        }
    }

    auto astrides = aTensorDesc.GetStrides();
    auto bstrides = bTensorDesc.GetStrides();
    auto bsize    = blens.size();
    auto cstrides = cTensorDesc.GetStrides();

    auto first_not_one = std::find_if(blens.rbegin(), blens.rend(), [](int i) { return i != 1; });
    auto d             = std::distance(blens.begin(), first_not_one.base());

    int num_wg      = 1;
    num_wg          = *first_not_one == 0 ? 1 : *first_not_one;
    int work_per_wg = std::accumulate(clens.begin() + d, clens.end(), 1, std::multiplies<int>());

    unsigned int bitmap = 0;
    // update bitmap for first_not_one
    bitmap |= (1 << (bsize - d));

    // (d-2) is because distance starts from 1 and 0
    // also, we need to go past the "first_not_one" as that is already
    // accounted for in the bitmap
    CreateBitmapAndGrid(bitmap, blens, clens, num_wg, work_per_wg, (d - 2));

#if(MIO_TENSOROCL_DEBUG == 1)
    printf("bitmap: %u\n", bitmap);
    printf("work_per_wg: %d, num_wg: %d\n", work_per_wg, num_wg);
#endif

    // Forward Convolution Bias specialization
    // for fwd-bias, bitmap looks like <0, 1, 0, 0>
    // Is the no. of work-groups and the work for each wg balanced?
    auto fwd_conv_bias = bitmap == (1 << 2) ? 1 : 0;
    auto incr_wg       = 0;
    // This block gives off indexing for 5d tensors, skipping
    if(fwd_conv_bias == 1 && dims < 5 && num_wg < 640 && work_per_wg > 256 && clens[0] > 0)
    { // 640 workgroups of size 256 needed to completely fill the GPU

        work_per_wg /= clens[0]; // c_n;
        num_wg *= clens[0];      // c_n;
        incr_wg = 1;
    }
    size_t local_threads = 256;

    // Does the bitmap contain leading ones, i.e. 1,1,1,0 or 1,1,0,0
    // or 1,1,1,1 or 1,0,0,0
    bool leading_ones = IsBitmapLeadingOnes(bitmap, dims, (d - 2));
    if(leading_ones && work_per_wg < 64)
    {
        local_threads = 64;
    }

    std::string parms = " -DFWD_CONV_BIAS=" + std::to_string(fwd_conv_bias) + " -DINCR_WG=" +
                        std::to_string(incr_wg) + " -DLEADING_ONES=" +
                        std::to_string(leading_ones) + " -DMIOPEN_TYPE=" +
                        GetDataType(bTensorDesc.GetType()) + " -DFIRST_NOT_ONE=" +
                        std::to_string(d - 1) + " -DMIOPEN_TENSOR_DIMS=" + std::to_string(bsize);

    parms += " -DMIOPEN_TENSOR_OP=";
    switch(tensorOp)
    {
    case 0: parms += "miopenAdd"; break;
    case 1: parms += "miopenMul"; break;
    case 2: parms += "miopenMin"; break;
    case 3: parms += "miopenMax"; break;
    }
    std::string program_name = "MIOpenTensorKernels.cl";

    const std::vector<size_t> vld{local_threads, 1, 1};

    // Special case for adding tensors in place
    size_t global_threads;
    // if(dims == 4)
    // global_threads = (leading_ones == 1 && (d - 1) == 3) ? num_wg : num_wg * local_threads;
    // else
    global_threads = (leading_ones == 1 && (d - 1) == dims) ? num_wg : num_wg * local_threads;
    global_threads = (global_threads < local_threads) ? local_threads : global_threads;

    const std::vector<size_t> vgd{global_threads, 1, 1};

    float miopen_alpha0, miopen_alpha1, miopen_beta;
    switch(bTensorDesc.GetType())
    {
    case miopenFloat:
    case miopenHalf:
    {
        miopen_alpha0 = *(static_cast<const float*>(alpha0));
        miopen_alpha1 = *(static_cast<const float*>(alpha1));
        miopen_beta   = *(static_cast<const float*>(beta));
    }
    break;
    }

    if(bsize == 5)
    {
        handle.GetKernel(
            "Op5dTensorGeneric", "", program_name, "Op5dTensorGeneric", vld, vgd, parms)(
            ATensor,
            int(astrides[0]),
            int(astrides[1]),
            int(astrides[2]),
            int(astrides[3]),
            BTensor,
            int(blens[1]),    // b_c,
            int(blens[2]),    // b_d,
            int(blens[3]),    // b_h,
            int(blens[4]),    // b_w,
            int(bstrides[0]), // b_nstride,
            int(bstrides[1]), // b_cstride,
            int(bstrides[2]), // b_dstride,
            int(bstrides[3]), // b_hstride,
            CTensor,
            int(clens[1]),    // c_c,
            int(clens[2]),    // c_d,
            int(clens[3]),    // c_h,
            int(clens[4]),    // c_w,
            int(cstrides[0]), // c_nstride,
            int(cstrides[1]), // c_cstride,
            int(cstrides[2]), // c_dstride,
            int(cstrides[3]), // c_hstride,
            miopen_alpha0,
            miopen_alpha1,
            miopen_beta,
            bitmap,
            work_per_wg,
            long(Aoffset),
            long(Boffset),
            long(Coffset));
    }
    else if(bsize == 4)
    {
        handle.GetKernel(
            "Op4dTensorGeneric", "", program_name, "Op4dTensorGeneric", vld, vgd, parms)(
            ATensor,
            int(astrides[0]), // a_nstride,
            int(astrides[1]), // a_cstride,
            int(astrides[2]), // a_hstride,
            BTensor,
            int(blens[1]),    // b_c,
            int(blens[2]),    // b_h,
            int(blens[3]),    // b_w,
            int(bstrides[0]), // b_nstride,
            int(bstrides[1]), // b_cstride,
            int(bstrides[2]), // b_hstride,
            CTensor,
            int(clens[1]),    // c_c,
            int(clens[2]),    // c_h,
            int(clens[3]),    // c_w,
            int(cstrides[0]), // c_nstride,
            int(cstrides[1]), // c_cstride,
            int(cstrides[2]), // c_hstride,
            miopen_alpha0,
            miopen_alpha1,
            miopen_beta,
            bitmap,
            work_per_wg,
            long(Aoffset),
            long(Boffset),
            long(Coffset));
    }
    else if(bsize == 3)
    {
        handle.GetKernel(
            "Op3dTensorGeneric", "", program_name, "Op3dTensorGeneric", vld, vgd, parms)(
            ATensor,
            int(astrides[0]), // a_nstride,
            int(astrides[1]), // a_cstride,
            BTensor,
            int(blens[1]),    // b_c,
            int(blens[2]),    // b_h,
            int(bstrides[0]), // b_nstride,
            int(bstrides[1]), // b_cstride,
            CTensor,
            int(clens[1]),    // c_c,
            int(clens[2]),    // c_h,
            int(cstrides[0]), // c_nstride,
            int(cstrides[1]), // c_cstride,
            miopen_alpha0,
            miopen_alpha1,
            miopen_beta,
            bitmap,
            work_per_wg,
            long(Aoffset),
            long(Boffset),
            long(Coffset));
    }
    else if(bsize == 2)
    {
        handle.GetKernel(
            "Op2dTensorGeneric", "", program_name, "Op2dTensorGeneric", vld, vgd, parms)(
            ATensor,
            int(astrides[0]),
            BTensor,
            int(blens[1]),
            int(bstrides[0]),
            CTensor,
            int(clens[1]),
            int(cstrides[0]),
            miopen_alpha0,
            miopen_alpha1,
            miopen_beta,
            bitmap,
            work_per_wg,
            long(Aoffset),
            long(Boffset),
            long(Coffset));
    }
    else if(bsize == 1)
    {
        handle.GetKernel(
            "Op1dTensorGeneric", "", program_name, "Op1dTensorGeneric", vld, vgd, parms)(
            ATensor,
            BTensor,
            int(blens[0]),
            CTensor,
            int(clens[0]),
            miopen_alpha0,
            miopen_alpha1,
            miopen_beta,
            bitmap,
            work_per_wg,
            long(Aoffset),
            long(Boffset),
            long(Coffset));
    }
    else if(fwd_conv_bias)
    {
        handle.GetKernel("OpTensorFwdBias", "", program_name, "OpTensorFwdBias", vld, vgd, parms)(
            ATensor,
            int(astrides[0]),
            int(astrides[1]),
            BTensor,
            int(blens[1]),
            int(bstrides[0]),
            int(bstrides[1]),
            CTensor,
            int(clens[0]),
            int(cstrides[0]),
            int(cstrides[1]),
            miopen_alpha0,
            miopen_alpha1,
            miopen_beta,
            work_per_wg,
            long(Aoffset),
            long(Boffset),
            long(Coffset));
    }
    else if(leading_ones)
    {
        handle.GetKernel(
            "OpTensorLeadingOnes", "", program_name, "OpTensorLeadingOnes", vld, vgd, parms)(
            ATensor,
            BTensor,
            CTensor,
            int(clens[1]),
            int(clens[2]),
            int(clens[3]),
            int(cstrides[0]),
            int(cstrides[1]),
            miopen_alpha0,
            miopen_alpha1,
            miopen_beta,
            work_per_wg,
            long(Aoffset),
            long(Boffset),
            long(Coffset));
    }
    else
    {
        handle.GetKernel(
            "Op4dTensorGeneric", "", program_name, "Op4dTensorGeneric", vld, vgd, parms)(
            ATensor,
            int(astrides[0]), // a_nstride,
            int(astrides[1]), // a_cstride,
            int(astrides[2]), // a_hstride,
            BTensor,
            int(blens[1]),    // b_c,
            int(blens[2]),    // b_h,
            int(blens[3]),    // b_w,
            int(bstrides[0]), // b_nstride,
            int(bstrides[1]), // b_cstride,
            int(bstrides[2]), // b_hstride,
            CTensor,
            int(clens[1]),    // c_c,
            int(clens[2]),    // c_h,
            int(clens[3]),    // c_w,
            int(cstrides[0]), // c_nstride,
            int(cstrides[1]), // c_cstride,
            int(cstrides[2]), // c_hstride,
            miopen_alpha0,
            miopen_alpha1,
            miopen_beta,
            bitmap,
            work_per_wg,
            long(Aoffset),
            long(Boffset),
            long(Coffset));
    }
}

void CopyTensor(Handle& handle,
                const TensorDescriptor& srcDesc,
                ConstData_t src,
                const TensorDescriptor& destDesc,
                Data_t dest)
{

    if(srcDesc.GetElementSize() != destDesc.GetElementSize() ||
       srcDesc.GetType() != destDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    size_t srcSize = srcDesc.GetElementSize();

    handle.Copy(src, dest, srcSize * sizeof(srcDesc.GetType()));
}

} // namespace miopen
