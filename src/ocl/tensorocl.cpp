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
#include <miopen/float_equal.hpp>
#include <miopen/visit_float.hpp>
#include <numeric>

#define MIO_TENSOROCL_DEBUG 0

namespace miopen {

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

void OpTensor3d(Handle& handle,
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
    auto alens = aTensorDesc.GetLengths();
    auto blens = bTensorDesc.GetLengths();
    auto clens = cTensorDesc.GetLengths();

    auto astrides = aTensorDesc.GetStrides();
    auto bstrides = bTensorDesc.GetStrides();
    auto cstrides = cTensorDesc.GetStrides();

    auto bsize = blens.size();

    // first_not_one is incorrect if btensor size equal to 1
    auto first_not_one = std::find_if(blens.rbegin(), blens.rend(), [](int i) { return i != 1; });
    auto d             = std::distance(blens.begin(), first_not_one.base());

    // quick fix
    int num_wg = first_not_one != blens.rend() ? (*first_not_one == 0 ? 1 : *first_not_one) : 1;
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

    int num_wg_orig = num_wg;
    int max_num_wg  = 4096;
    num_wg          = num_wg > max_num_wg ? max_num_wg : num_wg;

    size_t local_threads = 256;

    std::string network_config{};

    network_config = std::to_string(bTensorDesc.GetType()) + std::to_string(aTensorDesc.GetType()) +
                     std::to_string(tensorOp);

    visit_float(bTensorDesc.GetType(), [&](auto as_float) {

        auto miopen_alpha0 = as_float(*(static_cast<const float*>(alpha0)));
        auto miopen_alpha1 = as_float(*(static_cast<const float*>(alpha1)));
        auto miopen_beta   = as_float(*(static_cast<const float*>(beta)));

        if(clens[0] == 1 && blens[0] == 1 && alens[0] == 1 && blens[1] == clens[1] &&
           blens[2] == clens[2])
        {

            network_config +=
                std::to_string(clens[2]) + std::to_string(clens[1]) + std::to_string(miopen_beta);

            auto&& kernels = handle.GetKernels("Op2dTensorLite", network_config);

            if(!kernels.empty())
            {
                auto kernel = kernels.front();

                kernel(ATensor,
                       int(astrides[1]), // a_cstride,
                       BTensor,
                       int(bstrides[1]), // b_cstride,
                       CTensor,
                       int(cstrides[1]), // c_cstride,
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       long(Aoffset),
                       long(Boffset),
                       long(Coffset));

                return;
            }
        }
        else
        {

            network_config +=
                std::to_string(max_num_wg) + std::to_string(local_threads) + std::to_string(num_wg);

            auto&& kernels = handle.GetKernels("Op3dTensorGeneric", network_config);

            if(!kernels.empty())
            {
                auto kernel = kernels.front();

                kernel(ATensor,
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
                       long(Coffset),
                       int(num_wg_orig));

                return;
            }
        }

        std::string parms = " -DMIOPEN_TYPE=" + GetDataType(bTensorDesc.GetType());

        if(aTensorDesc.GetType() == miopenFloat)
        {
            parms += " -DMIOPEN_USE_FP16=0";
            parms += " -DMIOPEN_USE_FP32=1";
        }
        else if(aTensorDesc.GetType() == miopenHalf)
        {
            parms += " -DMIOPEN_USE_FP16=1";
            parms += " -DMIOPEN_USE_FP32=0";
        }

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

        if(clens[0] == 1 && blens[0] == 1 && alens[0] == 1 && blens[1] == clens[1] &&
           blens[2] == clens[2])
        {
            parms += " -DUSE_2D_TENSOR_LITE";

            // for naive tensor ops
            size_t RD_BLCK              = (clens[2] % 4 == 0) ? 4 : (clens[2] % 2 == 0) ? 2 : 1;
            const std::string data_type = GetDataType(bTensorDesc.GetType());
            const std::string READ_TYPE =
                (RD_BLCK == 1) ? data_type : data_type + std::to_string(RD_BLCK);

            size_t MAP_RD = clens[2] / RD_BLCK;
            parms += " -DRD_BLCK=" + std::to_string(RD_BLCK) + " -DMAP_RD=" +
                     std::to_string(MAP_RD) + " -DREAD_TYPE=" + READ_TYPE;

            if(!float_equal(miopen_beta, 0.0))
            {
                parms += " -DBETA";
            }

            const std::vector<size_t> vgd1{MAP_RD, clens[1], 1};

            handle.AddKernel(
                "Op2dTensorLite", network_config, program_name, "Op2dTensorLite", vld, vgd1, parms)(
                ATensor,
                int(astrides[1]), // a_cstride,
                BTensor,
                int(bstrides[1]), // b_cstride,
                CTensor,
                int(cstrides[1]), // c_cstride,
                miopen_alpha0,
                miopen_alpha1,
                miopen_beta,
                long(Aoffset),
                long(Boffset),
                long(Coffset));
        }
        else
        {
            // Special case for adding tensors in place
            size_t global_threads;
            global_threads = num_wg * local_threads;
            const std::vector<size_t> vgd{global_threads, 1, 1};

            parms += " -DUSE_3D_TENSOR_GENERIC";
            parms += " -DMAX_NUM_WG=" + std::to_string(max_num_wg);

            handle.AddKernel("Op3dTensorGeneric",
                             network_config,
                             program_name,
                             "Op3dTensorGeneric",
                             vld,
                             vgd,
                             parms)(ATensor,
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
                                    long(Coffset),
                                    int(num_wg_orig));
        }
    });
}

void OpTensor4d(Handle& handle,
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
    auto blens = bTensorDesc.GetLengths();
    auto clens = cTensorDesc.GetLengths();
    auto dims  = clens.size();

    auto astrides = aTensorDesc.GetStrides();
    auto bstrides = bTensorDesc.GetStrides();
    auto bsize    = blens.size();
    auto cstrides = cTensorDesc.GetStrides();

    // first_not_one is incorrect if btensor size equal to 1
    auto first_not_one = std::find_if(blens.rbegin(), blens.rend(), [](int i) { return i != 1; });
    auto d             = std::distance(blens.begin(), first_not_one.base());

    // quick fix
    int num_wg = first_not_one != blens.rend() ? (*first_not_one == 0 ? 1 : *first_not_one) : 1;
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

    int num_wg_orig = num_wg;
    int max_num_wg  = 4096;
    num_wg          = num_wg > max_num_wg ? max_num_wg : num_wg;

    size_t local_threads = 256;

    // Does the bitmap contain leading ones, i.e. 1,1,1,0 or 1,1,0,0
    // or 1,1,1,1 or 1,0,0,0
    bool leading_ones = IsBitmapLeadingOnes(bitmap, dims, (d - 2));
    if(leading_ones && work_per_wg < 64)
    {
        local_threads = 64;
    }

    std::string network_config{};

    network_config += GetDataType(bTensorDesc.GetType()) + std::to_string(max_num_wg);

    std::string program_name = "MIOpenTensorKernels.cl";

    const std::vector<size_t> vld{local_threads, 1, 1};

    // Special case for adding tensors in place
    size_t global_threads;
    global_threads = (leading_ones == 1 && (d - 1) == 3) ? num_wg : num_wg * local_threads;
    global_threads = (global_threads < local_threads) ? local_threads : global_threads;

    const std::vector<size_t> vgd{global_threads, 1, 1};

    bool packed_tensor = true;

    // auto alens = aTensorDesc.GetLengths();
    packed_tensor &= aTensorDesc.IsPacked();
    packed_tensor &= bTensorDesc.IsPacked();
    packed_tensor &= cTensorDesc.IsPacked();

#if(MIO_TENSOROCL_DEBUG == 1)
    printf("packed_tensor: %d\n", packed_tensor);
#endif

    network_config += std::to_string(bTensorDesc.GetType()) +
                      std::to_string(aTensorDesc.GetType()) + std::to_string(tensorOp) +
                      std::to_string(global_threads) + std::to_string(local_threads);

    visit_float(bTensorDesc.GetType(), [&](auto as_float) {

        auto miopen_alpha0 = as_float(*(static_cast<const float*>(alpha0)));
        auto miopen_alpha1 = as_float(*(static_cast<const float*>(alpha1)));
        auto miopen_beta   = as_float(*(static_cast<const float*>(beta)));

        if(fwd_conv_bias)
        {
            network_config += std::to_string(incr_wg);

            if(packed_tensor)
            {
                auto&& kernels = handle.GetKernels("OpTensorFwdBias", network_config);

                if(!kernels.empty())
                {
                    auto kernel = kernels.front();
                    kernel(ATensor,
                           BTensor,
                           int(blens[1]),
                           CTensor,
                           int(clens[0]),
                           int(cstrides[0]),
                           int(cstrides[1]),
                           work_per_wg,
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           long(Aoffset),
                           long(Boffset),
                           long(Coffset),
                           int(num_wg_orig));

                    return;
                }
            }
            else
            {

                auto&& kernels = handle.GetKernels("OpTensorFwdBiasGeneric", network_config);

                if(!kernels.empty())
                {
                    auto kernel = kernels.front();
                    kernel(ATensor,
                           int(astrides[0]),
                           int(astrides[1]),
                           int(astrides[2]),
                           BTensor,
                           int(blens[1]),
                           int(bstrides[1]),
                           CTensor,
                           int(clens[0]),
                           int(clens[3]),
                           int(cstrides[0]),
                           int(cstrides[1]),
                           int(cstrides[2]),
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           work_per_wg,
                           long(Aoffset),
                           long(Boffset),
                           long(Coffset),
                           int(num_wg_orig));
                    return;
                }
            }
        }
        else if(leading_ones)
        {
            network_config += std::to_string(d - 1);
            if(packed_tensor)
            {

                auto&& kernels = handle.GetKernels("OpTensorLeadingOnes", network_config);

                if(!kernels.empty())
                {
                    auto kernel = kernels.front();
                    kernel(ATensor,
                           BTensor,
                           CTensor,
                           int(clens[1]),
                           int(clens[2]),
                           int(clens[3]),
                           int(cstrides[0]),
                           int(cstrides[1]),
                           work_per_wg,
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           long(Aoffset),
                           long(Boffset),
                           long(Coffset),
                           int(num_wg_orig));

                    return;
                }
            }
            else
            {
                auto&& kernels = handle.GetKernels("OpTensorLeadingOnesGeneric", network_config);

                if(!kernels.empty())
                {
                    auto kernel = kernels.front();
                    kernel(ATensor,
                           int(astrides[0]),
                           int(astrides[1]),
                           int(astrides[2]),
                           BTensor,
                           int(bstrides[0]),
                           int(bstrides[1]),
                           int(bstrides[2]),
                           CTensor,
                           int(clens[1]),
                           int(clens[2]),
                           int(clens[3]),
                           int(cstrides[0]),
                           int(cstrides[1]),
                           int(cstrides[2]),
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           work_per_wg,
                           long(Aoffset),
                           long(Boffset),
                           long(Coffset),
                           int(num_wg_orig));
                    return;
                }
            }
        }
        else
        {
            auto&& kernels = handle.GetKernels("Op4dTensorGeneric", network_config);

            if(!kernels.empty())
            {
                auto kernel = kernels.front();
                kernel(ATensor,
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
                       long(Coffset),
                       int(num_wg_orig));
                return;
            }
        }

        std::string parms = " -DMIOPEN_TYPE=" + GetDataType(bTensorDesc.GetType()) +
                            " -DMAX_NUM_WG=" + std::to_string(max_num_wg);

        if(aTensorDesc.GetType() == miopenFloat)
        {
            parms += " -DMIOPEN_USE_FP16=0";
            parms += " -DMIOPEN_USE_FP32=1";
        }
        else if(aTensorDesc.GetType() == miopenHalf)
        {
            parms += " -DMIOPEN_USE_FP16=1";
            parms += " -DMIOPEN_USE_FP32=0";
        }

        parms += " -DMIOPEN_TENSOR_OP=";
        switch(tensorOp)
        {
        case 0: parms += "miopenAdd"; break;
        case 1: parms += "miopenMul"; break;
        case 2: parms += "miopenMin"; break;
        case 3: parms += "miopenMax"; break;
        }

        if(fwd_conv_bias)
        {
            parms += " -DINCR_WG=" + std::to_string(incr_wg);

            if(packed_tensor)
            {
                parms += " -DUSE_FWD_BIAS";

                handle.AddKernel("OpTensorFwdBias",
                                 network_config,
                                 program_name,
                                 "OpTensorFwdBias",
                                 vld,
                                 vgd,
                                 parms)(ATensor,
                                        BTensor,
                                        int(blens[1]),
                                        CTensor,
                                        int(clens[0]),
                                        int(cstrides[0]),
                                        int(cstrides[1]),
                                        work_per_wg,
                                        miopen_alpha0,
                                        miopen_alpha1,
                                        miopen_beta,
                                        long(Aoffset),
                                        long(Boffset),
                                        long(Coffset),
                                        int(num_wg_orig));
            }
            else
            {
                parms += " -DUSE_FWD_BIAS_GENERIC";
                handle.AddKernel("OpTensorFwdBiasGeneric",
                                 network_config,
                                 program_name,
                                 "OpTensorFwdBiasGeneric",
                                 vld,
                                 vgd,
                                 parms)(ATensor,
                                        int(astrides[0]),
                                        int(astrides[1]),
                                        int(astrides[2]),
                                        BTensor,
                                        int(blens[1]),
                                        int(bstrides[1]),
                                        CTensor,
                                        int(clens[0]),
                                        int(clens[3]),
                                        int(cstrides[0]),
                                        int(cstrides[1]),
                                        int(cstrides[2]),
                                        miopen_alpha0,
                                        miopen_alpha1,
                                        miopen_beta,
                                        work_per_wg,
                                        long(Aoffset),
                                        long(Boffset),
                                        long(Coffset),
                                        int(num_wg_orig));
            }
        }
        else if(leading_ones)
        {
            parms += " -DFIRST_NOT_ONE=" + std::to_string(d - 1);
            if(packed_tensor)
            {
                parms += " -DUSE_LEADING_ONES";
                handle.AddKernel("OpTensorLeadingOnes",
                                 network_config,
                                 program_name,
                                 "OpTensorLeadingOnes",
                                 vld,
                                 vgd,
                                 parms)(ATensor,
                                        BTensor,
                                        CTensor,
                                        int(clens[1]),
                                        int(clens[2]),
                                        int(clens[3]),
                                        int(cstrides[0]),
                                        int(cstrides[1]),
                                        work_per_wg,
                                        miopen_alpha0,
                                        miopen_alpha1,
                                        miopen_beta,
                                        long(Aoffset),
                                        long(Boffset),
                                        long(Coffset),
                                        int(num_wg_orig));
            }
            else
            {

                parms += " -DUSE_LEADING_ONES_GENERIC";

                handle.AddKernel("OpTensorLeadingOnesGeneric",
                                 network_config,
                                 program_name,
                                 "OpTensorLeadingOnesGeneric",
                                 vld,
                                 vgd,
                                 parms)(ATensor,
                                        int(astrides[0]),
                                        int(astrides[1]),
                                        int(astrides[2]),
                                        BTensor,
                                        int(bstrides[0]),
                                        int(bstrides[1]),
                                        int(bstrides[2]),
                                        CTensor,
                                        int(clens[1]),
                                        int(clens[2]),
                                        int(clens[3]),
                                        int(cstrides[0]),
                                        int(cstrides[1]),
                                        int(cstrides[2]),
                                        miopen_alpha0,
                                        miopen_alpha1,
                                        miopen_beta,
                                        work_per_wg,
                                        long(Aoffset),
                                        long(Boffset),
                                        long(Coffset),
                                        int(num_wg_orig));
            }
        }
        else
        {
            parms += " -DUSE_4D_TENSOR_GENERIC";

            handle.AddKernel("Op4dTensorGeneric",
                             network_config,
                             program_name,
                             "Op4dTensorGeneric",
                             vld,
                             vgd,
                             parms)(ATensor,
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
                                    long(Coffset),
                                    int(num_wg_orig));
        }
    });
}

void OpTensorOther(Handle& handle,
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
    auto blens = bTensorDesc.GetLengths();
    auto clens = cTensorDesc.GetLengths();

    auto astrides = aTensorDesc.GetStrides();
    auto bstrides = bTensorDesc.GetStrides();
    auto bsize    = blens.size();
    auto cstrides = cTensorDesc.GetStrides();

    // first_not_one is incorrect if btensor size equal to 1
    auto first_not_one = std::find_if(blens.rbegin(), blens.rend(), [](int i) { return i != 1; });
    auto d             = std::distance(blens.begin(), first_not_one.base());

    // quick fix
    int num_wg = first_not_one != blens.rend() ? (*first_not_one == 0 ? 1 : *first_not_one) : 1;
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

    int num_wg_orig = num_wg;
    int max_num_wg  = 4096;
    num_wg          = num_wg > max_num_wg ? max_num_wg : num_wg;

    size_t local_threads = 256;

    std::string program_name = "MIOpenTensorKernels.cl";

    const std::vector<size_t> vld{local_threads, 1, 1};

    // Special case for adding tensors in place
    size_t global_threads;
    global_threads = num_wg * local_threads;

    const std::vector<size_t> vgd{global_threads, 1, 1};

    std::string network_config{};
    network_config += std::to_string(bTensorDesc.GetType()) +
                      std::to_string(aTensorDesc.GetType()) + std::to_string(tensorOp) +
                      std::to_string(global_threads) + std::to_string(local_threads);

    visit_float(bTensorDesc.GetType(), [&](auto as_float) {

        auto miopen_alpha0 = as_float(*(static_cast<const float*>(alpha0)));
        auto miopen_alpha1 = as_float(*(static_cast<const float*>(alpha1)));
        auto miopen_beta   = as_float(*(static_cast<const float*>(beta)));

        if(bsize == 5)
        {
            auto&& kernels = handle.GetKernels("Op5dTensorGeneric", network_config);

            if(!kernels.empty())
            {
                auto kernel = kernels.front();
                kernel(ATensor,
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
                       long(Coffset),
                       int(num_wg_orig));
                return;
            }
        }
        else if(bsize == 2)
        {
            auto&& kernels = handle.GetKernels("Op2dTensorGeneric", network_config);

            if(!kernels.empty())
            {
                auto kernel = kernels.front();
                kernel(ATensor,
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
                       long(Coffset),
                       int(num_wg_orig));
                return;
            }
        }
        else if(bsize == 1)
        {
            auto&& kernels = handle.GetKernels("Op1dTensorGeneric", network_config);

            if(!kernels.empty())
            {

                auto kernel = kernels.front();
                kernel(ATensor,
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
                       long(Coffset),
                       int(num_wg_orig));
                return;
            }
        }

        std::string parms = " -DMIOPEN_TYPE=" + GetDataType(bTensorDesc.GetType()) +
                            " -DMAX_NUM_WG=" + std::to_string(max_num_wg);

        if(aTensorDesc.GetType() == miopenFloat)
        {
            parms += " -DMIOPEN_USE_FP16=0";
            parms += " -DMIOPEN_USE_FP32=1";
        }
        else if(aTensorDesc.GetType() == miopenHalf)
        {
            parms += " -DMIOPEN_USE_FP16=1";
            parms += " -DMIOPEN_USE_FP32=0";
        }

        parms += " -DMIOPEN_TENSOR_OP=";
        switch(tensorOp)
        {
        case 0: parms += "miopenAdd"; break;
        case 1: parms += "miopenMul"; break;
        case 2: parms += "miopenMin"; break;
        case 3: parms += "miopenMax"; break;
        }

        if(bsize == 5)
        {
            parms += " -DUSE_5D_TENSOR_GENERIC";

            handle.AddKernel("Op5dTensorGeneric",
                             network_config,
                             program_name,
                             "Op5dTensorGeneric",
                             vld,
                             vgd,
                             parms)(ATensor,
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
                                    long(Coffset),
                                    int(num_wg_orig));
        }
        else if(bsize == 2)
        {
            parms += " -DUSE_2D_TENSOR_GENERIC";

            handle.AddKernel("Op2dTensorGeneric",
                             network_config,
                             program_name,
                             "Op2dTensorGeneric",
                             vld,
                             vgd,
                             parms)(ATensor,
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
                                    long(Coffset),
                                    int(num_wg_orig));
        }
        else if(bsize == 1)
        {
            parms += " -DUSE_1D_TENSOR_GENERIC";

            handle.AddKernel("Op1dTensorGeneric",
                             network_config,
                             program_name,
                             "Op1dTensorGeneric",
                             vld,
                             vgd,
                             parms)(ATensor,
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
                                    long(Coffset),
                                    int(num_wg_orig));
        }

    });
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

    // if(aTensorDesc != cTensorDesc)
    if(aTensorDesc.GetElementSize() != cTensorDesc.GetElementSize())
    {
        MIOPEN_THROW("A and C Tensors do not match");
    }

    if(bTensorDesc.GetType() != cTensorDesc.GetType())
    {
        MIOPEN_THROW("Datatypes for B and C tensors do not match !");
    }

    auto blens = bTensorDesc.GetLengths();
#if(MIO_TENSOROCL_DEBUG == 1)
    printf("blen:[");
    for(auto len : blens)
    {
        printf(" %lu", len);
    }
    printf("]\n");
#endif
    auto clens = cTensorDesc.GetLengths();

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

    auto bsize = blens.size();
    if(bsize == 3)
    {
        OpTensor3d(handle,
                   tensorOp,
                   alpha0,
                   aTensorDesc,
                   ATensor,
                   alpha1,
                   bTensorDesc,
                   BTensor,
                   beta,
                   cTensorDesc,
                   CTensor,
                   Aoffset,
                   Boffset,
                   Coffset);
    }
    else if(bsize == 4)
    {
        OpTensor4d(handle,
                   tensorOp,
                   alpha0,
                   aTensorDesc,
                   ATensor,
                   alpha1,
                   bTensorDesc,
                   BTensor,
                   beta,
                   cTensorDesc,
                   CTensor,
                   Aoffset,
                   Boffset,
                   Coffset);
    }
    else
    {
        OpTensorOther(handle,
                      tensorOp,
                      alpha0,
                      aTensorDesc,
                      ATensor,
                      alpha1,
                      bTensorDesc,
                      BTensor,
                      beta,
                      cTensorDesc,
                      CTensor,
                      Aoffset,
                      Boffset,
                      Coffset);
    }
}

static std::string parms_half_or_float(const miopenDataType_t t)
{
    std::string s{};

    switch(t)
    {
    case miopenHalf:
    {
        s = " -DMIOPEN_USE_FP16=1 -DMIOPEN_USE_FP32=0";
        break;
    }
    case miopenFloat:
    {
        s = " -DMIOPEN_USE_FP16=0 -DMIOPEN_USE_FP32=1";
        break;
    }
    }

    return s;
}

struct two_exp_ceiling_t
{
    std::size_t operator() ( std::size_t n ) const
    {
        assert( n > 0 );
    
        std::size_t i = 1;
    
        n--;
        while( n != 0 )
        {
            i *= 2;
            n /= 2;
        }
    
        return i;
    }
};

static std::size_t two_exp_ceiling ( std::size_t n )
{
    assert( n > 0 );

    std::size_t i = 1;

    n--;
    while( n != 0 )
    {
        i *= 2;
        n /= 2;
    }

    return i;
}

void SetTensor(
    Handle& handle, const TensorDescriptor& yDesc, Data_t y, const void* alpha, const int offset)
{
    if(y == nullptr || alpha == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    auto ydim = yDesc.GetLengths().size();

    assert(ydim > 0 && ydim <= 5);

    switch(ydim)
    {
    case 1:
    {
        const std::vector<size_t> & lens = yDesc.GetLengths();

        std::string network_config = std::to_string(yDesc.GetType()) + " " +
                                     std::to_string(lens[0]);

        auto&& kernels = handle.GetKernels("SetTensor1d", network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenTensorSetKernel.cl";

            size_t wld = 256;
            size_t wgd = 65536;

            std::vector<size_t> work_lens(1);

            work_lens[0] = two_exp_ceiling(lens[0]);

            std::string parms = parms_half_or_float(yDesc.GetType()) + " -DWORK_LENGTH_0=" +
                                std::to_string(work_lens[0]);

            kernel = handle.AddKernel(
                "SetTensor1d", network_config, program_name, "SetTensor1d", {wld,1,1}, {wgd,1,1}, parms);
        }

        visit_float(yDesc.GetType(), [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   int(yDesc.GetStrides()[0]),
                   int(yDesc.GetLengths()[0]));
        });

        break;
    }
    case 2:
    {
        const std::vector<size_t> & lens = yDesc.GetLengths();

        std::string network_config = std::to_string(yDesc.GetType()) + " " +
                                     std::to_string(lens[0]) + " " + std::to_string(lens[1]);

        auto&& kernels = handle.GetKernels("SetTensor2d", network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenTensorSetKernel.cl";

            size_t wld = 256;
            size_t wgd = 65536;

            std::vector<size_t> work_lens(2);

            work_lens[0] = two_exp_ceiling(lens[0]);
            work_lens[1] = two_exp_ceiling(lens[1]);

            std::string parms = parms_half_or_float(yDesc.GetType()) + " -DWORK_LENGTH_0=" +
                                std::to_string(work_lens[0]) + " -DWORK_LENGTH_1=" +
                                std::to_string(work_lens[1]);

            kernel = handle.AddKernel(
                "SetTensor2d", network_config, program_name, "SetTensor2d", {wld,1,1}, {wgd,1,1}, parms);
        }

        visit_float(yDesc.GetType(), [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   int(yDesc.GetStrides()[0]),
                   int(yDesc.GetStrides()[1]),
                   int(yDesc.GetLengths()[0]),
                   int(yDesc.GetLengths()[1]));
        });

        break;
    }
    case 3:
    {
        std::string kernel_name = "SetTensor" + std::to_string(ydim) + "d";

        const std::vector<std::size_t> & lens = yDesc.GetLengths();

        std::string network_config = std::to_string(yDesc.GetType());
        for(auto & len : lens)
        {
            network_config += " " + std::to_string(len);
        }

        auto&& kernels = handle.GetKernels(kernel_name, network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenTensorSetKernel.cl";

            std::vector<std::size_t> work_lens(ydim);

            std::transform(lens.begin(), lens.end(), work_lens.begin(), two_exp_ceiling_t{} );

            std::size_t wgd = std::accumulate(work_lens.begin()+1, work_lens.end(), *(work_lens.begin()), [](auto a, auto b) {return a*b;} );

            std::size_t wld = 256 < wgd ? 256 : wgd;

            std::string parms = parms_half_or_float(yDesc.GetType());
            for( int i = 0; i < ydim; ++i )
            {
                parms += " -DWORK_LENGTH_" + std::to_string(i) +  std::to_string(work_lens[i]);
            }

            kernel = handle.AddKernel(
                kernel_name, network_config, program_name, kernel_name, {wld,1,1}, {wgd,1,1}, parms);
        }

        visit_float(yDesc.GetType(), [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   int(yDesc.GetStrides()[0]),
                   int(yDesc.GetStrides()[1]),
                   int(yDesc.GetStrides()[2]),
                   int(yDesc.GetLengths()[0]),
                   int(yDesc.GetLengths()[1]),
                   int(yDesc.GetLengths()[2]));
        });

        break;
    }
    case 4:
    {
        const std::vector<size_t> & lens = yDesc.GetLengths();

        std::string network_config = std::to_string(yDesc.GetType()) + " " +
                                     std::to_string(lens[0]) + " " + std::to_string(lens[1]) + " " +
                                     std::to_string(lens[2]) + " " + std::to_string(lens[3]);

        auto&& kernels = handle.GetKernels("SetTensor4d", network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenTensorSetKernel.cl";

            size_t wld = 256;
            size_t wgd = 65536;

            std::vector<size_t> work_lens(4);

            work_lens[0] = two_exp_ceiling(lens[0]);
            work_lens[1] = two_exp_ceiling(lens[1]);
            work_lens[2] = two_exp_ceiling(lens[2]);
            work_lens[3] = two_exp_ceiling(lens[3]);

            std::string parms = parms_half_or_float(yDesc.GetType()) + " -DWORK_LENGTH_0=" +
                                std::to_string(work_lens[0]) + " -DWORK_LENGTH_1=" +
                                std::to_string(work_lens[1]) + " -DWORK_LENGTH_2=" +
                                std::to_string(work_lens[2]) + " -DWORK_LENGTH_3=" +
                                std::to_string(work_lens[3]);

            kernel = handle.AddKernel(
                "SetTensor4d", network_config, program_name, "SetTensor4d", {wld,1,1}, {wgd,1,1}, parms);
        }

        visit_float(yDesc.GetType(), [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   int(yDesc.GetStrides()[0]),
                   int(yDesc.GetStrides()[1]),
                   int(yDesc.GetStrides()[2]),
                   int(yDesc.GetStrides()[3]),
                   int(yDesc.GetLengths()[0]),
                   int(yDesc.GetLengths()[1]),
                   int(yDesc.GetLengths()[2]),
                   int(yDesc.GetLengths()[3]));
        });

        break;
    }
    case 5:
    {
        const std::vector<size_t> & lens = yDesc.GetLengths();

        std::string network_config = std::to_string(yDesc.GetType()) + " " +
                                     std::to_string(lens[0]) + " " + std::to_string(lens[1]) + " " +
                                     std::to_string(lens[2]) + " " + std::to_string(lens[3]) + " " +
                                     std::to_string(lens[4]);

        auto&& kernels = handle.GetKernels("SetTensor5d", network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenTensorSetKernel.cl";

            size_t wld = 256;
            size_t wgd = 65536;

            std::vector<size_t> work_lens(5);

            work_lens[0] = two_exp_ceiling(lens[0]);
            work_lens[1] = two_exp_ceiling(lens[1]);
            work_lens[2] = two_exp_ceiling(lens[2]);
            work_lens[3] = two_exp_ceiling(lens[3]);
            work_lens[4] = two_exp_ceiling(lens[4]);

            std::string parms = parms_half_or_float(yDesc.GetType()) + " -DWORK_LENGTH_0=" +
                                std::to_string(work_lens[0]) + " -DWORK_LENGTH_1=" +
                                std::to_string(work_lens[1]) + " -DWORK_LENGTH_2=" +
                                std::to_string(work_lens[2]) + " -DWORK_LENGTH_3=" +
                                std::to_string(work_lens[3]) + " -DWORK_LENGTH_4=" +
                                std::to_string(work_lens[4]);

            kernel = handle.AddKernel(
                "SetTensor5d", network_config, program_name, "SetTensor5d", {wld,1,1}, {wgd,1,1}, parms);
        }

        visit_float(yDesc.GetType(), [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   int(yDesc.GetStrides()[0]),
                   int(yDesc.GetStrides()[1]),
                   int(yDesc.GetStrides()[2]),
                   int(yDesc.GetStrides()[3]),
                   int(yDesc.GetStrides()[4]),
                   int(yDesc.GetLengths()[0]),
                   int(yDesc.GetLengths()[1]),
                   int(yDesc.GetLengths()[2]),
                   int(yDesc.GetLengths()[3]),
                   int(yDesc.GetLengths()[4]));
        });

        break;
    }
    }
}

void ScaleTensor(
    Handle& handle, const TensorDescriptor& yDesc, Data_t y, const void* alpha, const int offset)
{
    if(y == nullptr || alpha == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    auto ydim = yDesc.GetLengths().size();

    assert(ydim > 0 && ydim <= 5);

    switch(ydim)
    {
    case 1:
    {
        const std::vector<size_t> & lens = yDesc.GetLengths();

        std::string network_config = std::to_string(yDesc.GetType()) + " " +
                                     std::to_string(lens[0]);

        auto&& kernels = handle.GetKernels("ScaleTensor1d", network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenTensorScaletKernel.cl";

            size_t wld = 256;
            size_t wgd = 65536;

            std::vector<size_t> work_lens(1);

            work_lens[0] = two_exp_ceiling(lens[0]);

            std::string parms = parms_half_or_float(yDesc.GetType()) + " -DWORK_LENGTH_0=" +
                                std::to_string(work_lens[0]);

            kernel = handle.AddKernel(
                "ScaleTensor1d", network_config, program_name, "ScaleTensor1d", {wld,1,1}, {wgd,1,1}, parms);
        }

        visit_float(yDesc.GetType(), [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   int(yDesc.GetStrides()[0]),
                   int(yDesc.GetLengths()[0]));
        });

        break;
    }
    case 2:
    {
        const std::vector<size_t> & lens = yDesc.GetLengths();

        std::string network_config = std::to_string(yDesc.GetType()) + " " +
                                     std::to_string(lens[0]) + " " + std::to_string(lens[1]);

        auto&& kernels = handle.GetKernels("ScaleTensor2d", network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenTensorScaleKernel.cl";

            size_t wld = 256;
            size_t wgd = 65536;

            std::vector<size_t> work_lens(2);

            work_lens[0] = two_exp_ceiling(lens[0]);
            work_lens[1] = two_exp_ceiling(lens[1]);

            std::string parms = parms_half_or_float(yDesc.GetType()) + " -DWORK_LENGTH_0=" +
                                std::to_string(work_lens[0]) + " -DWORK_LENGTH_1=" +
                                std::to_string(work_lens[1]);

            kernel = handle.AddKernel(
                "ScaleTensor2d", network_config, program_name, "ScaleTensor2d", {wld,1,1}, {wgd,1,1}, parms);
        }

        visit_float(yDesc.GetType(), [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   int(yDesc.GetStrides()[0]),
                   int(yDesc.GetStrides()[1]),
                   int(yDesc.GetLengths()[0]),
                   int(yDesc.GetLengths()[1]));
        });

        break;
    }
    case 3:
    {
        const std::vector<size_t> & lens = yDesc.GetLengths();

        std::string network_config = std::to_string(yDesc.GetType()) + " " +
                                     std::to_string(lens[0]) + " " + std::to_string(lens[1]) + " " +
                                     std::to_string(lens[2]);

        auto&& kernels = handle.GetKernels("ScaleTensor3d", network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenTensorScaleKernel.cl";

            size_t wld = 256;
            size_t wgd = 65536;

            std::vector<size_t> work_lens(3);

            work_lens[0] = two_exp_ceiling(lens[0]);
            work_lens[1] = two_exp_ceiling(lens[1]);
            work_lens[2] = two_exp_ceiling(lens[2]);

            std::string parms = parms_half_or_float(yDesc.GetType()) + " -DWORK_LENGTH_0=" +
                                std::to_string(work_lens[0]) + " -DWORK_LENGTH_1=" +
                                std::to_string(work_lens[1]) + " -DWORK_LENGTH_2=" +
                                std::to_string(work_lens[2]);

            kernel = handle.AddKernel(
                "ScaleTensor3d", network_config, program_name, "ScaleTensor3d", {wld,1,1}, {wgd,1,1}, parms);
        }

        visit_float(yDesc.GetType(), [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   int(yDesc.GetStrides()[0]),
                   int(yDesc.GetStrides()[1]),
                   int(yDesc.GetStrides()[2]),
                   int(yDesc.GetLengths()[0]),
                   int(yDesc.GetLengths()[1]),
                   int(yDesc.GetLengths()[2]));
        });

        break;
    }
    case 4:
    {
        const std::vector<size_t> & lens = yDesc.GetLengths();

        std::string network_config = std::to_string(yDesc.GetType()) + " " +
                                     std::to_string(lens[0]) + " " + std::to_string(lens[1]) + " " +
                                     std::to_string(lens[2]) + " " + std::to_string(lens[3]);

        auto&& kernels = handle.GetKernels("ScaleTensor4d", network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenTensorScaleKernel.cl";

            size_t wld = 256;
            size_t wgd = 65536;

            std::vector<size_t> work_lens(4);

            work_lens[0] = two_exp_ceiling(lens[0]);
            work_lens[1] = two_exp_ceiling(lens[1]);
            work_lens[2] = two_exp_ceiling(lens[2]);
            work_lens[3] = two_exp_ceiling(lens[3]);

            std::string parms = parms_half_or_float(yDesc.GetType()) + " -DWORK_LENGTH_0=" +
                                std::to_string(work_lens[0]) + " -DWORK_LENGTH_1=" +
                                std::to_string(work_lens[1]) + " -DWORK_LENGTH_2=" +
                                std::to_string(work_lens[2]) + " -DWORK_LENGTH_3=" +
                                std::to_string(work_lens[3]);

            kernel = handle.AddKernel(
                "ScaleTensor4d", network_config, program_name, "ScaleTensor4d", {wld,1,1}, {wgd,1,1}, parms);
        }

        visit_float(yDesc.GetType(), [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   int(yDesc.GetStrides()[0]),
                   int(yDesc.GetStrides()[1]),
                   int(yDesc.GetStrides()[2]),
                   int(yDesc.GetStrides()[3]),
                   int(yDesc.GetLengths()[0]),
                   int(yDesc.GetLengths()[1]),
                   int(yDesc.GetLengths()[2]),
                   int(yDesc.GetLengths()[3]));
        });

        break;
    }
    case 5:
    {
        const std::vector<size_t> & lens = yDesc.GetLengths();

        std::string network_config = std::to_string(yDesc.GetType()) + " " +
                                     std::to_string(lens[0]) + " " + std::to_string(lens[1]) + " " +
                                     std::to_string(lens[2]) + " " + std::to_string(lens[3]) + " " +
                                     std::to_string(lens[4]);

        auto&& kernels = handle.GetKernels("ScaleTensor5d", network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenTensorScaleKernel.cl";

            size_t wld = 256;
            size_t wgd = 65536;

            std::vector<size_t> work_lens(5);

            work_lens[0] = two_exp_ceiling(lens[0]);
            work_lens[1] = two_exp_ceiling(lens[1]);
            work_lens[2] = two_exp_ceiling(lens[2]);
            work_lens[3] = two_exp_ceiling(lens[3]);
            work_lens[4] = two_exp_ceiling(lens[4]);

            std::string parms = parms_half_or_float(yDesc.GetType()) + " -DWORK_LENGTH_0=" +
                                std::to_string(work_lens[0]) + " -DWORK_LENGTH_1=" +
                                std::to_string(work_lens[1]) + " -DWORK_LENGTH_2=" +
                                std::to_string(work_lens[2]) + " -DWORK_LENGTH_3=" +
                                std::to_string(work_lens[3]) + " -DWORK_LENGTH_4=" +
                                std::to_string(work_lens[4]);

            kernel = handle.AddKernel(
                "ScaleTensor5d", network_config, program_name, "ScaleTensor5d", {wld,1,1}, {wgd,1,1}, parms);
        }

        visit_float(yDesc.GetType(), [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   int(yDesc.GetStrides()[0]),
                   int(yDesc.GetStrides()[1]),
                   int(yDesc.GetStrides()[2]),
                   int(yDesc.GetStrides()[3]),
                   int(yDesc.GetStrides()[4]),
                   int(yDesc.GetLengths()[0]),
                   int(yDesc.GetLengths()[1]),
                   int(yDesc.GetLengths()[2]),
                   int(yDesc.GetLengths()[3]),
                   int(yDesc.GetLengths()[4]));
        });

        break;
    }
    }
}

void CopyTensor(Handle& handle,
                const TensorDescriptor& srcDesc,
                ConstData_t src,
                const TensorDescriptor& dstDesc,
                Data_t dst,
                int srcOffset,
                int dstOffset)
{
    if(src == nullptr || dst == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }
    if(srcDesc.GetElementSize() != dstDesc.GetElementSize())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor data sizes do not match.");
    }

    if(srcDesc.GetType() != dstDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
    }

    if(srcDesc.GetLengths().size() != dstDesc.GetLengths().size())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    if(srcDesc.GetLengths().size() > 5 || dstDesc.GetLengths().size() > 5)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension sizes unsupported.");
    }

    if(srcOffset > 0 || dstOffset > 0 || srcDesc != dstDesc ||
       (srcDesc.GetElementSpace() != srcDesc.GetElementSize() ||
        dstDesc.GetElementSpace() != dstDesc.GetElementSize()))
    {
        auto srcDim = srcDesc.GetLengths().size();

        assert(srcDim > 0 && srcDim <= 5);

        switch(srcDim)
        {
        case 1:
        {
            std::vector<size_t> vld             = {256, 1, 1};
            std::vector<size_t> data_per_thread = {16, 1, 1};
            std::vector<size_t> vgd             = {1, 1, 1};

            vgd[0] = ((srcDesc.GetLengths()[0] - 1) / (vld[0] * data_per_thread[0]) + 1) * vld[0];

            std::string network_config =
                std::to_string(srcDesc.GetType()) + " " + std::to_string(vgd[0]);

            auto&& kernels = handle.GetKernels("CopyTensor1d", network_config);

            KernelInvoke kernel;

            if(!kernels.empty())
            {
                kernel = kernels.front();
            }
            else
            {
                std::string parms = parms_half_or_float(srcDesc.GetType()) +
                                    " -DGLOBAL_WORK_SIZE_X=" + std::to_string(vgd[0]);

                std::string program_name = "MIOpenTensorCopyKernel.cl";
                kernel                   = handle.AddKernel(
                    "CopyTensor1d", network_config, program_name, "CopyTensor1d", vld, vgd, parms);
            }

            kernel(src,
                   srcOffset,
                   int(srcDesc.GetStrides()[0]),
                   int(srcDesc.GetLengths()[0]),
                   dst,
                   dstOffset,
                   int(dstDesc.GetStrides()[0]));

            break;
        }
        case 2:
        {
            std::vector<size_t> vld             = {16, 16, 1};
            std::vector<size_t> data_per_thread = {4, 4, 1};
            std::vector<size_t> vgd             = {1, 1, 1};

            vgd[0] = ((srcDesc.GetLengths()[0] - 1) / (vld[0] * data_per_thread[0]) + 1) * vld[0];
            vgd[1] = ((srcDesc.GetLengths()[1] - 1) / (vld[1] * data_per_thread[1]) + 1) * vld[1];

            std::string network_config = std::to_string(srcDesc.GetType()) + " " +
                                         std::to_string(vgd[0]) + " " + std::to_string(vgd[1]);

            auto&& kernels = handle.GetKernels("CopyTensor2d", network_config);

            KernelInvoke kernel;

            if(!kernels.empty())
            {
                kernel = kernels.front();
            }
            else
            {
                std::string parms = parms_half_or_float(srcDesc.GetType()) +
                                    " -DGLOBAL_WORK_SIZE_X=" + std::to_string(vgd[0]) +
                                    " -DGLOBAL_WORK_SIZE_Y=" + std::to_string(vgd[1]);

                std::string program_name = "MIOpenTensorCopyKernel.cl";
                kernel                   = handle.AddKernel(
                    "CopyTensor2d", network_config, program_name, "CopyTensor2d", vld, vgd, parms);
            }

            kernel(src,
                   srcOffset,
                   int(srcDesc.GetStrides()[0]),
                   int(srcDesc.GetStrides()[1]),
                   int(srcDesc.GetLengths()[0]),
                   int(srcDesc.GetLengths()[1]),
                   dst,
                   dstOffset,
                   int(dstDesc.GetStrides()[0]),
                   int(dstDesc.GetStrides()[1]));

            break;
        }
        case 3:
        {
            const std::vector<size_t> & lens = srcDesc.GetLengths();

            std::string network_config = std::to_string(srcDesc.GetType()) + " " +
                                         std::to_string(lens[0]) + " " + std::to_string(lens[1]) + " " +
                                         std::to_string(lens[2]);

            auto&& kernels = handle.GetKernels("CopyTensor3d", network_config);

            KernelInvoke kernel;

            if(!kernels.empty())
            {
                kernel = kernels.front();
            }
            else
            {
                std::string program_name = "MIOpenTensorCopyKernel.cl";

                std::vector<size_t> work_lens(3);

                work_lens[0] = two_exp_ceiling(lens[0]);
                work_lens[1] = two_exp_ceiling(lens[1]);
                work_lens[2] = two_exp_ceiling(lens[2]);

                size_t wgd = work_lens[0] * work_lens[1] * work_lens[2];
                size_t wld = 256 < wgd ? 256 : wgd;

                std::string parms = parms_half_or_float(srcDesc.GetType()) + 
                                    " -DWORK_LENGTH_0=" + std::to_string(work_lens[0]) +
                                    " -DWORK_LENGTH_1=" + std::to_string(work_lens[1]) + 
                                    " -DWORK_LENGTH_2=" + std::to_string(work_lens[2]);

                kernel = handle.AddKernel(
                    "CopyTensor3d", network_config, program_name, "CopyTensor3d", {wld,1,1}, {wgd,1,1}, parms);
            }

            kernel(src,
                   srcOffset,
                   int(srcDesc.GetStrides()[0]),
                   int(srcDesc.GetStrides()[1]),
                   int(srcDesc.GetStrides()[2]),
                   int(srcDesc.GetLengths()[0]),
                   int(srcDesc.GetLengths()[1]),
                   int(srcDesc.GetLengths()[2]),
                   dst,
                   dstOffset,
                   int(dstDesc.GetStrides()[0]),
                   int(dstDesc.GetStrides()[1]),
                   int(dstDesc.GetStrides()[2]));

            break;
        }
        case 4:
        {
            std::vector<size_t> vld             = {4, 8, 8};
            std::vector<size_t> data_per_thread = {4, 2, 2};
            std::vector<size_t> vgd             = {1, 1, 1};

            vgd[0] = ((srcDesc.GetLengths()[0] - 1) / (vld[0] * data_per_thread[0]) + 1) * vld[0];
            vgd[1] = ((srcDesc.GetLengths()[1] - 1) / (vld[1] * data_per_thread[1]) + 1) * vld[1];
            vgd[2] = ((srcDesc.GetLengths()[2] - 1) / (vld[2] * data_per_thread[2]) + 1) * vld[2];

            std::string network_config = std::to_string(srcDesc.GetType()) + " " +
                                         std::to_string(vgd[0]) + " " + std::to_string(vgd[1]) +
                                         " " + std::to_string(vgd[2]);

            auto&& kernels = handle.GetKernels("CopyTensor4d", network_config);

            KernelInvoke kernel;

            if(!kernels.empty())
            {
                kernel = kernels.front();
            }
            else
            {
                std::string parms = parms_half_or_float(srcDesc.GetType()) +
                                    " -DGLOBAL_WORK_SIZE_X=" + std::to_string(vgd[0]) +
                                    " -DGLOBAL_WORK_SIZE_Y=" + std::to_string(vgd[1]) +
                                    " -DGLOBAL_WORK_SIZE_Z=" + std::to_string(vgd[2]);

                std::string program_name = "MIOpenTensorCopyKernel.cl";
                kernel                   = handle.AddKernel(
                    "CopyTensor4d", network_config, program_name, "CopyTensor4d", vld, vgd, parms);
            }

            kernel(src,
                   srcOffset,
                   int(srcDesc.GetStrides()[0]),
                   int(srcDesc.GetStrides()[1]),
                   int(srcDesc.GetStrides()[2]),
                   int(srcDesc.GetStrides()[3]),
                   int(srcDesc.GetLengths()[0]),
                   int(srcDesc.GetLengths()[1]),
                   int(srcDesc.GetLengths()[2]),
                   int(srcDesc.GetLengths()[3]),
                   dst,
                   dstOffset,
                   int(dstDesc.GetStrides()[0]),
                   int(dstDesc.GetStrides()[1]),
                   int(dstDesc.GetStrides()[2]),
                   int(dstDesc.GetStrides()[3]));

            break;
        }
        case 5:
        {
            std::vector<size_t> vld             = {4, 8, 8};
            std::vector<size_t> data_per_thread = {4, 2, 2};
            std::vector<size_t> vgd             = {1, 1, 1};

            vgd[0] = ((srcDesc.GetLengths()[0] - 1) / (vld[0] * data_per_thread[0]) + 1) * vld[0];
            vgd[1] = ((srcDesc.GetLengths()[1] - 1) / (vld[1] * data_per_thread[1]) + 1) * vld[1];
            vgd[2] = ((srcDesc.GetLengths()[2] - 1) / (vld[2] * data_per_thread[2]) + 1) * vld[2];

            std::string network_config = std::to_string(srcDesc.GetType()) + " " +
                                         std::to_string(vgd[0]) + " " + std::to_string(vgd[1]) +
                                         " " + std::to_string(vgd[2]);

            auto&& kernels = handle.GetKernels("CopyTensor5d", network_config);

            KernelInvoke kernel;

            if(!kernels.empty())
            {
                kernel = kernels.front();
            }
            else
            {
                std::string parms = parms_half_or_float(srcDesc.GetType()) +
                                    " -DGLOBAL_WORK_SIZE_X=" + std::to_string(vgd[0]) +
                                    " -DGLOBAL_WORK_SIZE_Y=" + std::to_string(vgd[1]) +
                                    " -DGLOBAL_WORK_SIZE_Z=" + std::to_string(vgd[2]);

                std::string program_name = "MIOpenTensorCopyKernel.cl";
                kernel                   = handle.AddKernel(
                    "CopyTensor5d", network_config, program_name, "CopyTensor5d", vld, vgd, parms);
            }

            kernel(src,
                   srcOffset,
                   int(srcDesc.GetStrides()[0]),
                   int(srcDesc.GetStrides()[1]),
                   int(srcDesc.GetStrides()[2]),
                   int(srcDesc.GetStrides()[3]),
                   int(srcDesc.GetStrides()[4]),
                   int(srcDesc.GetLengths()[0]),
                   int(srcDesc.GetLengths()[1]),
                   int(srcDesc.GetLengths()[2]),
                   int(srcDesc.GetLengths()[3]),
                   int(srcDesc.GetLengths()[4]),
                   dst,
                   dstOffset,
                   int(dstDesc.GetStrides()[0]),
                   int(dstDesc.GetStrides()[1]),
                   int(dstDesc.GetStrides()[2]),
                   int(dstDesc.GetStrides()[3]),
                   int(dstDesc.GetStrides()[4]));

            break;
        }
        }
    }
    else
    {
        handle.Copy(src, dst, srcDesc.GetElementSize() * sizeof(srcDesc.GetType()));
    }
}

} // namespace miopen
