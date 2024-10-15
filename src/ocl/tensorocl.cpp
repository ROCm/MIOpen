/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/tensor.hpp>
#include <miopen/errors.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/datatype.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/util.hpp>
#include <miopen/logger.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <boost/range/combine.hpp>

#define MIO_TENSOROCL_DEBUG 0

namespace miopen {

TensorDescriptor GetFlattenedTensorDescriptor(const TensorDescriptor& desc)
{
    // is packed
    if(desc.IsPacked())
        return {desc.GetType(), {desc.GetElementSize()}, {static_cast<std::size_t>(1)}};

    // start flattening tensor
    std::vector<std::size_t> flat_lengths;
    std::vector<std::size_t> flat_strides;

    auto non1_length_strides = boost::combine(desc.GetLengths(), desc.GetStrides()) |
                               boost::adaptors::filtered(f_length_is_not_1_t());

    auto i               = non1_length_strides.begin();
    std::size_t flat_len = boost::get<0>(*i);
    auto i_previous      = i++;

    // the 0-th dimension full-length doesn't matter
    for(; i != non1_length_strides.end(); ++i)
    {
        std::size_t len             = boost::get<0>(*i);
        std::size_t stride          = boost::get<1>(*i);
        std::size_t previous_stride = boost::get<1>(*i_previous);
        std::size_t full_len        = previous_stride / stride;

        if(len == full_len)
        {
            flat_len *= len;
        }
        else
        {
            flat_lengths.push_back(flat_len);
            flat_strides.push_back(previous_stride);
            flat_len = len;
        }
        i_previous = i;
    }
    flat_lengths.push_back(flat_len);
    flat_strides.push_back(boost::get<1>(*i_previous));

    return {desc.GetType(), flat_lengths, flat_strides};
}

// Free Tensor Functions
static void CreateBitmapAndGrid(unsigned int& bitmap,
                                const std::vector<std::size_t>& a_lens,
                                const std::vector<std::size_t>& c_lens,
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

static bool IsBitmapLeadingOnes(unsigned int bitmap, int n_size, int first_not_one)
{
    bool leading_ones = true;

    for(int i = first_not_one; i >= 0; i--)
    {
        bool is_one = (bitmap & (1 << (n_size - 1 - i))) != 0u;
        leading_ones &= is_one;
    }
    return leading_ones;
}

void OpTensor3d(const Handle& handle,
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
                const size_t Coffset,
                const bool nonStandardSquash)
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
    int num_wg      = first_not_one != blens.rend()
                          ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                          : 1;
    int work_per_wg = std::accumulate(clens.begin() + d, clens.end(), 1, std::multiplies<int>());

    unsigned int bitmap = 0;
    // update bitmap for first_not_one
    bitmap |= (1 << (bsize - d));

    // (d-2) is because distance starts from 1 and 0
    // also, we need to go past the "first_not_one" as that is already
    // accounted for in the bitmap
    CreateBitmapAndGrid(bitmap, blens, clens, num_wg, work_per_wg, static_cast<int>(d - 2));

#if(MIO_TENSOROCL_DEBUG == 1)
    printf("bitmap: %u\n", bitmap);
    printf("work_per_wg: %d, num_wg: %d\n", work_per_wg, num_wg);
#endif

    int num_wg_orig = num_wg;
    int max_num_wg  = 4096;
    num_wg          = num_wg > max_num_wg ? max_num_wg : num_wg;

    size_t local_threads = 256;

    std::string network_config{};

    network_config = std::to_string(bTensorDesc.GetType()) + "-" +
                     std::to_string(aTensorDesc.GetType()) + "-" + std::to_string(tensorOp) + "-";

    // for naive tensor ops
    size_t RD_BLCK              = (clens[2] % 4 == 0) ? 4 : (clens[2] % 2 == 0) ? 2 : 1;
    const std::string data_type = GetDataType(bTensorDesc.GetType());
    const std::string READ_TYPE = (RD_BLCK == 1) ? data_type : data_type + std::to_string(RD_BLCK);

    size_t total_work = std::max(clens[2] / RD_BLCK, size_t(1));
    size_t grp_sz     = (total_work + local_threads - 1) / local_threads;

    // opencl kernels are no longer supported, fallback to generic case
    bool lite_applicable = grp_sz <= size_t(max_num_wg);

    bool is_lite = clens[0] == 1 && blens[0] == 1 && alens[0] == 1 &&
                   (blens[1] == clens[1] || blens[1] == 1) && blens[2] == clens[2];

    bool is_squashed = nonStandardSquash && !is_lite &&
                       (blens[0] == 1 && clens[0] == 1 && clens[1] == 1 && blens[2] == clens[2]);

    grp_sz        = std::min(size_t(max_num_wg), grp_sz);
    size_t glb_sz = local_threads * grp_sz;

    size_t local_threads2 = 64;
    size_t total_work2    = clens[1];
    size_t grp_sz2        = (total_work2 + local_threads2 - 1) / local_threads2;
    grp_sz2               = std::min(size_t(max_num_wg / grp_sz), grp_sz2);
    size_t glb_sz2        = local_threads2 * grp_sz2;

    visit_float(bTensorDesc.GetType(), [&](auto as_float) {
        auto miopen_alpha0 = as_float(*(static_cast<const float*>(alpha0)));
        auto miopen_alpha1 = as_float(*(static_cast<const float*>(alpha1)));
        auto miopen_beta   = as_float(*(static_cast<const float*>(beta)));

        if(lite_applicable && is_lite)
        {

            network_config += std::to_string(RD_BLCK) + "x" + std::to_string(local_threads) + "x" +
                              std::to_string(grp_sz) + std::to_string(local_threads2) +
                              std::to_string(grp_sz2);

            auto&& kernels = handle.GetKernels("Op2dTensorLite", network_config);

            if(!kernels.empty())
            {
                auto kernel = kernels.front();

                kernel(ATensor,
                       static_cast<int>(astrides[1]), // a_cstride,
                       BTensor,
                       static_cast<int>(bstrides[1]), // b_cstride,
                       CTensor,
                       static_cast<int>(cstrides[1]), // c_cstride,
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       static_cast<int64_t>(Aoffset),
                       static_cast<int64_t>(Boffset),
                       static_cast<int64_t>(Coffset),
                       static_cast<int64_t>(total_work),
                       static_cast<int64_t>(total_work2),
                       static_cast<int>(!float_equal(miopen_beta, 0.0)),
                       static_cast<int>(blens[1] == 1));

                return;
            }
        }
        else if(is_squashed)
        {
            network_config += std::to_string(RD_BLCK) + "x" + std::to_string(local_threads) + "x" +
                              std::to_string(grp_sz);

            auto&& kernels = handle.GetKernels("Op2dTensorSquash", network_config);

            if(!kernels.empty())
            {
                auto kernel = kernels.front();

                kernel(ATensor,
                       BTensor,
                       static_cast<int>(blens[1]),    // b_c,
                       static_cast<int>(bstrides[1]), // b_cstride,
                       CTensor,
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       static_cast<int64_t>(Aoffset),
                       static_cast<int64_t>(Boffset),
                       static_cast<int64_t>(Coffset),
                       static_cast<int64_t>(total_work),
                       static_cast<int>(!float_equal(miopen_alpha0, 0.0)),
                       static_cast<int>(!float_equal(miopen_alpha1, 0.0)),
                       static_cast<int>(!float_equal(miopen_beta, 0.0)));

                return;
            }
        }
        else
        {

            network_config += std::to_string(max_num_wg) + "-" + std::to_string(local_threads) +
                              "x" + std::to_string(num_wg);

            auto&& kernels = handle.GetKernels("Op3dTensorGeneric", network_config);

            if(!kernels.empty())
            {
                auto kernel = kernels.front();

                kernel(ATensor,
                       static_cast<int>(astrides[0]), // a_nstride,
                       static_cast<int>(astrides[1]), // a_cstride,
                       BTensor,
                       static_cast<int>(blens[1]),    // b_c,
                       static_cast<int>(blens[2]),    // b_h,
                       static_cast<int>(bstrides[0]), // b_nstride,
                       static_cast<int>(bstrides[1]), // b_cstride,
                       CTensor,
                       static_cast<int>(clens[1]),    // c_c,
                       static_cast<int>(clens[2]),    // c_h,
                       static_cast<int>(cstrides[0]), // c_nstride,
                       static_cast<int>(cstrides[1]), // c_cstride,
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       bitmap,
                       work_per_wg,
                       static_cast<int64_t>(Aoffset),
                       static_cast<int64_t>(Boffset),
                       static_cast<int64_t>(Coffset),
                       static_cast<int>(num_wg_orig));

                return;
            }
        }

        std::string parms = " -DMIOPEN_TYPE=" + GetDataType(bTensorDesc.GetType());

        parms += GetDataTypeKernelParams(aTensorDesc.GetType());

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

        if(lite_applicable && is_lite)
        {
            parms += " -DUSE_2D_TENSOR_LITE";
            parms += " -DRD_BLCK=" + std::to_string(RD_BLCK) + " -DREAD_TYPE=" + READ_TYPE;

            const std::vector<size_t> vgd1{glb_sz, glb_sz2, 1};

            handle.AddKernel(
                "Op2dTensorLite", network_config, program_name, "Op2dTensorLite", vld, vgd1, parms)(
                ATensor,
                static_cast<int>(astrides[1]), // a_cstride,
                BTensor,
                static_cast<int>(bstrides[1]), // b_cstride,
                CTensor,
                static_cast<int>(cstrides[1]), // c_cstride,
                miopen_alpha0,
                miopen_alpha1,
                miopen_beta,
                static_cast<int64_t>(Aoffset),
                static_cast<int64_t>(Boffset),
                static_cast<int64_t>(Coffset),
                static_cast<int64_t>(total_work),
                static_cast<int64_t>(total_work2),
                static_cast<int>(!float_equal(miopen_beta, 0.0)),
                static_cast<int>(blens[1] == 1));
        }
        else if(is_squashed)
        {
            parms += " -DUSE_2D_TENSOR_SQUASH";
            parms += " -DRD_BLCK=" + std::to_string(RD_BLCK) + " -DREAD_TYPE=" + READ_TYPE;

            const std::vector<size_t> vgd1{glb_sz, 1, 1};

            handle.AddKernel("Op2dTensorSquash",
                             network_config,
                             program_name,
                             "Op2dTensorSquash",
                             vld,
                             vgd1,
                             parms)(ATensor,
                                    BTensor,
                                    static_cast<int>(blens[1]),    // b_c,
                                    static_cast<int>(bstrides[1]), // b_cstride,
                                    CTensor,
                                    miopen_alpha0,
                                    miopen_alpha1,
                                    miopen_beta,
                                    static_cast<int64_t>(Aoffset),
                                    static_cast<int64_t>(Boffset),
                                    static_cast<int64_t>(Coffset),
                                    static_cast<int64_t>(total_work),
                                    static_cast<int>(!float_equal(miopen_alpha0, 0.0)),
                                    static_cast<int>(!float_equal(miopen_alpha1, 0.0)),
                                    static_cast<int>(!float_equal(miopen_beta, 0.0)));
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
                                    static_cast<int>(astrides[0]), // a_nstride,
                                    static_cast<int>(astrides[1]), // a_cstride,
                                    BTensor,
                                    static_cast<int>(blens[1]),    // b_c,
                                    static_cast<int>(blens[2]),    // b_h,
                                    static_cast<int>(bstrides[0]), // b_nstride,
                                    static_cast<int>(bstrides[1]), // b_cstride,
                                    CTensor,
                                    static_cast<int>(clens[1]),    // c_c,
                                    static_cast<int>(clens[2]),    // c_h,
                                    static_cast<int>(cstrides[0]), // c_nstride,
                                    static_cast<int>(cstrides[1]), // c_cstride,
                                    miopen_alpha0,
                                    miopen_alpha1,
                                    miopen_beta,
                                    bitmap,
                                    work_per_wg,
                                    static_cast<int64_t>(Aoffset),
                                    static_cast<int64_t>(Boffset),
                                    static_cast<int64_t>(Coffset),
                                    static_cast<int>(num_wg_orig));
        }
    });
}

void OpTensor4d(const Handle& handle,
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
    int num_wg      = first_not_one != blens.rend()
                          ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                          : 1;
    int work_per_wg = std::accumulate(clens.begin() + d, clens.end(), 1, std::multiplies<int>());

    unsigned int bitmap = 0;
    // update bitmap for first_not_one
    bitmap |= (1 << (bsize - d));

    // (d-2) is because distance starts from 1 and 0
    // also, we need to go past the "first_not_one" as that is already
    // accounted for in the bitmap
    CreateBitmapAndGrid(bitmap, blens, clens, num_wg, work_per_wg, static_cast<int>(d - 2));

    // quick fix for btensor = <1, 1, 1, 1>
    if(bTensorDesc.GetElementSize() == 1)
        bitmap = 4;

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
    bool leading_ones = IsBitmapLeadingOnes(bitmap, dims, static_cast<int>(d - 2));
    if(leading_ones && work_per_wg < 64)
    {
        local_threads = 64;
    }

    std::string program_name = "MIOpenTensorKernels.cl";

    const std::vector<size_t> vld{local_threads, 1, 1};

    // Special case for adding tensors in place
    size_t global_threads;
    global_threads =
        (static_cast<int>(leading_ones) == 1 && (d - 1) == 3) ? num_wg : num_wg * local_threads;
    global_threads = (global_threads < local_threads) ? local_threads : global_threads;

    const std::vector<size_t> vgd{global_threads, 1, 1};

    bool packed_tensor = true;

    // auto alens = aTensorDesc.GetLengths();
    packed_tensor &= aTensorDesc.IsPacked();
    packed_tensor &= bTensorDesc.IsPacked();
    packed_tensor &= cTensorDesc.IsPacked();

    bool packed_equal_tensor =
        packed_tensor && (bTensorDesc.GetElementSize() == cTensorDesc.GetElementSize());

#if(MIO_TENSOROCL_DEBUG == 1)
    printf("packed_tensor: %d\n", packed_tensor);
    printf("equal_tensor: %d\n", bTensorDesc.GetElementSize() == cTensorDesc.GetElementSize());
#endif

    // for naive tensor ops
    const std::string data_type = GetDataType(bTensorDesc.GetType());

    size_t TENS_LEN             = cTensorDesc.GetElementSize();
    size_t RD_BLCK              = (TENS_LEN % 4 == 0) ? 4 : (TENS_LEN % 2 == 0) ? 2 : 1;
    const std::string READ_TYPE = (RD_BLCK == 1) ? data_type : data_type + std::to_string(RD_BLCK);

    size_t total_work = std::max(TENS_LEN / RD_BLCK, size_t(1));
    size_t grp_sz     = (total_work + local_threads - 1) / local_threads;
    grp_sz            = std::min(size_t(max_num_wg), grp_sz);
    size_t glb_sz     = local_threads * grp_sz;

    std::string network_config{};
    network_config +=
        std::to_string(bTensorDesc.GetType()) + "-" + std::to_string(aTensorDesc.GetType()) + "-" +
        std::to_string(tensorOp) + "-" + std::to_string(max_num_wg) + "-" +
        ((fwd_conv_bias == 0 && packed_equal_tensor) ? "" : std::to_string(global_threads)) + "-" +
        std::to_string(local_threads);

    visit_float(bTensorDesc.GetType(), [&](auto as_float) {
        auto miopen_alpha0 = as_float(*(static_cast<const float*>(alpha0)));
        auto miopen_alpha1 = as_float(*(static_cast<const float*>(alpha1)));
        auto miopen_beta   = as_float(*(static_cast<const float*>(beta)));

        if(fwd_conv_bias != 0)
        {
            if(packed_tensor)
            {
                auto&& kernels = handle.GetKernels("OpTensorFwdBias", network_config);

                if(!kernels.empty())
                {
                    auto kernel = kernels.front();
                    kernel(ATensor,
                           BTensor,
                           static_cast<int>(blens[1]),
                           CTensor,
                           static_cast<int>(clens[0]),
                           static_cast<int>(cstrides[0]),
                           static_cast<int>(cstrides[1]),
                           work_per_wg,
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           static_cast<int64_t>(Aoffset),
                           static_cast<int64_t>(Boffset),
                           static_cast<int64_t>(Coffset),
                           static_cast<int>(num_wg_orig),
                           static_cast<int>(incr_wg));

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
                           static_cast<int>(astrides[0]),
                           static_cast<int>(astrides[1]),
                           static_cast<int>(astrides[2]),
                           BTensor,
                           static_cast<int>(blens[1]),
                           static_cast<int>(bstrides[1]),
                           CTensor,
                           static_cast<int>(clens[0]),
                           static_cast<int>(clens[3]),
                           static_cast<int>(cstrides[0]),
                           static_cast<int>(cstrides[1]),
                           static_cast<int>(cstrides[2]),
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           work_per_wg,
                           static_cast<int64_t>(Aoffset),
                           static_cast<int64_t>(Boffset),
                           static_cast<int64_t>(Coffset),
                           static_cast<int>(num_wg_orig),
                           static_cast<int>(incr_wg));
                    return;
                }
            }
        }
        // precede leading_ones for bitmap = 1,1,1,1
        else if(packed_equal_tensor)
        {
            network_config += "x" + std::to_string(grp_sz) + "x" + std::to_string(RD_BLCK);
            auto&& kernels = handle.GetKernels("Op4dTensorLite", network_config);
            if(!kernels.empty())
            {
                auto kernel = kernels.front();
                kernel(ATensor,
                       BTensor,
                       CTensor,
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       static_cast<int64_t>(Aoffset),
                       static_cast<int64_t>(Boffset),
                       static_cast<int64_t>(Coffset),
                       static_cast<int64_t>(total_work),
                       static_cast<int>(!float_equal(miopen_beta, 0.0)));
                return;
            }
        }
        else if(leading_ones)
        {
            if(packed_tensor)
            {

                auto&& kernels = handle.GetKernels("OpTensorLeadingOnes", network_config);

                if(!kernels.empty())
                {
                    auto kernel = kernels.front();
                    kernel(ATensor,
                           BTensor,
                           CTensor,
                           static_cast<int>(clens[1]),
                           static_cast<int>(clens[2]),
                           static_cast<int>(clens[3]),
                           static_cast<int>(cstrides[0]),
                           static_cast<int>(cstrides[1]),
                           work_per_wg,
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           static_cast<int64_t>(Aoffset),
                           static_cast<int64_t>(Boffset),
                           static_cast<int64_t>(Coffset),
                           static_cast<int>(num_wg_orig),
                           bitmap);

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
                           static_cast<int>(astrides[0]),
                           static_cast<int>(astrides[1]),
                           static_cast<int>(astrides[2]),
                           BTensor,
                           static_cast<int>(bstrides[0]),
                           static_cast<int>(bstrides[1]),
                           static_cast<int>(bstrides[2]),
                           CTensor,
                           static_cast<int>(clens[1]),
                           static_cast<int>(clens[2]),
                           static_cast<int>(clens[3]),
                           static_cast<int>(cstrides[0]),
                           static_cast<int>(cstrides[1]),
                           static_cast<int>(cstrides[2]),
                           miopen_alpha0,
                           miopen_alpha1,
                           miopen_beta,
                           work_per_wg,
                           static_cast<int64_t>(Aoffset),
                           static_cast<int64_t>(Boffset),
                           static_cast<int64_t>(Coffset),
                           static_cast<int>(num_wg_orig),
                           bitmap);
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
                       static_cast<int>(astrides[0]), // a_nstride,
                       static_cast<int>(astrides[1]), // a_cstride,
                       static_cast<int>(astrides[2]), // a_hstride,
                       BTensor,
                       static_cast<int>(blens[1]),    // b_c,
                       static_cast<int>(blens[2]),    // b_h,
                       static_cast<int>(blens[3]),    // b_w,
                       static_cast<int>(bstrides[0]), // b_nstride,
                       static_cast<int>(bstrides[1]), // b_cstride,
                       static_cast<int>(bstrides[2]), // b_hstride,
                       CTensor,
                       static_cast<int>(clens[1]),    // c_c,
                       static_cast<int>(clens[2]),    // c_h,
                       static_cast<int>(clens[3]),    // c_w,
                       static_cast<int>(cstrides[0]), // c_nstride,
                       static_cast<int>(cstrides[1]), // c_cstride,
                       static_cast<int>(cstrides[2]), // c_hstride,
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       bitmap,
                       work_per_wg,
                       static_cast<int64_t>(Aoffset),
                       static_cast<int64_t>(Boffset),
                       static_cast<int64_t>(Coffset),
                       static_cast<int>(num_wg_orig));
                return;
            }
        }

        std::string parms = " -DMIOPEN_TYPE=" + GetDataType(bTensorDesc.GetType()) +
                            " -DMAX_NUM_WG=" + std::to_string(max_num_wg);

        parms += GetDataTypeKernelParams(aTensorDesc.GetType());

        parms += " -DMIOPEN_TENSOR_OP=";
        switch(tensorOp)
        {
        case 0: parms += "miopenAdd"; break;
        case 1: parms += "miopenMul"; break;
        case 2: parms += "miopenMin"; break;
        case 3: parms += "miopenMax"; break;
        }

        if(fwd_conv_bias != 0)
        {
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
                                        static_cast<int>(blens[1]),
                                        CTensor,
                                        static_cast<int>(clens[0]),
                                        static_cast<int>(cstrides[0]),
                                        static_cast<int>(cstrides[1]),
                                        work_per_wg,
                                        miopen_alpha0,
                                        miopen_alpha1,
                                        miopen_beta,
                                        static_cast<int64_t>(Aoffset),
                                        static_cast<int64_t>(Boffset),
                                        static_cast<int64_t>(Coffset),
                                        static_cast<int>(num_wg_orig),
                                        static_cast<int>(incr_wg));
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
                                        static_cast<int>(astrides[0]),
                                        static_cast<int>(astrides[1]),
                                        static_cast<int>(astrides[2]),
                                        BTensor,
                                        static_cast<int>(blens[1]),
                                        static_cast<int>(bstrides[1]),
                                        CTensor,
                                        static_cast<int>(clens[0]),
                                        static_cast<int>(clens[3]),
                                        static_cast<int>(cstrides[0]),
                                        static_cast<int>(cstrides[1]),
                                        static_cast<int>(cstrides[2]),
                                        miopen_alpha0,
                                        miopen_alpha1,
                                        miopen_beta,
                                        work_per_wg,
                                        static_cast<int64_t>(Aoffset),
                                        static_cast<int64_t>(Boffset),
                                        static_cast<int64_t>(Coffset),
                                        static_cast<int>(num_wg_orig),
                                        static_cast<int>(incr_wg));
            }
        }
        // precede leading_ones for bitmap = 1,1,1,1
        else if(packed_equal_tensor)
        {
            parms += " -DUSE_4D_TENSOR_LITE";
            parms += " -DRD_BLCK=" + std::to_string(RD_BLCK) + " -DREAD_TYPE=" + READ_TYPE;

            const std::vector<size_t> vgd1{glb_sz, 1, 1};

            handle.AddKernel(
                "Op4dTensorLite", network_config, program_name, "Op4dTensorLite", vld, vgd1, parms)(
                ATensor,
                BTensor,
                CTensor,
                miopen_alpha0,
                miopen_alpha1,
                miopen_beta,
                static_cast<int64_t>(Aoffset),
                static_cast<int64_t>(Boffset),
                static_cast<int64_t>(Coffset),
                static_cast<int64_t>(total_work),
                static_cast<int>(!float_equal(miopen_beta, 0.0)));
        }
        else if(leading_ones)
        {
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
                                        static_cast<int>(clens[1]),
                                        static_cast<int>(clens[2]),
                                        static_cast<int>(clens[3]),
                                        static_cast<int>(cstrides[0]),
                                        static_cast<int>(cstrides[1]),
                                        work_per_wg,
                                        miopen_alpha0,
                                        miopen_alpha1,
                                        miopen_beta,
                                        static_cast<int64_t>(Aoffset),
                                        static_cast<int64_t>(Boffset),
                                        static_cast<int64_t>(Coffset),
                                        static_cast<int>(num_wg_orig),
                                        bitmap);
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
                                        static_cast<int>(astrides[0]),
                                        static_cast<int>(astrides[1]),
                                        static_cast<int>(astrides[2]),
                                        BTensor,
                                        static_cast<int>(bstrides[0]),
                                        static_cast<int>(bstrides[1]),
                                        static_cast<int>(bstrides[2]),
                                        CTensor,
                                        static_cast<int>(clens[1]),
                                        static_cast<int>(clens[2]),
                                        static_cast<int>(clens[3]),
                                        static_cast<int>(cstrides[0]),
                                        static_cast<int>(cstrides[1]),
                                        static_cast<int>(cstrides[2]),
                                        miopen_alpha0,
                                        miopen_alpha1,
                                        miopen_beta,
                                        work_per_wg,
                                        static_cast<int64_t>(Aoffset),
                                        static_cast<int64_t>(Boffset),
                                        static_cast<int64_t>(Coffset),
                                        static_cast<int>(num_wg_orig),
                                        bitmap);
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
                                    static_cast<int>(astrides[0]), // a_nstride,
                                    static_cast<int>(astrides[1]), // a_cstride,
                                    static_cast<int>(astrides[2]), // a_hstride,
                                    BTensor,
                                    static_cast<int>(blens[1]),    // b_c,
                                    static_cast<int>(blens[2]),    // b_h,
                                    static_cast<int>(blens[3]),    // b_w,
                                    static_cast<int>(bstrides[0]), // b_nstride,
                                    static_cast<int>(bstrides[1]), // b_cstride,
                                    static_cast<int>(bstrides[2]), // b_hstride,
                                    CTensor,
                                    static_cast<int>(clens[1]),    // c_c,
                                    static_cast<int>(clens[2]),    // c_h,
                                    static_cast<int>(clens[3]),    // c_w,
                                    static_cast<int>(cstrides[0]), // c_nstride,
                                    static_cast<int>(cstrides[1]), // c_cstride,
                                    static_cast<int>(cstrides[2]), // c_hstride,
                                    miopen_alpha0,
                                    miopen_alpha1,
                                    miopen_beta,
                                    bitmap,
                                    work_per_wg,
                                    static_cast<int64_t>(Aoffset),
                                    static_cast<int64_t>(Boffset),
                                    static_cast<int64_t>(Coffset),
                                    static_cast<int>(num_wg_orig));
        }
    });
}

void OpTensorOther(const Handle& handle,
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

    const bool case_1d = bsize == 1;
    const bool case_2d = bsize == 2;
    const bool case_5d = bsize == 5;

    const bool use_hip = case_1d || case_2d;

    // first_not_one is incorrect if btensor size equal to 1
    auto first_not_one = std::find_if(blens.rbegin(), blens.rend(), [](int i) { return i != 1; });
    auto d             = std::distance(blens.begin(), first_not_one.base());

    // quick fix
    int num_wg      = first_not_one != blens.rend()
                          ? static_cast<int>(*first_not_one == 0 ? 1 : *first_not_one)
                          : 1;
    int work_per_wg = std::accumulate(clens.begin() + d, clens.end(), 1, std::multiplies<int>());

    unsigned int bitmap = 0;
    // update bitmap for first_not_one
    bitmap |= (1 << (bsize - d));

    // (d-2) is because distance starts from 1 and 0
    // also, we need to go past the "first_not_one" as that is already
    // accounted for in the bitmap
    CreateBitmapAndGrid(bitmap, blens, clens, num_wg, work_per_wg, static_cast<int>(d - 2));

#if(MIO_TENSOROCL_DEBUG == 1)
    printf("bitmap: %u\n", bitmap);
    printf("work_per_wg: %d, num_wg: %d\n", work_per_wg, num_wg);
#endif

    int num_wg_orig = num_wg;
    int max_num_wg  = 4096;

    size_t local_threads = 256;

    if(case_2d)
        local_threads = 32;

    if(case_1d)
        num_wg = std::clamp(clens[0] / local_threads, size_t(1), size_t(max_num_wg));
    if(case_2d)
        num_wg = std::clamp((clens[0] * clens[1]) / local_threads, size_t(1), size_t(max_num_wg));
    num_wg = num_wg > max_num_wg ? max_num_wg : num_wg;

    const std::vector<size_t> vld{local_threads, 1, 1};

    // Special case for adding tensors in place
    size_t global_threads;
    global_threads = num_wg * local_threads;

    const std::vector<size_t> vgd{global_threads, 1, 1};

    std::string program_name = use_hip ? "MIOpenTensorKernelsHip.cpp" : "MIOpenTensorKernels.cl";

    std::string network_config{};
    network_config += std::to_string(bTensorDesc.GetType()) + "-" +
                      std::to_string(aTensorDesc.GetType()) + "-" + std::to_string(tensorOp) + "-" +
                      std::to_string(global_threads) + "-" + std::to_string(local_threads);

    visit_float(bTensorDesc.GetType(), [&](auto as_float) {
        auto miopen_alpha0 = as_float(*(static_cast<const float*>(alpha0)));
        auto miopen_alpha1 = as_float(*(static_cast<const float*>(alpha1)));
        auto miopen_beta   = as_float(*(static_cast<const float*>(beta)));

        if(case_5d)
        {
            auto&& kernels = handle.GetKernels("Op5dTensorGeneric", network_config);

            if(!kernels.empty())
            {
                auto kernel = kernels.front();
                kernel(ATensor,
                       static_cast<int>(astrides[0]),
                       static_cast<int>(astrides[1]),
                       static_cast<int>(astrides[2]),
                       static_cast<int>(astrides[3]),
                       BTensor,
                       static_cast<int>(blens[1]),    // b_c,
                       static_cast<int>(blens[2]),    // b_d,
                       static_cast<int>(blens[3]),    // b_h,
                       static_cast<int>(blens[4]),    // b_w,
                       static_cast<int>(bstrides[0]), // b_nstride,
                       static_cast<int>(bstrides[1]), // b_cstride,
                       static_cast<int>(bstrides[2]), // b_dstride,
                       static_cast<int>(bstrides[3]), // b_hstride,
                       CTensor,
                       static_cast<int>(clens[1]),    // c_c,
                       static_cast<int>(clens[2]),    // c_d,
                       static_cast<int>(clens[3]),    // c_h,
                       static_cast<int>(clens[4]),    // c_w,
                       static_cast<int>(cstrides[0]), // c_nstride,
                       static_cast<int>(cstrides[1]), // c_cstride,
                       static_cast<int>(cstrides[2]), // c_dstride,
                       static_cast<int>(cstrides[3]), // c_hstride,
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       bitmap,
                       work_per_wg,
                       static_cast<int64_t>(Aoffset),
                       static_cast<int64_t>(Boffset),
                       static_cast<int64_t>(Coffset),
                       static_cast<int>(num_wg_orig));
                return;
            }
        }
        else if(case_2d)
        {
            auto&& kernels = handle.GetKernels("Op2dTensorGeneric", network_config);

            if(!kernels.empty())
            {
                auto kernel = kernels.front();
                kernel(ATensor,
                       BTensor,
                       CTensor,
                       static_cast<long>(Aoffset),
                       static_cast<long>(Boffset),
                       static_cast<long>(Coffset),
                       static_cast<uint32_t>(blens[1] == 1 ? clens[1] : blens[1]),
                       static_cast<uint32_t>(clens[1]),
                       static_cast<uint32_t>(astrides[0]),
                       static_cast<uint32_t>(astrides[1]),
                       static_cast<uint32_t>(blens[0] == 1 ? 0 : bstrides[0]),
                       static_cast<uint32_t>(blens[1] == 1 ? 0 : bstrides[1]),
                       static_cast<uint32_t>(cstrides[0]),
                       static_cast<uint32_t>(cstrides[1]),
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       static_cast<uint32_t>(clens[0]),
                       !float_equal(miopen_beta, 0.0));
                return;
            }
        }
        else if(case_1d)
        {
            auto&& kernels = handle.GetKernels("Op1dTensorGeneric", network_config);

            if(!kernels.empty())
            {

                auto kernel = kernels.front();
                kernel(ATensor,
                       BTensor,
                       CTensor,
                       static_cast<uint64_t>(Aoffset),
                       static_cast<uint64_t>(Boffset),
                       static_cast<uint64_t>(Coffset),
                       static_cast<uint32_t>(astrides[0]),
                       static_cast<uint32_t>(blens[0] == 1 ? 0 : bstrides[0]),
                       static_cast<uint32_t>(cstrides[0]),
                       miopen_alpha0,
                       miopen_alpha1,
                       miopen_beta,
                       static_cast<uint32_t>(clens[0]),
                       !float_equal(miopen_beta, 0.0));
                return;
            }
        }

        std::string parms = " -DMIOPEN_TYPE=" + GetDataType(bTensorDesc.GetType()) +
                            " -DMAX_NUM_WG=" + std::to_string(max_num_wg);

        parms += GetDataTypeKernelParams(aTensorDesc.GetType());

        parms += " -DMIOPEN_TENSOR_OP=";
        switch(tensorOp)
        {
        case 0: parms += "miopenAdd"; break;
        case 1: parms += "miopenMul"; break;
        case 2: parms += "miopenMin"; break;
        case 3: parms += "miopenMax"; break;
        }

        if(case_5d)
        {
            parms += " -DUSE_5D_TENSOR_GENERIC";

            handle.AddKernel("Op5dTensorGeneric",
                             network_config,
                             program_name,
                             "Op5dTensorGeneric",
                             vld,
                             vgd,
                             parms)(ATensor,
                                    static_cast<int>(astrides[0]),
                                    static_cast<int>(astrides[1]),
                                    static_cast<int>(astrides[2]),
                                    static_cast<int>(astrides[3]),
                                    BTensor,
                                    static_cast<int>(blens[1]),    // b_c,
                                    static_cast<int>(blens[2]),    // b_d,
                                    static_cast<int>(blens[3]),    // b_h,
                                    static_cast<int>(blens[4]),    // b_w,
                                    static_cast<int>(bstrides[0]), // b_nstride,
                                    static_cast<int>(bstrides[1]), // b_cstride,
                                    static_cast<int>(bstrides[2]), // b_dstride,
                                    static_cast<int>(bstrides[3]), // b_hstride,
                                    CTensor,
                                    static_cast<int>(clens[1]),    // c_c,
                                    static_cast<int>(clens[2]),    // c_d,
                                    static_cast<int>(clens[3]),    // c_h,
                                    static_cast<int>(clens[4]),    // c_w,
                                    static_cast<int>(cstrides[0]), // c_nstride,
                                    static_cast<int>(cstrides[1]), // c_cstride,
                                    static_cast<int>(cstrides[2]), // c_dstride,
                                    static_cast<int>(cstrides[3]), // c_hstride,
                                    miopen_alpha0,
                                    miopen_alpha1,
                                    miopen_beta,
                                    bitmap,
                                    work_per_wg,
                                    static_cast<int64_t>(Aoffset),
                                    static_cast<int64_t>(Boffset),
                                    static_cast<int64_t>(Coffset),
                                    static_cast<int>(num_wg_orig));
        }
        else if(case_2d)
        {
            parms += " -DUSE_2D_TENSOR_GENERIC";

            handle.AddKernel("Op2dTensorGeneric",
                             network_config,
                             program_name,
                             "Op2dTensorGeneric",
                             vld,
                             vgd,
                             parms)(ATensor,
                                    BTensor,
                                    CTensor,
                                    static_cast<long>(Aoffset),
                                    static_cast<long>(Boffset),
                                    static_cast<long>(Coffset),
                                    static_cast<uint32_t>(blens[1] == 1 ? clens[1] : blens[1]),
                                    static_cast<uint32_t>(clens[1]),
                                    static_cast<uint32_t>(astrides[0]),
                                    static_cast<uint32_t>(astrides[1]),
                                    static_cast<uint32_t>(blens[0] == 1 ? 0 : bstrides[0]),
                                    static_cast<uint32_t>(blens[1] == 1 ? 0 : bstrides[1]),
                                    static_cast<uint32_t>(cstrides[0]),
                                    static_cast<uint32_t>(cstrides[1]),
                                    miopen_alpha0,
                                    miopen_alpha1,
                                    miopen_beta,
                                    static_cast<uint32_t>(clens[0]),
                                    !float_equal(miopen_beta, 0.0));
        }
        else if(case_1d)
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
                                    CTensor,
                                    static_cast<uint64_t>(Aoffset),
                                    static_cast<uint64_t>(Boffset),
                                    static_cast<uint64_t>(Coffset),
                                    static_cast<uint32_t>(astrides[0]),
                                    static_cast<uint32_t>(blens[0] == 1 ? 0 : bstrides[0]),
                                    static_cast<uint32_t>(cstrides[0]),
                                    miopen_alpha0,
                                    miopen_alpha1,
                                    miopen_beta,
                                    static_cast<uint32_t>(clens[0]),
                                    !float_equal(miopen_beta, 0.0));
        }
    });
}

void OpTensor(const Handle& handle,
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
              const size_t Coffset,
              bool nonStandardSquash)
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

    if(!nonStandardSquash)
    {
        for(std::size_t i = 0; i < clens.size(); i++)
        {
            if(blens[i] != 1 && blens[i] != clens[i])
            {
                MIOPEN_THROW("BTensor dim != 1 && BTensor dim != CTensor dim: " +
                             std::to_string(i));
            }
        }
    }
    else
    {
        // non standard behavior because blens[1] can be not equalt to clens[1]
        if(!(clens.size() == 3 && blens[0] == 1 && clens[0] == 1 && blens[2] == clens[2]))
        {
            MIOPEN_THROW("Non standard squashed operation supported only for 3d tensors and for "
                         "the specific configuration");
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
                   Coffset,
                   nonStandardSquash);
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

struct two_exp_ceiling_t
{
    std::size_t operator()(std::size_t n) const
    {
        assert(n > 0);

        std::size_t i = 1;

        n--;
        while(n != 0)
        {
            i *= 2;
            n /= 2;
        }

        return i;
    }
};

static std::vector<std::size_t> get_worker_sizes(const std::vector<std::size_t>& data_sizes)
{
    const std::size_t dim = data_sizes.size();

    std::vector<std::size_t> worker_sizes(dim);

    std::transform(data_sizes.begin(), data_sizes.end(), worker_sizes.begin(), two_exp_ceiling_t{});

    std::size_t wgd = std::accumulate(
        worker_sizes.begin(), worker_sizes.end(), std::size_t{1}, std::multiplies<std::size_t>());

    if(wgd > 65536)
    {
        std::size_t n = wgd / 65536;

        int i = 0;
        while(n > 1 && i < dim)
        {
            std::size_t size_old = worker_sizes[i];
            worker_sizes[i]      = (size_old - 1) / n + 1;
            n /= size_old / worker_sizes[i];
            ++i;
        }
    }

    return worker_sizes;
}

void SetTensor(const Handle& handle,
               const TensorDescriptor& yDesc,
               Data_t y,
               const void* alpha,
               const int offset)
{
    if(y == nullptr || alpha == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    const TensorDescriptor yDesc_flat = GetFlattenedTensorDescriptor(yDesc);

#ifndef NDEBUG
    if(yDesc.GetNumDims() != yDesc_flat.GetNumDims())
    {
        MIOPEN_LOG_I2("real descriptor: " << yDesc);
        MIOPEN_LOG_I2("flat descriptor: " << yDesc_flat);
    }
#endif

    const std::size_t yDim_flat = yDesc_flat.GetNumDims();

    assert(yDim_flat > 0 && yDim_flat <= 5);

    std::string kernel_name = "SubTensorOpWithScalar" + std::to_string(yDim_flat) + "d";

    const miopenDataType_t dataType = yDesc_flat.GetType();

    std::string network_config = "set " + std::to_string(dataType);
    for(auto& len : yDesc_flat.GetLengths())
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
        std::string program_name = "MIOpenSubTensorOpWithScalarKernel.cl";

        std::vector<std::size_t> worker_sizes = get_worker_sizes(yDesc_flat.GetLengths());

        std::size_t wgd = std::accumulate(worker_sizes.begin(),
                                          worker_sizes.end(),
                                          std::size_t{1},
                                          std::multiplies<std::size_t>());

        std::size_t wld = 256 < wgd ? 256 : wgd;
        std::stringstream ss;
        ss << "-DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET"
           << GetDataTypeKernelParams(dataType);
        for(int i = 0; i < yDim_flat; ++i)
        {
            ss << " -DWORK_LENGTH_" << std::to_string(i) << "=" << std::to_string(worker_sizes[i]);
        }

        kernel = handle.AddKernel(kernel_name,
                                  network_config,
                                  program_name,
                                  kernel_name,
                                  {wld, 1, 1},
                                  {wgd, 1, 1},
                                  ss.str());
    }

    switch(yDim_flat)
    {
    case 1: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]));
        });

        break;
    }
    case 2: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]));
        });

        break;
    }
    case 3: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]));
        });

        break;
    }
    case 4: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetStrides()[3]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[3]));
        });

        break;
    }
    case 5: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetStrides()[3]),
                   static_cast<int>(yDesc_flat.GetStrides()[4]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[3]),
                   static_cast<int>(yDesc_flat.GetLengths()[4]));
        });

        break;
    }
    default: assert(false);
    }
}

void ScaleTensor(const Handle& handle,
                 const TensorDescriptor& yDesc,
                 Data_t y,
                 const void* alpha,
                 const int offset)
{
    if(y == nullptr || alpha == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    const TensorDescriptor yDesc_flat = GetFlattenedTensorDescriptor(yDesc);

#ifndef NDEBUG
    if(yDesc.GetNumDims() != yDesc_flat.GetNumDims())
    {
        MIOPEN_LOG_I2("real descriptor: " << yDesc);
        MIOPEN_LOG_I2("flat descriptor: " << yDesc_flat);
    }
#endif

    const std::size_t yDim_flat = yDesc_flat.GetNumDims();

    assert(yDim_flat > 0 && yDim_flat <= 5);

    const miopenDataType_t dataType = yDesc_flat.GetType();

    if(!(dataType == miopenHalf     //
         || dataType == miopenFloat //
         || dataType == miopenInt32 //
         || dataType == miopenDouble))
    {
        MIOPEN_THROW(miopenStatusBadParm, "ScaleTensor: unsupported data type.");
    }

    std::string kernel_name = "SubTensorOpWithScalar" + std::to_string(yDim_flat) + "d";

    const std::vector<std::size_t>& lens = yDesc_flat.GetLengths();

    std::string network_config = "scale " + std::to_string(yDesc_flat.GetType());
    for(auto& len : lens)
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
        std::string program_name = "MIOpenSubTensorOpWithScalarKernel.cl";

        std::vector<std::size_t> worker_sizes = get_worker_sizes(lens);

        std::size_t wgd = std::accumulate(worker_sizes.begin(),
                                          worker_sizes.end(),
                                          std::size_t{1},
                                          std::multiplies<std::size_t>());

        std::size_t wld = 256 < wgd ? 256 : wgd;

        std::string parms = "-DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_MULTIPLY" +
                            GetDataTypeKernelParams(dataType);
        for(int i = 0; i < yDim_flat; ++i)
        {
            parms += " -DWORK_LENGTH_" + std::to_string(i) + "=" + std::to_string(worker_sizes[i]);
        }

        kernel = handle.AddKernel(kernel_name,
                                  network_config,
                                  program_name,
                                  kernel_name,
                                  {wld, 1, 1},
                                  {wgd, 1, 1},
                                  parms);
    }

    switch(yDim_flat)
    {
    case 1: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]));
        });

        break;
    }
    case 2: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]));
        });

        break;
    }
    case 3: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]));
        });

        break;
    }
    case 4: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetStrides()[3]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[3]));
        });

        break;
    }
    case 5: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetStrides()[3]),
                   static_cast<int>(yDesc_flat.GetStrides()[4]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[3]),
                   static_cast<int>(yDesc_flat.GetLengths()[4]));
        });

        break;
    }
    default: assert(false);
    }
}

void CopyTensor(const Handle& handle,
                const TensorDescriptor& srcDesc,
                ConstData_t src,
                const TensorDescriptor& dstDesc,
                Data_t dst,
                int srcOffset,
                int dstOffset,
                bool forseAsync)
{
    if(src == nullptr || dst == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    if(srcDesc.GetType() != dstDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
    }

    if(srcDesc.GetLengths() != dstDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    auto flat_descriptors = GetConsistentFlattenedTensorDescriptors(srcDesc, dstDesc);
    const TensorDescriptor& srcDesc_flat = std::get<0>(flat_descriptors);
    const TensorDescriptor& dstDesc_flat = std::get<1>(flat_descriptors);

#ifndef NDEBUG
    if(srcDesc.GetNumDims() != srcDesc_flat.GetNumDims())
    {
        MIOPEN_LOG_I2("src real descriptor: " << srcDesc);
        MIOPEN_LOG_I2("src flat descriptor: " << srcDesc_flat);
        MIOPEN_LOG_I2("dst real descriptor: " << dstDesc);
        MIOPEN_LOG_I2("dst flat descriptor: " << dstDesc_flat);
    }
#endif

    std::size_t srcDim_flat = srcDesc_flat.GetNumDims();

    if(srcDim_flat < 1 || srcDim_flat > 5)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension sizes unsupported.");
    }

    if(forseAsync || srcOffset > 0 || dstOffset > 0 ||
       (!(srcDesc_flat.IsPacked() && dstDesc_flat.IsPacked())))
    {
        std::string kernel_name = "SubTensorOpWithSubTensor" + std::to_string(srcDim_flat) + "d";

        const std::vector<std::size_t>& lens = srcDesc_flat.GetLengths();

        std::string network_config = "copy " + std::to_string(srcDesc_flat.GetType());
        for(auto& len : lens)
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
            std::string program_name = "MIOpenSubTensorOpWithSubTensorKernel.cl";

            std::vector<std::size_t> worker_sizes = get_worker_sizes(lens);

            std::size_t wgd = std::accumulate(worker_sizes.begin(),
                                              worker_sizes.end(),
                                              std::size_t{1},
                                              std::multiplies<std::size_t>());

            std::size_t wld = 256 < wgd ? 256 : wgd;

            std::string parms = "-DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY" +
                                GetDataTypeKernelParams(srcDesc_flat.GetType());
            for(std::size_t i = 0; i < srcDim_flat; ++i)
            {
                parms +=
                    " -DWORK_LENGTH_" + std::to_string(i) + "=" + std::to_string(worker_sizes[i]);
            }

            kernel = handle.AddKernel(kernel_name,
                                      network_config,
                                      program_name,
                                      kernel_name,
                                      {wld, 1, 1},
                                      {wgd, 1, 1},
                                      parms);
        }

        switch(srcDim_flat)
        {
        case 1: {
            kernel(src,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]));

            break;
        }
        case 2: {
            kernel(src,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]));

            break;
        }
        case 3: {
            kernel(src,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]));

            break;
        }
        case 4: {
            kernel(src,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetStrides()[3]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[3]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]),
                   static_cast<int>(dstDesc_flat.GetStrides()[3]));

            break;
        }
        case 5: {
            kernel(src,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetStrides()[3]),
                   static_cast<int>(srcDesc_flat.GetStrides()[4]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[3]),
                   static_cast<int>(srcDesc_flat.GetLengths()[4]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]),
                   static_cast<int>(dstDesc_flat.GetStrides()[3]),
                   static_cast<int>(dstDesc_flat.GetStrides()[4]));

            break;
        }
        default: assert(false);
        }
    }
    else
    {
        handle.Copy(src, dst, srcDesc_flat.GetElementSize() * GetTypeSize(srcDesc_flat.GetType()));
    }
}

std::string GetCastTensorBuildOptionFromType(const std::string& buildOption, miopenDataType_t type)
{
    std::string option(buildOption);
    switch(type)
    {
    case miopenInt8: return option += "0";
    case miopenInt32: return option += "1";
    case miopenHalf: return option += "2";
    case miopenFloat: return option += "3";
    case miopenBFloat16: return option += "4";
    case miopenFloat8:
        MIOPEN_THROW(miopenStatusBadParm, "miopenFloat8 data type not supported in cast tensor.");
    case miopenBFloat8:
        MIOPEN_THROW(miopenStatusBadParm, "miopenBFloat8 data type not supported in cast tensor.");
    case miopenDouble:
        // TODO
        MIOPEN_THROW(miopenStatusBadParm, "miopenDouble data type not supported in cast tensor.");
    case miopenInt64:
        MIOPEN_THROW(miopenStatusBadParm, "miopenInt64 data type not supported in cast tensor.");
    default: MIOPEN_THROW(miopenStatusBadParm, "Invalid data type in cast tensor desc.");
    }
}

void CastTensor(const Handle& handle,
                const void* alpha,
                const bool clamping,
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

    if(srcDesc.GetLengths() != dstDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    auto flat_descriptors = GetConsistentFlattenedTensorDescriptors(srcDesc, dstDesc);
    const TensorDescriptor& srcDesc_flat = std::get<0>(flat_descriptors);
    const TensorDescriptor& dstDesc_flat = std::get<1>(flat_descriptors);

#ifndef NDEBUG
    if(srcDesc.GetNumDims() != srcDesc_flat.GetNumDims())
    {
        MIOPEN_LOG_I2("src real descriptor: " << srcDesc);
        MIOPEN_LOG_I2("src flat descriptor: " << srcDesc_flat);
        MIOPEN_LOG_I2("dst real descriptor: " << dstDesc);
        MIOPEN_LOG_I2("dst flat descriptor: " << dstDesc_flat);
    }
#endif

    std::size_t srcDim_flat = srcDesc_flat.GetNumDims();

    if(srcDim_flat < 1 || srcDim_flat > 5)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension sizes unsupported.");
    }

    if(srcDesc.GetType() == dstDesc.GetType() && srcOffset == 0 && dstOffset == 0 &&
       srcDesc_flat.IsPacked() && dstDesc_flat.IsPacked())
    {
        handle.Copy(src, dst, srcDesc_flat.GetElementSize() * GetTypeSize(srcDesc_flat.GetType()));
    }
    else
    {
        std::string kernel_name = "SubTensorOpWithCastTensor" + std::to_string(srcDim_flat) + "d";

        const std::vector<std::size_t>& lens = srcDesc_flat.GetLengths();

        std::string network_config = "cast " + std::to_string(dstDesc_flat.GetType());
        for(auto& len : lens)
        {
            network_config += " " + std::to_string(len);
        }

        auto&& kernels = handle.GetKernels(kernel_name, network_config);
        KernelInvoke kernel;

        auto miopen_alpha = *(static_cast<const float*>(alpha));

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenSubTensorOpWithCastTensorKernel.cl";

            std::vector<std::size_t> worker_sizes = get_worker_sizes(lens);

            std::size_t wgd = std::accumulate(worker_sizes.begin(),
                                              worker_sizes.end(),
                                              std::size_t{1},
                                              std::multiplies<std::size_t>());

            std::size_t wld = 256 < wgd ? 256 : wgd;

            std::string parms =
                GetCastTensorBuildOptionFromType(" -DMIOPEN_SRC_TYPE=", srcDesc_flat.GetType()) +
                GetCastTensorBuildOptionFromType(" -DMIOPEN_DST_TYPE=", dstDesc_flat.GetType());

            for(std::size_t i = 0; i < srcDim_flat; ++i)
            {
                parms +=
                    " -DWORK_LENGTH_" + std::to_string(i) + "=" + std::to_string(worker_sizes[i]);
            }

            if(dstDesc_flat.GetType() == miopenBFloat16)
            {
                parms += " -DMIOPEN_USE_RNE_BFLOAT16=1";
            }

            kernel = handle.AddKernel(kernel_name,
                                      network_config,
                                      program_name,
                                      kernel_name,
                                      {wld, 1, 1},
                                      {wgd, 1, 1},
                                      parms);
        }

        const int clamping_arg = clamping ? 1 : 0;
        switch(srcDim_flat)
        {
        case 1: {
            kernel(src,
                   miopen_alpha,
                   clamping_arg,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]));

            break;
        }
        case 2: {
            kernel(src,
                   miopen_alpha,
                   clamping_arg,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]));

            break;
        }
        case 3: {
            kernel(src,
                   miopen_alpha,
                   clamping_arg,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]));

            break;
        }
        case 4: {
            kernel(src,
                   miopen_alpha,
                   clamping_arg,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetStrides()[3]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[3]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]),
                   static_cast<int>(dstDesc_flat.GetStrides()[3]));

            break;
        }
        case 5: {
            kernel(src,
                   miopen_alpha,
                   clamping_arg,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetStrides()[3]),
                   static_cast<int>(srcDesc_flat.GetStrides()[4]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[3]),
                   static_cast<int>(srcDesc_flat.GetLengths()[4]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]),
                   static_cast<int>(dstDesc_flat.GetStrides()[3]),
                   static_cast<int>(dstDesc_flat.GetStrides()[4]));

            break;
        }
        default: assert(false);
        }
    }
}

void TransformTensor(const Handle& handle,
                     const void* alpha,
                     const TensorDescriptor& xDesc,
                     ConstData_t x,
                     const void* beta,
                     const TensorDescriptor& yDesc,
                     Data_t y,
                     size_t Xoffset,
                     size_t Yoffset)
{
    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(alpha == nullptr || beta == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    auto x_len = xDesc.GetLengths();
    auto y_len = yDesc.GetLengths();

    if(x_len.size() != y_len.size())
    {
        MIOPEN_THROW("Tensor dimension must be the same");
    }

    if(x_len[0] != y_len[0])
    {
        MIOPEN_THROW("Tensor x and y batch sizes do not match");
    }

    const auto is_alpha_one = float_equal(*(static_cast<const float*>(alpha)), 1);
    const auto is_beta_zero = float_equal(*(static_cast<const float*>(beta)), 0);

    if(xDesc.GetType() == miopenInt8 && yDesc.GetType() == miopenInt8 && x_len.size() >= 3)
    {
        if(x_len[1] <= y_len[1])
        {
            if(x_len[1] <= (y_len[1] - 4) || y_len[1] % 4 != 0)
            {
                MIOPEN_THROW("Invalid y channel size");
            }

            int8_t zero = 0;
            SetTensor(handle, yDesc, y, &zero);
        }
        else if(x_len[1] % 4 != 0)
        {
            MIOPEN_THROW("Invalid x channel size");
        }

        size_t batch_n = x_len[0];

        x_len[0] = 1;
        y_len[0] = 1;

        miopen::TensorDescriptor x_batch_desc, y_batch_desc;
        x_batch_desc = miopen::TensorDescriptor(miopenInt8, x_len);
        y_batch_desc = miopen::TensorDescriptor(miopenInt8, y_len);

        size_t x_batch_sz = x_batch_desc.GetElementSize();
        size_t y_batch_sz = y_batch_desc.GetElementSize();

        for(size_t i = 0; i < batch_n; i++)
        {
            size_t x_offset = i * x_batch_sz;
            size_t y_offset = i * y_batch_sz;

            if(is_alpha_one && is_beta_zero)
            {
                CopyTensor(handle,
                           ((x_len[1] <= y_len[1]) ? x_batch_desc : y_batch_desc),
                           x,
                           ((x_len[1] <= y_len[1]) ? x_batch_desc : y_batch_desc),
                           y,
                           x_offset,
                           y_offset);
            }
            else
            {
                MIOPEN_THROW(miopenStatusNotImplemented,
                             "y=alpha*x+beta*y is not supported for int8 yet");
            }
        }
    }
    else
    {
        auto x_y_len          = boost::combine(x_len, y_len);
        bool same_spatial_len = std::all_of(x_y_len.begin(), x_y_len.end(), [](auto v) {
            return boost::get<0>(v) == boost::get<1>(v);
        });

        if(!same_spatial_len)
        {
            MIOPEN_THROW("Tensor x and y spatial sizes do not match");
        }

        auto flat_descriptors              = GetConsistentFlattenedTensorDescriptors(xDesc, yDesc);
        const TensorDescriptor& xDesc_flat = std::get<0>(flat_descriptors);
        const TensorDescriptor& yDesc_flat = std::get<1>(flat_descriptors);

        if(xDesc.GetNumDims() != xDesc_flat.GetNumDims())
        {
            MIOPEN_LOG_I2("x real descriptor: " << xDesc);
            MIOPEN_LOG_I2("x flat descriptor: " << xDesc_flat);
        }

        if(yDesc.GetNumDims() != yDesc_flat.GetNumDims())
        {
            MIOPEN_LOG_I2("y real descriptor: " << yDesc);
            MIOPEN_LOG_I2("y flat descriptor: " << yDesc_flat);
        }

        const std::size_t yDim_flat = yDesc_flat.GetNumDims();

        assert(yDim_flat > 0 && yDim_flat <= 5);

        const miopenDataType_t dataTypex = xDesc_flat.GetType();
        const miopenDataType_t dataTypey = yDesc_flat.GetType();

        if(!(dataTypex == miopenHalf        //
             || dataTypex == miopenFloat    //
             || dataTypex == miopenInt32    //
             || dataTypex == miopenBFloat16 //
             || dataTypex == miopenDouble))
        {
            MIOPEN_THROW("Tensor x is a unsupported data type");
        }

        if(!(dataTypey == miopenHalf        //
             || dataTypey == miopenFloat    //
             || dataTypey == miopenInt32    //
             || dataTypey == miopenBFloat16 //
             || dataTypey == miopenDouble))
        {
            MIOPEN_THROW("Tensor y is a unsupported data type");
        }

        if(dataTypex != dataTypey)
        {
            MIOPEN_THROW("Tensor x and y have different data types");
        }

        std::string kernel_name = "SubTensorOpWithTransform" + std::to_string(yDim_flat) + "d";

        const std::vector<std::size_t>& lens = yDesc_flat.GetLengths();

        std::string network_config = "transform " + std::to_string(yDesc_flat.GetType());
        for(auto& len : lens)
        {
            network_config += "x" + std::to_string(len);
        }

        if(is_beta_zero)
            network_config += "xBETA_IS_ZERO";
        if(is_alpha_one)
            network_config += "xALPHA_IS_ONE";

        auto&& kernels = handle.GetKernels(kernel_name, network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenSubTensorOpWithTransformKernel.cl";

            std::vector<std::size_t> worker_sizes = get_worker_sizes(lens);

            std::size_t wgd = std::accumulate(worker_sizes.begin(),
                                              worker_sizes.end(),
                                              std::size_t{1},
                                              std::multiplies<std::size_t>());

            std::size_t wld = 256 < wgd ? 256 : wgd;

            std::string parms =
                GetDataTypeKernelParams(dataTypey)                                           //
                + " -DMIOPEN_BETA_IS_ZERO=" + std::to_string(static_cast<int>(is_beta_zero)) //
                + " -DMIOPEN_ALPHA_IS_ONE=" + std::to_string(static_cast<int>(is_alpha_one));

            for(int i = 0; i < yDim_flat; ++i)
            {
                parms +=
                    " -DWORK_LENGTH_" + std::to_string(i) + "=" + std::to_string(worker_sizes[i]);
            }

            kernel = handle.AddKernel(kernel_name,
                                      network_config,
                                      program_name,
                                      kernel_name,
                                      {wld, 1, 1},
                                      {wgd, 1, 1},
                                      parms);
        }

        switch(yDim_flat)
        {
        case 1: {
            visit_float(dataTypey, [&](auto as_float) {
                kernel(x,
                       *as_float(alpha),
                       y,
                       *as_float(beta),
                       static_cast<unsigned>(Xoffset),
                       static_cast<unsigned>(Yoffset),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[0]));
            });

            break;
        }
        case 2: {
            visit_float(dataTypey, [&](auto as_float) {
                kernel(x,
                       *as_float(alpha),
                       y,
                       *as_float(beta),
                       static_cast<unsigned>(Xoffset),
                       static_cast<unsigned>(Yoffset),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[0]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[1]));
            });

            break;
        }
        case 3: {
            visit_float(dataTypey, [&](auto as_float) {
                kernel(x,
                       *as_float(alpha),
                       y,
                       *as_float(beta),
                       static_cast<unsigned>(Xoffset),
                       static_cast<unsigned>(Yoffset),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[0]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[1]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[2]));
            });

            break;
        }
        case 4: {
            visit_float(dataTypey, [&](auto as_float) {
                kernel(x,
                       *as_float(alpha),
                       y,
                       *as_float(beta),
                       static_cast<unsigned>(Xoffset),
                       static_cast<unsigned>(Yoffset),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[3]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[3]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[0]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[1]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[2]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[3]));
            });

            break;
        }
        case 5: {
            visit_float(dataTypey, [&](auto as_float) {
                kernel(x,
                       *as_float(alpha),
                       y,
                       *as_float(beta),
                       static_cast<unsigned>(Xoffset),
                       static_cast<unsigned>(Yoffset),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[3]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[4]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[3]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[4]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[0]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[1]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[2]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[3]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[4]));
            });

            break;
        }
        default: assert(false);
        }
    }
}

} // namespace miopen
