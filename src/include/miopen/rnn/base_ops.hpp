#pragma once

#include <miopen/rnn.hpp>
#include <miopen/gemm_v2.hpp>

namespace miopen {

namespace rnn_base {

template <typename T, size_t N, size_t M>
std::array<T, N + M> Concat(const std::array<T, N>& a, const std::array<T, M>& b)
{
    std::array<T, N + M> result;
    std::copy(a.cbegin(), a.cend(), result.begin());
    std::copy(b.cbegin(), b.cend(), result.begin() + N);
    return result;
}

inline miopen::GemmDescriptor GemmDescriptor64BitWraper(bool isColMajor_,
                                                        bool transA_,
                                                        bool transB_,
                                                        size_t m_,
                                                        size_t n_,
                                                        size_t k_,
                                                        size_t lda_,
                                                        size_t ldb_,
                                                        size_t ldc_,
                                                        size_t batch_count_,
                                                        long long int strideA_,
                                                        long long int strideB_,
                                                        long long int strideC_,
                                                        float alpha_,
                                                        float beta_,
                                                        miopenDataType_t dataType_,
                                                        bool deterministic_)
{
    return GemmDescriptor{isColMajor_,
                          transA_,
                          transB_,
                          static_cast<int>(m_),
                          static_cast<int>(n_),
                          static_cast<int>(k_),
                          static_cast<int>(lda_),
                          static_cast<int>(ldb_),
                          static_cast<int>(ldc_),
                          static_cast<int>(batch_count_),
                          strideA_,
                          strideB_,
                          strideC_,
                          alpha_,
                          beta_,
                          dataType_,
                          deterministic_};
}

class RnnBaseFunctions
{
    RnnBaseFunctions() {}

public:
    static miopenStatus_t BWD_GEMM_Hidden_Prop(Handle& handle,
                                               ConstData_t comb_gates_src_ptr,
                                               size_t comb_gates_src_offset,
                                               ConstData_t filter_src_ptr,
                                               size_t filter_offset,
                                               Data_t ht_dst_ptr,
                                               size_t ht_dst_offset,
                                               size_t gemm_batch_size,
                                               size_t ht_dest_vec_size,
                                               size_t comb_gates_size,
                                               size_t tmp_gate_row_stride,
                                               size_t filter_row_stride,
                                               size_t ht_dest_row_stride,
                                               miopenDataType_t data_type,
                                               bool add_assign = true)
    {
        // no gemm work
        if(gemm_batch_size == 0)
            return miopenStatusSuccess;

        const miopen::GemmDescriptor gemm_desc =
            GemmDescriptor64BitWraper(false,
                                      false,
                                      false,
                                      gemm_batch_size,
                                      ht_dest_vec_size,
                                      comb_gates_size,
                                      tmp_gate_row_stride,
                                      filter_row_stride,
                                      ht_dest_row_stride,
                                      1,                  // batch count
                                      0,                  // Stride A
                                      0,                  // Stride B
                                      0,                  // Stride C
                                      1,                  // alpha
                                      add_assign ? 1 : 0, // beta
                                      data_type,
                                      false);

        return CallGemm(handle,
                        gemm_desc,
                        comb_gates_src_ptr,
                        comb_gates_src_offset,
                        filter_src_ptr,
                        filter_offset,
                        ht_dst_ptr,
                        ht_dst_offset,
                        GemmBackend_t::rocblas);
    }

    static miopenStatus_t BWD_GEMM_Hidden_Prop(Handle& handle,
                                               ConstData_t comb_gates_src_ptr,
                                               size_t comb_gates_src_offset,
                                               const miopen::TensorDescriptor& tmp_gates_src_dsc,

                                               ConstData_t filter_src_ptr,
                                               size_t filter_src_offset,
                                               const miopen::TensorDescriptor& filter_src_dsc,

                                               Data_t ht_dst_ptr,
                                               size_t ht_dst_offset,
                                               const miopen::TensorDescriptor& ht_dest_dsc,
                                               bool add_assign = true)
    {
        assert(filter_src_dsc.GetNumDims() == 2 && tmp_gates_src_dsc.GetNumDims() == 2 &&
               ht_dest_dsc.GetNumDims() == 2);

        const size_t batch_size      = tmp_gates_src_dsc.GetLengths()[0];
        const size_t comb_gates_size = tmp_gates_src_dsc.GetLengths()[1];
        const size_t ht_vec_size     = ht_dest_dsc.GetLengths()[1];

        assert(filter_src_dsc.GetLengths()[0] == comb_gates_size);
        assert(filter_src_dsc.GetLengths()[1] == ht_vec_size);
        assert(ht_dest_dsc.GetLengths()[0] == batch_size);

        const size_t tmp_gates_ld_stride = tmp_gates_src_dsc.GetStrides()[0]; // {batch, comb_gates}
        const size_t filter_ld_stride    = filter_src_dsc.GetStrides()[0]; // {comb_gates, ht_vec}
        const size_t ht_dest_ld_stride   = ht_dest_dsc.GetStrides()[0];    // {batch, ht_vec}

        return BWD_GEMM_Hidden_Prop(handle,
                                    comb_gates_src_ptr,
                                    comb_gates_src_offset,
                                    filter_src_ptr,
                                    filter_src_offset,
                                    ht_dst_ptr,
                                    ht_dst_offset,
                                    batch_size,
                                    ht_vec_size,
                                    comb_gates_size,
                                    tmp_gates_ld_stride,
                                    filter_ld_stride,
                                    ht_dest_ld_stride,
                                    ht_dest_dsc.GetType(),
                                    add_assign);
    }
};

} // namespace rnn_base
} // namespace miopen
