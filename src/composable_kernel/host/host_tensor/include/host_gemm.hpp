#pragma once
#include "host_tensor.hpp"
#include "gemm_common.hpp"

template <typename AType, typename BType, typename CType>
void host_gemm(const Tensor<AType>& a,
               const Tensor<BType>& b,
               Tensor<CType>& c,
               const GemmMatrixLayout layout)
{
    if(layout == GemmMatrixLayout::MK_KN_MN)
    {
        auto f_mk_kn_mn = [&](auto m, auto n) {
            const int K = a.mDesc.GetLengths()[1];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += static_cast<const double>(a(m, k)) * static_cast<const double>(b(k, n));
            }

            c(m, n) = v;
        };

        make_ParallelTensorFunctor(f_mk_kn_mn, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::MK_NK_MN)
    {
        auto f_mk_nk_mn = [&](auto m, auto n) {
            const int K = a.mDesc.GetLengths()[1];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += static_cast<const double>(a(m, k)) * static_cast<const double>(b(n, k));
            }

            c(m, n) = v;
        };

        make_ParallelTensorFunctor(f_mk_nk_mn, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::KM_KN_MN)
    {
        auto f_km_kn_mn = [&](auto m, auto n) {
            const int K = a.mDesc.GetLengths()[0];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += static_cast<const double>(a(k, m)) * static_cast<const double>(b(k, n));
            }

            c(m, n) = v;
        };

        make_ParallelTensorFunctor(f_km_kn_mn, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::KM_NK_MN)
    {
        auto f_km_nk_mn = [&](auto m, auto n) {
            const int K = a.mDesc.GetLengths()[0];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += static_cast<const double>(a(k, m)) * static_cast<const double>(b(n, k));
            }

            c(m, n) = v;
        };

        make_ParallelTensorFunctor(f_km_nk_mn, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::MK_KN_NM)
    {
        auto f_mk_kn_nm = [&](auto n, auto m) {
            const int K = a.mDesc.GetLengths()[1];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += static_cast<const double>(a(m, k)) * static_cast<const double>(b(k, n));
            }

            c(n, m) = v;
        };

        make_ParallelTensorFunctor(f_mk_kn_nm, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::MK_NK_NM)
    {
        auto f_mk_nk_nm = [&](auto n, auto m) {
            const int K = a.mDesc.GetLengths()[1];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += static_cast<const double>(a(m, k)) * static_cast<const double>(b(n, k));
            }

            c(n, m) = v;
        };

        make_ParallelTensorFunctor(f_mk_nk_nm, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::KM_KN_NM)
    {
        auto f_km_kn_nm = [&](auto n, auto m) {
            const int K = a.mDesc.GetLengths()[0];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += static_cast<const double>(a(k, m)) * static_cast<const double>(b(k, n));
            }

            c(n, m) = v;
        };

        make_ParallelTensorFunctor(f_km_kn_nm, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::KM_NK_NM)
    {
        auto f_km_nk_nm = [&](auto n, auto m) {
            const int K = a.mDesc.GetLengths()[0];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += static_cast<const double>(a(k, m)) * static_cast<const double>(b(n, k));
            }

            c(n, m) = v;
        };

        make_ParallelTensorFunctor(f_km_nk_nm, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else
    {
        throw std::runtime_error("wrong! not supported layout");
    }
}
