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

#ifndef MIOPEN_GUARD_MLOPEN_ANY_SOLVER_HPP
#define MIOPEN_GUARD_MLOPEN_ANY_SOLVER_HPP

#include <miopen/conv_solution.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/mlo_internal.hpp>

#include <cassert>
#include <memory>
#include <typeinfo>

namespace miopen {
namespace solver {

struct AnySolver
{
    using Db = decltype(std::declval<mlo_construct_base>().GetDb());

    AnySolver() : ptr_value(nullptr){};
    template <class U>
    AnySolver(U src) : ptr_value(new AnySolver_tmpl<U>(std::forward<U>(src))){};
    bool IsApplicable(const ConvolutionContext& ctx) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->IsApplicable(ctx);
    };
    bool IsTunable() const
    {
        assert(ptr_value != nullptr);
        return ptr_value->IsTunable();
    };
    bool IsDynamic() const
    {
        assert(ptr_value != nullptr);
        return ptr_value->IsDynamic();
    };
    float GetWti(const ConvolutionContext& ctx) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->GetWti(ctx);
    };
    const std::type_info& Type() const
    {
        assert(ptr_value != nullptr);
        return ptr_value->Type();
    };
    bool IsEmpty() const { return ptr_value == nullptr; };
    ConvSolution FindSolution(const ConvolutionContext& ctx,
                              Db& db,
                              const miopen::AnyInvokeParams& invoke_ctx) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->FindSolution(ctx, db, invoke_ctx);
    };
    std::string GetSolverDbId() const
    {
        assert(ptr_value != nullptr);
        return ptr_value->GetSolverDbId();
    }

    size_t GetWorkspaceSize(const ConvolutionContext& ctx) const
    {
        assert(ptr_value != nullptr);
        return ptr_value->GetWorkspaceSize(ctx);
    }

    // virtual base class
    struct AnySolver_base
    {
        using ptr = std::shared_ptr<const AnySolver_base>;

        virtual ~AnySolver_base(){};
        virtual bool IsApplicable(const ConvolutionContext& ctx) const                     = 0;
        virtual bool IsTunable() const                                                     = 0;
        virtual bool IsDynamic() const                                                     = 0;
        virtual float GetWti(const ConvolutionContext& ctx) const                          = 0;
        virtual const std::type_info& Type() const                                         = 0;
        virtual std::string GetSolverDbId() const                                          = 0;
        virtual ConvSolution FindSolution(const ConvolutionContext& ctx,
                                          Db& db,
                                          const miopen::AnyInvokeParams& invoke_ctx) const = 0;
        virtual size_t GetWorkspaceSize(const ConvolutionContext& ctx) const               = 0;
    };

    // templated derived class
    template <class T>
    struct AnySolver_tmpl : AnySolver_base
    {
        struct TunableSolver
        {
            template<typename U> static constexpr auto Test(U*)
            ->typename
                std::is_same<
                    decltype(std::declval<U>().GetSolution(std::declval<const ConvolutionContext&>(),
                        std::declval<const decltype(std::declval<U>().GetPerformanceConfig(
                            std::declval<const ConvolutionContext&>()))&>(),
                        std::declval<const bool>())),
                    ConvSolution
                >::type;

            template<typename U> static constexpr std::false_type Test(...);

            using type = decltype(Test<T>(nullptr));
            static constexpr bool Is = type::value;
        };

        AnySolver_tmpl(T obj) : value(std::move(obj)){};
        bool IsApplicable(const ConvolutionContext& ctx) const override
        {
            return value.IsApplicable(ctx);
        }
        bool IsTunable() const override
        {
            return TunableSolver::Is;
        }
        bool IsDynamic() const override { return value.IsDynamic(); }
        float GetWti(const ConvolutionContext& ctx) const override { return value.GetWti(ctx); }
        ConvSolution FindSolution(const ConvolutionContext& ctx,
                                  Db& db,
                                  const miopen::AnyInvokeParams& invoke_ctx) const override
        {
            return miopen::solver::FindSolution(value, ctx, db, invoke_ctx);
        };
        size_t GetWorkspaceSize(const ConvolutionContext& ctx) const override
        {
            return value.GetWorkspaceSize(ctx);
        }
        const std::type_info& Type() const override { return typeid(T); };
        std::string GetSolverDbId() const override { return ComputeSolverDbId(value); }

        private:
        T value;
    };

    AnySolver_base::ptr ptr_value;
};

} // namespace solver
} // namespace miopen

#endif
