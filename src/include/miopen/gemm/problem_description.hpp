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

#pragma once

#include <miopen/gemm.hpp>
#include <miopen/tensor.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/sqlite_db.hpp>
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace gemm {

struct ProblemDescription : ProblemDescriptionBase
#if MIOPEN_ENABLE_SQLITE
    ,
                            SQLiteSerializable<ProblemDescription>
#endif

{
    ProblemDescription() {}
    ProblemDescription(const GemmDesc& gemmDesc_,
                       const TensorDescriptor& ADesc_,
                       const TensorDescriptor& BDesc_,
                       const TensorDescriptor& CDesc_)
        : gemmDesc(gemmDesc_), ADesc(ADesc_), BDesc(BDesc_), CDesc(CDesc_)
    {
        assert(ADesc.GetLengths().size() == 2);
        assert(BDesc.GetLengths().size() == 2);
        assert(CDesc.GetLengths().size() == 2);
    }

    const GemmDesc& GetGemmDescriptor() const { return gemmDesc; }
    const TensorDescriptor& GetADesc() const { return ADesc; }
    const TensorDescriptor& GetBDesc() const { return BDesc; }
    const TensorDescriptor& GetCDesc() const { return CDesc; }

    int GetK() const
    {
        const auto& lens = ADesc.GetLengths();
        return lens[1]; // A = [M, K]
    }

    int GetM() const
    {
        const auto& lens = ADesc.GetLengths();
        return lens[0]; // A = [M, K]
    }

    int GetN() const
    {
        const auto& lens = BDesc.GetLengths();
        return lens[1]; // B = [K, N]
    }

    miopenDataType_t GetADataType() const { return ADesc.GetType(); }
    miopenDataType_t GetBDataType() const { return BDesc.GetType(); }
    miopenDataType_t GetCDataType() const { return CDesc.GetType(); }

    NetworkConfig MakeNetworkConfig() const;

    void Serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

    template <class Self>
    static void Visit(Self&& self, std::function<void(int, std::string)> f)
    {
        // Once we tune gemm kernels we will not ignore.
        std::ignore = self;
        std::ignore = f;
    }

    template <class Self>
    static void Visit(Self&& self, std::function<void(std::string, std::string)> f)
    {
        // Once we tune gemm kernels we will not ignore.
        std::ignore = self;
        std::ignore = f;
    }

private:
    GemmDesc gemmDesc;
    TensorDescriptor ADesc;
    TensorDescriptor BDesc;
    TensorDescriptor CDesc;
};

} // namespace gemm

} // namespace miopen
