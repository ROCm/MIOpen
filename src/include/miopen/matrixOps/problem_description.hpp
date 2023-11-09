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

#pragma once

#include <miopen/problem_description_base.hpp>
#include <miopen/sqlite_db.hpp>
#include <miopen/matrixOps.hpp>
#include <miopen/tensor.hpp>
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace gemm {

enum class Direction
{
    ForwardTraining,
    ForwardInference,
    Backward,
};

struct ProblemDescription : ProblemDescriptionBase
#if MIOPEN_ENABLE_SQLITE
    ,
                            SQLiteSerializable<ProblemDescription>
#endif

{
    // C = AxB
    ProblemDescription(const matrixOps::GemmDescriptor& gemm_desc_,
                       const TensorDescriptor& ADesc_,
                       const TensorDescriptor& BDesc_,
                       const TensorDescriptor& CDesc_)
        : gemm_desc(gemm_desc_), ADesc(ADesc_), BDesc(BDesc_), CDesc(CDesc_)
    {
        assert(ADesc.GetLengths().size() == 2);
        assert(BDesc.GetLengths().size() == 2);
        assert(CDesc.GetLengths().size() == 2);
    }

    const matrixOps::GemmDescriptor& GetGemmDescriptor() const { return gemm_desc; }

    const TensorDescriptor& GetADesc() const { return ADesc; }
    const TensorDescriptor& GetBDesc() const { return BDesc; }
    const TensorDescriptor& GetCDesc() const { return CDesc; }

    int GetM() const { return gemm_desc.GetM(); }

    int GetN() const { return gemm_desc.GetN(); }

    int GetK() const { return gemm_desc.GetK(); }

    miopenDataType_t GetADataType() const { return ADesc.GetType(); }
    miopenDataType_t GetBDataType() const { return BDesc.GetType(); }
    miopenDataType_t GetCDataType() const { return CDesc.GetType(); }

    NetworkConfig MakeNetworkConfig() const override;

    void Serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

    template <class Self>
    static void Visit([[maybe_unused]] Self&& self,
                      [[maybe_unused]] std::function<void(int, std::string)> f)
    {
        // Once we tune gemm kernels we will not ignore.
        MIOPEN_THROW(miopenStatusNotImplemented);
    }

    template <class Self>
    static void Visit([[maybe_unused]] Self&& self,
                      [[maybe_unused]] std::function<void(std::string, std::string)> f)
    {
        // Once we tune gemm kernels we will not ignore.
        MIOPEN_THROW(miopenStatusNotImplemented);
    }

private:
    matrixOps::GemmDescriptor gemm_desc;
    TensorDescriptor ADesc;
    TensorDescriptor BDesc;
    TensorDescriptor CDesc;
};

} // namespace gemm

namespace matrixAdd {

enum class Direction
{
    ForwardTraining,
    ForwardInference,
    Backward,
};

struct ProblemDescription : ProblemDescriptionBase
#if MIOPEN_ENABLE_SQLITE
    ,
                            SQLiteSerializable<ProblemDescription>
#endif

{
    // E = C + D
    ProblemDescription(const matrixOps::MatrixAdditionDescriptor& MatAddDescriptor_,
                       const TensorDescriptor& CDesc_,
                       const TensorDescriptor& DDesc_,
                       const TensorDescriptor& EDesc_)
        : matrix_add_desc(MatAddDescriptor_), CDesc(CDesc_), DDesc(DDesc_), EDesc(EDesc_)
    {
        assert(CDesc.GetLengths().size() == 2);
        assert(DDesc.GetLengths().size() == 2);
        assert(EDesc.GetLengths().size() == 2);
    }

    const matrixOps::MatrixAdditionDescriptor& GetMatrixAddDescriptor() const
    {
        return matrix_add_desc;
    }

    const TensorDescriptor& GetCDesc() const { return CDesc; }
    const TensorDescriptor& GetDDesc() const { return DDesc; }
    const TensorDescriptor& GetEDesc() const { return EDesc; }

    miopenDataType_t GetCDataType() const { return CDesc.GetType(); }
    miopenDataType_t GetDDataType() const { return DDesc.GetType(); }
    miopenDataType_t GetEDataType() const { return EDesc.GetType(); }

    NetworkConfig MakeNetworkConfig() const override;

    void Serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& os, const ProblemDescription& obj)
    {
        obj.Serialize(os);
        return os;
    }

    template <class Self>
    static void Visit([[maybe_unused]] Self&& self,
                      [[maybe_unused]] std::function<void(int, std::string)> f)
    {
        // Once we tune gemm kernels we will not ignore.
        MIOPEN_THROW(miopenStatusNotImplemented);
    }

    template <class Self>
    static void Visit([[maybe_unused]] Self&& self,
                      [[maybe_unused]] std::function<void(std::string, std::string)> f)
    {
        // Once we tune gemm kernels we will not ignore.
        MIOPEN_THROW(miopenStatusNotImplemented);
    }

private:
    matrixOps::MatrixAdditionDescriptor matrix_add_desc;
    TensorDescriptor CDesc;
    TensorDescriptor DDesc;
    TensorDescriptor EDesc;
};

} // namespace matrixAdd

using GemmAddProblemDescription =
    std::tuple<miopen::gemm::ProblemDescription, miopen::matrixAdd::ProblemDescription>;
} // namespace miopen
