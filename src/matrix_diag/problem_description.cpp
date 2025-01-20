/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/matrix_diag/problem_description.hpp>
#include <miopen/names.hpp>
#include <miopen/errors.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>

#include <sstream>

namespace miopen {

namespace matrix_diag {

NetworkConfig MatrixSetDiagForwardProblemDescription::MakeNetworkConfig() const
{
    auto dtype      = diagDesc.GetType();
    auto outputSize = outputDesc.GetElementSize();

    std::ostringstream ss;

    ss << "matrix_set_diag_fwd" << IsAllContiguous();
    ss << " dtype" << dtype;
    ss << " outputSize" << outputSize;
    ss << " align" << align;

    return NetworkConfig{ss.str()};
}

NetworkConfig MatrixSetDiagBackwardProblemDescription::MakeNetworkConfig() const
{
    auto dtype     = outputGradDesc.GetType();
    auto inputSize = inputGradDesc.GetElementSize();
    auto diagSize  = diagGradDesc.GetElementSize();

    std::ostringstream ss;

    ss << "matrix_set_diag_bwd" << IsAllContiguous();
    ss << " dtype" << dtype;
    ss << " inputSize" << inputSize;
    ss << " diagSize" << diagSize;
    ss << " align" << align;

    return NetworkConfig{ss.str()};
}

NetworkConfig MatrixDiagPartForwardProblemDescription::MakeNetworkConfig() const
{
    auto dtype      = padDesc.GetType();
    auto outputSize = outputDesc.GetElementSize();

    std::ostringstream ss;

    ss << "matrix_diag_part_fwd" << IsAllContiguous();
    ss << " dtype" << dtype;
    ss << " outputSize" << outputSize;
    ss << " align" << align;

    return NetworkConfig{ss.str()};
}

bool IsValidMatrixDiag(const TensorDescriptor& padDesc,
                       const TensorDescriptor& diagDesc,
                       const TensorDescriptor& outDesc,
                       const int64_t diagOffset0,
                       const int64_t diagOffset1,
                       const std::string& refer_name,
                       const std::string& pad_alias_name,
                       const std::string& diag_alias_name,
                       const std::string& out_alias_name,
                       const bool hasPad,
                       const bool hasDiag)
{
    // Valid type
    if(hasPad && hasDiag && diagDesc.GetType() != padDesc.GetType())
        MIOPEN_THROW(miopenStatusBadParm,
                     (std::stringstream() << refer_name << ": " << diag_alias_name << " and "
                                          << pad_alias_name << " datatype do not match.")
                         .str());
    if(hasDiag && diagDesc.GetType() != outDesc.GetType())
        MIOPEN_THROW(miopenStatusBadParm,
                     (std::stringstream() << refer_name << ": " << diag_alias_name << " and "
                                          << out_alias_name << " datatype do not match.")
                         .str());
    if(hasPad && padDesc.GetType() != outDesc.GetType())
        MIOPEN_THROW(miopenStatusBadParm,
                     (std::stringstream() << refer_name << ": " << pad_alias_name << " and "
                                          << out_alias_name << " datatype do not match.")
                         .str());

    // Valid size
    const int64_t M = static_cast<int64_t>(outDesc.GetLengths()[outDesc.GetNumDims() - 2]);
    const int64_t N = static_cast<int64_t>(outDesc.GetLengths()[outDesc.GetNumDims() - 1]);
    const int64_t max_diag_len =
        std::min(M + std::min(diagOffset1, 0L), N + std::min(-diagOffset0, 0L));
    if(outDesc.GetNumDims() < 2)
        MIOPEN_THROW(miopenStatusBadParm,
                     (std::stringstream() << refer_name << ": " << out_alias_name
                                          << " tensor must have at least 2 dimensions")
                         .str());
    if(-diagOffset0 >= M)
        MIOPEN_THROW(miopenStatusBadParm,
                     (std::stringstream()
                      << refer_name << ": diagOffset0 must be less than the second last "
                      << out_alias_name << " dimension")
                         .str());
    if(diagOffset1 >= N)
        MIOPEN_THROW(miopenStatusBadParm,
                     (std::stringstream()
                      << refer_name << ": diagOffset1 must be less than the last " << out_alias_name
                      << " dimension")
                         .str());
    if(hasDiag)
    {
        if(diagOffset0 == diagOffset1)
        {
            if(diagDesc.GetNumDims() < 1)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             (std::stringstream()
                              << refer_name << ": When diagOffset0 == diagOffset1, "
                              << diag_alias_name
                              << " tensor must "
                                 "have at least 1 dimension")
                                 .str());
            }
            if(diagDesc.GetNumDims() + 1 != outDesc.GetNumDims())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             (std::stringstream()
                              << refer_name << ": When diagOffset0 == diagOffset1, "
                              << out_alias_name
                              << " tensor must have "
                                 "1 more dimension than "
                              << diag_alias_name << " tensor")
                                 .str());
            }
            if(std::vector<size_t>(diagDesc.GetLengths().begin(),
                                   diagDesc.GetLengths().end() - 1) !=
               std::vector<size_t>(outDesc.GetLengths().begin(), outDesc.GetLengths().end() - 2))
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             (std::stringstream()
                              << refer_name << ": When diagOffset0 == diagOffset1, "
                              << diag_alias_name
                              << " tensor has "
                                 "shape [I, J, ..., L, M, N] then "
                              << out_alias_name
                              << " tensor must have shape [I, J, "
                                 "..., L, M, num_rows, num_cols]")
                                 .str());
            }
            if(diagDesc.GetLengths().back() != max_diag_len)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             (std::stringstream()
                              << refer_name << ": When diagOffset0 == diagOffset1, "
                              << diag_alias_name
                              << " tensor last "
                                 "dimension must match with "
                              << out_alias_name << " tensor diagonal length")
                                 .str());
            }
        }
        else
        {
            if(diagOffset0 > diagOffset1)
                MIOPEN_THROW(miopenStatusBadParm,
                             (std::stringstream()
                              << refer_name
                              << ": diagOffset0 must be less than or equal to diagOffset1")
                                 .str());
            if(diagDesc.GetNumDims() < 2)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             (std::stringstream()
                              << refer_name << ": When diagOffset0 == diagOffset1, "
                              << diag_alias_name
                              << " tensor must "
                                 "have at least 2 dimensions")
                                 .str());
            }
            if(diagDesc.GetNumDims() != outDesc.GetNumDims())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             (std::stringstream()
                              << refer_name << ": When diagOffset0 != diagOffset1, "
                              << out_alias_name
                              << " tensor must has "
                                 "the same number of dimension with "
                              << diag_alias_name << " tensor")
                                 .str());
            }
            if(std::vector<size_t>(diagDesc.GetLengths().begin(),
                                   diagDesc.GetLengths().end() - 2) !=
               std::vector<size_t>(outDesc.GetLengths().begin(), outDesc.GetLengths().end() - 2))
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             (std::stringstream()
                              << refer_name << ": When diagOffset0 != diagOffset1, "
                              << diag_alias_name
                              << " tensor has "
                                 "shape [I, J, ..., L, M, N] then "
                              << out_alias_name
                              << " tensor must have shape [I, J, "
                                 "..., L, num_rows, num_cols]")
                                 .str());
            }
            if(diagDesc.GetLengths()[diagDesc.GetNumDims() - 2] != diagOffset1 - diagOffset0 + 1)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             (std::stringstream()
                              << refer_name << ": When diagOffset0 != diagOffset1, "
                              << diag_alias_name
                              << " tensor second last "
                                 "dimension must equal to diagOffset1 - diagOffset0")
                                 .str());
            }
            if(diagDesc.GetLengths().back() != max_diag_len)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             (std::stringstream()
                              << refer_name << ": When diagOffset0 != diagOffset1, "
                              << diag_alias_name
                              << " tensor last "
                                 "dimension must match with "
                              << out_alias_name << " tensor diagonal length")
                                 .str());
            }
        }
    }
    if(hasPad && padDesc.GetElementSize() > 1 && padDesc.GetLengths() != outDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     (std::stringstream()
                      << refer_name << ": When there are more than 1 element in " << pad_alias_name
                      << " tensor, " << pad_alias_name << " and " << out_alias_name
                      << " tensor size must be same.")
                         .str());
    }

    return true;
}

} // namespace matrix_diag

namespace solver::matrix_diag {
KernelInfo make_hip_kernel(std::vector<size_t> localsize,
                           std::vector<size_t> gridsize,
                           std::string kernel_file,
                           std::string kernel_name,
                           KernelBuildParameters build_params)
{
    while(localsize.size() < 3)
        localsize.push_back(1);
    while(gridsize.size() < 3)
        gridsize.push_back(1);
    for(int i = 0; i < localsize.size(); ++i)
        gridsize[i] = AlignUp(gridsize[i], localsize[i]);
    return KernelInfo{
        build_params.GenerateFor(kbp::HIP{}), localsize, gridsize, kernel_file, kernel_name};
}
} // namespace solver::matrix_diag

} // namespace miopen
