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
#ifndef MIOPEN_FUSION_HPP_
#define MIOPEN_FUSION_HPP_

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <set>
#include <vector>

namespace miopen {

/*! @struct miopenEdge_t
* @brief Defines an operation edge
*/
typedef struct miopen_edge
{
    const char* name;                    /*!< edge name */
    double alpha;                        /*!< scale */
    bool immutable;                      /*!< immutable data */
    const miopenTensorDescriptor_t data; /*!< data */
} miopenEdge_t;

/*! @struct miopenOp_t
* @brief common part of op definition
*/
typedef struct miopen_op
{
    const char* name;                /*!< opration instance name if applicable */
    int n_inputEdges;                /*!< number of input edges */
    const miopenEdge_t* inputEdges;  /*!< input edges definitions */
    int n_outputEdges;               /*!< number of output edges */
    const miopenEdge_t* outputEdges; /*!< output edges definitions */
    int n_internEdges;               /*!< number of internal edges */
    const miopenEdge_t* internEdges; /*!< internal edges definitions (weights) */
} miopenOp_t;

struct OperatorArgs : miopenOperatorArgs
{
    OperatorArgs();

    friend std::ostream& operator<<(std::ostream& stream, const OperatorArgs& x);

    private:
    std::vector<double> parms;
};

struct OperatorDescriptor : miopenOperatorDescriptor
{
    OperatorDescriptor();

    friend std::ostream& operator<<(std::ostream& stream, const OperatorDescriptor& x);

    private:
    std::vector<double> parms;
};

struct FusionPlanDescriptor : miopenFusionPlanDescriptor
{
    FusionPlanDescriptor();

    friend std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& x);

    private:
    std::vector<double> parms;
};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenOperatorDescriptor, miopen::OperatorDescriptor);
MIOPEN_DEFINE_OBJECT(miopenFusionPlanDescriptor, miopen::FusionPlanDescriptor);
MIOPEN_DEFINE_OBJECT(miopenOperatorArgs, miopen::OperatorArgs);

#endif // _MIOPEN_FUSION_HPP_
