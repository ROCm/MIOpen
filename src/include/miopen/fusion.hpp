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

typedef struct
{
    miopenActivationMode_t mode;
    void* alpha;
    void* beta;
    double activAlpha;
    double activBeta;
    double activGamma;
} miopenActivationOpParams_t;

typedef struct
{
    miopenBatchNormMode_t bn_mode;
    void* alpha;
    void* beta;
    miopenTensorDescriptor_t bnScaleBiasMeanVarDesc;
    void* bnScale;
    void* bnBias;
    void* estimatedMean;
    void* estimatedVariance;
    double epsilon
} miopenBatchNormOpParams_t;

typedef struct
{
    miopenConvFwdAlgorithm_t algo;
    miopenTensorDescriptor_t wDesc;
    void* w;
} miopenConvolutionOpParams_t;

typedef struct
{
    void* alpha;
    void* beta;
    int windowHeight;
    int windowWidth;
    int pad_h;
    int pad_w;
    int vstride;
    int hstride;
} miopenPoolingOpParams_t;

// Valid fused operators
std::set<std::vector<miopenOperator_t>> validFusions = {
    {miopenBatchNormOp, miopenActivationOp},
};

struct OperatorDescriptor : miopenOperatorDescriptor
{
    OperatorDescriptor();
    OperatorDescriptor(miopenActivationMode_t m, const double* pparms);
    OperatorDescriptor(miopenActivationMode_t m, double alpha, double beta, double gamma);

    miopenStatus_t ForwardInference(Handle& handle);

    friend std::ostream& operator<<(std::ostream& stream, const OperatorDescriptor& x);

    private:
    std::vector<double> parms;
};

struct OperatorDescriptor : miopenOperatorDescriptor
{
    OperatorDescriptor();
    OperatorDescriptor(miopenActivationMode_t m, const double* pparms);
    OperatorDescriptor(miopenActivationMode_t m, double alpha, double beta, double gamma);

    miopenStatus_t ForwardInference(Handle& handle);

    friend std::ostream& operator<<(std::ostream& stream, const OperatorDescriptor& x);

    private:
    std::vector<double> parms;
};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenActivationDescriptor, miopen::ActivationDescriptor);
#endif // _MIOPEN_ACTIV_HPP_

#endif // _MIOPEN_FUSION_HPP_