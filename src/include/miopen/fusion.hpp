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
#include <miopen/activ.hpp>
#include <miopen/convolution.hpp>

#include <boost/spirit/home/support/detail/hold_any.hpp>

#include <set>
#include <vector>
#include <unordered_map>

namespace miopen {

typedef enum {
    miopenFusionOpConv      = 0,
    miopenFusionOpActiv     = 1,
    miopenFusionOpBatchNorm = 2,
    miopenFusionOpPool      = 3,
    miopenFusionOpBias      = 4,
} miopenFusionOp_t;

/*
typedef struct miopen_edge
{
    const char* name;
    double alpha;
    bool immutable;
    const miopenTensorDescriptor_t data;
} miopenEdge_t;


typedef struct miopen_op
{
    const char* name;
    int n_inputEdges;
    const miopenEdge_t* inputEdges;
    int n_outputEdges;
    const miopenEdge_t* outputEdges;
    int n_internEdges;
    const miopenEdge_t* internEdges;
} miopenOp_t;
*/

struct OperatorArgs : miopenOperatorArgs
{
    OperatorArgs();
    void append_arg(boost::spirit::hold_any&& v);
    friend std::ostream& operator<<(std::ostream& stream, const OperatorArgs& x);

    private:
    std::vector<boost::spirit::hold_any> args;
};

struct FusionOpDescriptor : miopenFusionOpDescriptor
{
    virtual ~FusionOpDescriptor(){};
    void SetIdx(int _id) { plan_idx = _id; };
    int GetIdx() { return plan_idx; };
    virtual miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc) = 0;
    friend std::ostream& operator<<(std::ostream& stream, const FusionOpDescriptor& x);
    virtual miopenFusionOp_t name() = 0;
    void SetInputDesc(TensorDescriptor i_desc) { input_desc = i_desc; };

    TensorDescriptor input_desc;

    private:
    int plan_idx                       = 0;
    std::shared_ptr<OperatorArgs> args = nullptr;
};

struct ActivFusionOpDescriptor : FusionOpDescriptor
{
    ActivFusionOpDescriptor(ActivationDescriptor& desc) : base_desc(desc){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc);
    miopenFusionOp_t name() { return miopenFusionOpActiv; };

    ActivationDescriptor& base_desc;
};

struct ConvForwardOpDescriptor : FusionOpDescriptor
{
    ConvForwardOpDescriptor(ConvolutionDescriptor& conv_descriptor,
                            TensorDescriptor& filter_descriptor,
                            miopenConvFwdAlgorithm_t fwd_algo)
        : base_desc(conv_descriptor), filter_desc(filter_descriptor), algo(fwd_algo){};
    miopenStatus_t GetOutputDesc(TensorDescriptor& output_desc);
    miopenStatus_t SetArgs(OperatorArgs& args, const void* alpha, const void* beta, const Data_t w);
    miopenFusionOp_t name() { return miopenFusionOpConv; };

    ConvolutionDescriptor& base_desc;
    TensorDescriptor& filter_desc;
    miopenConvFwdAlgorithm_t algo;
};

struct FusionPlanDescriptor : miopenFusionPlanDescriptor
{
    FusionPlanDescriptor(const miopenFusionDirection_t dir, const TensorDescriptor& inDesc)
        : fusion_dir(dir), input_desc(inDesc){};
    ~FusionPlanDescriptor();
    bool isValid();
    miopenStatus_t AddOp(std::shared_ptr<FusionOpDescriptor> desc);
    miopenStatus_t RemoveOp(FusionOpDescriptor& desc);
    TensorDescriptor DeriveOutputDescriptor();
    miopenStatus_t
    GetWorkspaceSizeImmed(Handle& handle, size_t& workSpaceSize, miopenConvFwdAlgorithm_t algo);
    miopenStatus_t Execute();
    friend std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& x);

    private:
    miopenFusionDirection_t fusion_dir;
    TensorDescriptor input_desc;
    TensorDescriptor output_desc;
    int op_count = 0;
    std::unordered_map<int, std::shared_ptr<FusionOpDescriptor>> op_map;
    std::vector<int> ins_order;
};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenFusionOpDescriptor, miopen::FusionOpDescriptor);
MIOPEN_DEFINE_OBJECT(miopenFusionPlanDescriptor, miopen::FusionPlanDescriptor);
MIOPEN_DEFINE_OBJECT(miopenOperatorArgs, miopen::OperatorArgs);

#endif // _MIOPEN_FUSION_HPP_
