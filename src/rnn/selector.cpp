#include <miopen/rnn.hpp>
#include <miopen/rnn_util.hpp>

#include <miopen/activ.hpp>
#include <miopen/env.hpp>
#include <miopen/gemm_v2.hpp>
#include <miopen/logger.hpp>

#include <vector>
#include <numeric>
#include <algorithm>

#include <miopen/rnn/solvers.hpp>

#include <miopen/rnn/tmp_buffer_utils.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_RNNBWDMS_exp)

namespace miopen {

void RNNDescriptor::ModularBackward(Handle& handle,
                                    const SeqTensorDescriptor& yDesc,
                                    ConstData_t dy,
                                    const TensorDescriptor& hDesc,
                                    ConstData_t /*hx*/,
                                    ConstData_t dhy,
                                    Data_t dhx,
                                    const TensorDescriptor& /*cDesc*/,
                                    ConstData_t cx,
                                    ConstData_t dcy,
                                    Data_t dcx,
                                    const SeqTensorDescriptor& xDesc,
                                    Data_t dx,
                                    ConstData_t w,
                                    Data_t workSpace,
                                    size_t /*workSpaceSize*/,
                                    Data_t reserveSpace,
                                    size_t /*reserveSpaceSize*/) const
{

    rnn_base::RNNModularSingleStreamBWD single_stream{*this, xDesc, yDesc, hDesc};

    single_stream.ComputeBWD(handle, dy, dhy, dhx, cx, dcy, dcx, dx, w, workSpace, reserveSpace);
}

} // namespace miopen
