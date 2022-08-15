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
#include <cassert>
#include <miopen/fusion.hpp>
#include <miopen/logger.hpp>

namespace miopen {
std::ostream& operator<<(std::ostream& stream, const FusionOpDescriptor& x)
{
    MIOPEN_LOG_ENUM(stream,
                    x.kind(),
                    miopenFusionOpConvForward,
                    miopenFusionOpActivForward,
                    miopenFusionOpBatchNormInference,
                    miopenFusionOpBiasForward,
                    miopenFusionOpBatchNormFwdTrain,
                    miopenFusionOpBatchNormBwdTrain,
                    miopenFusionOpActivBackward);
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const MDGraph_op_t& o)
{
    MIOPEN_LOG_ENUM(stream, o, OpEqual, OpNotEqual, OpAny, OpModulo, OpGTE, OpLTE);
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const boost::any& a)
{
    if(a.type() == typeid(std::string))
        stream << boost::any_cast<std::string>(a);
    else if(a.type() == typeid(int))
        stream << boost::any_cast<int>(a);
    else if(a.type() == typeid(miopenConvolutionMode_t))
        stream << boost::any_cast<miopenConvolutionMode_t>(a);
    else if(a.type() == typeid(miopenPaddingMode_t))
        stream << boost::any_cast<miopenPaddingMode_t>(a);
    else if(a.type() == typeid(size_t))
        stream << boost::any_cast<size_t>(a);
    else if(a.type() == typeid(miopenBatchNormMode_t))
        stream << boost::any_cast<miopenBatchNormMode_t>(a);
    else if(a.type() == typeid(miopenActivationMode_t))
        stream << boost::any_cast<miopenActivationMode_t>(a);
    else if(a.type() == typeid(miopenDataType_t))
        stream << boost::any_cast<miopenDataType_t>(a);
    else
        stream << "Unsupported any type: " << a.type().name();
    return stream;
}
} // namespace miopen
