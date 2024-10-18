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
#ifndef GUARD_MIOPEN_BATCHNORMALIZATION_HPP_
#define GUARD_MIOPEN_BATCHNORMALIZATION_HPP_

#include <miopen/common.hpp>
#include <miopen/miopen.h>

#include <string>
#include <vector>

#define MIO_BN_CPP_PROF 0
#define MIOPEN_BN_CPP_DEBUG 0
#define MIO_BN_STATIC_WGSIZE 256
#define MIO_BN_TIME_EVERYTHING 0

namespace miopen {

struct Handle;
struct TensorDescriptor;

MIOPEN_INTERNALS_EXPORT void DeriveBNTensorDescriptor(TensorDescriptor& derivedBnDesc,
                                                      const TensorDescriptor& xDesc,
                                                      miopenBatchNormMode_t bn_mode);

MIOPEN_INTERNALS_EXPORT TensorDescriptor
BuildReshaped4DTensorDescriptor(const miopen::TensorDescriptor& tDesc);

void bnBwdTrainSelectSingle(const Handle& handle,
                            miopenDataType_t dtype,
                            const std::string& program_name,
                            const std::string& algo_name,
                            const std::string& kernel_name,
                            const std::string& network_config,
                            const std::string& parms,
                            const std::vector<size_t>& vld,
                            const std::vector<size_t>& vgd,
                            ConstData_t x,
                            ConstData_t dy,
                            Data_t dx,
                            ConstData_t bnScale,
                            Data_t dScale,
                            Data_t dBias,
                            bool useSaved,
                            double epsilon,
                            ConstData_t savedMean,
                            ConstData_t savedInvVariance,
                            float inhw);

void bnFwdTrainSelectSingleFull(const Handle& handle,
                                int variant,
                                miopenDataType_t dtype,
                                const std::string& algo_name,
                                const std::string& network_config,
                                ConstData_t x,
                                Data_t y,
                                ConstData_t bnScale,
                                ConstData_t bnBias,
                                bool resultsave,
                                bool resultrunning,
                                double expAvgFactor,
                                Data_t resultRunningMean,
                                Data_t resultRunningVariance,
                                double epsilon,
                                Data_t resultSaveMean,
                                Data_t resultSaveInvVariance,
                                float inhw,
                                unsigned int in_cstride,
                                unsigned int in_nstride);

void bnBwdTrainSelectMulti(const Handle& handle,
                           miopenDataType_t dtype,
                           const std::string& program_name,
                           const std::string& algo_name,
                           const std::string& kernel_name,
                           const std::string& network_config,
                           const std::string& parms,
                           const std::vector<size_t>& vld,
                           const std::vector<size_t>& vgd,
                           ConstData_t x,
                           ConstData_t dy,
                           Data_t dx,
                           ConstData_t bnScale,
                           Data_t dScale,
                           Data_t dBias,
                           bool useSaved,
                           double epsilon,
                           ConstData_t savedMean,
                           ConstData_t savedInvVariance,
                           float inhw);

void bnFwdTrainSelectSingleEmpty(const Handle& handle,
                                 int variant,
                                 miopenDataType_t dtype,
                                 const std::string& program_name,
                                 const std::string& algo_name,
                                 const std::string& kernel_name,
                                 const std::string& network_config,
                                 const std::string& parms,
                                 const std::vector<size_t>& vld,
                                 const std::vector<size_t>& vgd,
                                 ConstData_t x,
                                 Data_t y,
                                 ConstData_t bnScale,
                                 ConstData_t bnBias,
                                 bool resultsave,
                                 bool resultrunning,
                                 double expAvgFactor,
                                 Data_t resultRunningMean,
                                 Data_t resultRunningVariance,
                                 double epsilon,
                                 Data_t resultSaveMean,
                                 Data_t resultSaveInvVariance,
                                 float inhw,
                                 unsigned int in_cstride,
                                 unsigned int in_nstride);

void bnFwdTrainSelectMulti(const Handle& handle,
                           miopenDataType_t dtype,
                           const std::string& program_name,
                           const std::string& algo_name,
                           const std::string& kernel_name,
                           const std::string& network_config,
                           const std::string& parms,
                           const std::vector<size_t>& vld,
                           const std::vector<size_t>& vgd,
                           ConstData_t x,
                           Data_t y,
                           ConstData_t bnScale,
                           ConstData_t bnBias,
                           bool resultsave,
                           bool resultrunning,
                           double expAvgFactor,
                           Data_t resultRunningMean,
                           Data_t resultRunningVariance,
                           double epsilon,
                           Data_t resultSaveMean,
                           Data_t resultSaveInvVariance,
                           float inhw);

void profileSequence(const Handle& handle, unsigned char select, float* ctime);

MIOPEN_INTERNALS_EXPORT void BatchNormForwardInference(Handle& handle,
                                                       miopenBatchNormMode_t bn_mode,
                                                       const void* alpha,
                                                       const void* beta,
                                                       const TensorDescriptor& xDesc,
                                                       ConstData_t x,
                                                       const TensorDescriptor& yDesc,
                                                       Data_t y,
                                                       const TensorDescriptor& scaleDesc,
                                                       const TensorDescriptor& BiasDesc,
                                                       const TensorDescriptor& estMeanDesc,
                                                       const TensorDescriptor& estVarianceDesc,
                                                       ConstData_t bnScale,
                                                       ConstData_t bnBias,
                                                       ConstData_t estimatedMean,
                                                       ConstData_t estimatedVariance,
                                                       double epsilon);

MIOPEN_INTERNALS_EXPORT void BatchNormForwardTraining(Handle& handle,
                                                      miopenBatchNormMode_t bn_mode,
                                                      const void* alpha,
                                                      const void* beta,
                                                      const TensorDescriptor& xDesc,
                                                      ConstData_t x,
                                                      const TensorDescriptor& yDesc,
                                                      Data_t y,
                                                      const TensorDescriptor& scaleDesc,
                                                      const TensorDescriptor& biasDesc,
                                                      const TensorDescriptor& savedMeanDesc,
                                                      const TensorDescriptor& savedVarianceDesc,
                                                      ConstData_t bnScale,
                                                      ConstData_t bnBias,
                                                      double expAvgFactor,
                                                      Data_t resultRunningMean,
                                                      Data_t resultRunningVariance,
                                                      double epsilon,
                                                      Data_t resultSaveMean,
                                                      Data_t resultSaveInvVariance);

MIOPEN_INTERNALS_EXPORT void BatchNormBackward(Handle& handle,
                                               miopenBatchNormMode_t bn_mode,
                                               const void* alphaDataDiff,
                                               const void* betaDataDiff,
                                               const void* alphaParamDiff,
                                               const void* betaParamDiff,
                                               const TensorDescriptor& xDesc,
                                               ConstData_t x,
                                               const TensorDescriptor& dyDesc,
                                               ConstData_t dy,
                                               const TensorDescriptor& dxDesc,
                                               Data_t dx,
                                               const TensorDescriptor& scaleDesc,
                                               const TensorDescriptor& BiasDesc,
                                               const TensorDescriptor& savedMeanDesc,
                                               const TensorDescriptor& savedVarianceDesc,
                                               ConstData_t bnScale,
                                               Data_t resultBnScaleDiff,
                                               Data_t resultBnBiasDiff,
                                               double epsilon,
                                               ConstData_t savedMean,
                                               ConstData_t savedInvVariance);

} // namespace miopen

#endif // GUARD_MIOPEN_BATCHNORMALIZATION_HPP_
