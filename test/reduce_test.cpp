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
#include "driver.hpp"
#include "test.hpp"
#include "verify.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/reducetensor.hpp>
#include <miopen/float_equal.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <limits>
#include <iostream>

#include <half.hpp>

using float16 = half_float::half;

static void get_all_indexes(const std::vector<std::size_t> & dimLengths, int dim, std::vector< std::vector<std::size_t> > & indexes)
{
   if ( dim < dimLengths.size() ) {
        std::vector< std::vector<std::size_t> > updated_indexes;

        if ( dim == 0 ) {
             assert( indxes.size() == 0 );
             assert( dimLengths[dim] > 0 ); 
             for (std::size_t i=0; i < dimLengths[dim]; i++) {
                  std::vector<std::size_t> index = { i };

                  updated_indexes.push_back(index);
             };
        }
        else {
             // go through all the current indexes
             for (std::size_t k=0; k < indexes.size(); k++)
                  for (std::size_t i=0; i < dimLengths[dim]; i++) {
                       std::vector<std::size_t> index = indexes[k];

                       index.push_back(i);

                       updated_indexes.push_back(index);
                  };
        };

       // update to the indexes (output)
       indexes = updated_indexes;

       // further to construct the indexes from the updated status
       get_all_indexes(dimLengths, dim+1, indexes);
   };
};

static std::size_t get_offset_from_index(const std::vector<std::size_t> & strides, const std::vector<std::size_t> & index)
{
   std::size_t offset = 0;

   assert( strides.size() == index.size() ); 

   for (int i=0; i < index.size(); i++)
        offset += strides[i] * index[i];

   return(offset);
};

static std::size_t get_flatten_offset(const std::vector<std::size_t> & lengths, const std::vector<std::size_t> & index)
{
   std::size_t offset = 0;

   assert( lengths.size() == index.size() );

   for (int i=lengths.size()-1; i > 0; i--)
        offset += index[i] * lengths[i-1];

   offset += index[0];

   return(offset);
};

template <typename compType>
static std::function<compType(compType, compType)>  ReduceOpFn(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
      case MIOPEN_REDUCE_TENSOR_ADD:
           return( [&](compType a_, compType b_) { return a_ + b_; } );

      case MIOPEN_REDUCE_TENSOR_MUL:
           return( [&](compType a_, compType b_) { return a_ * b_; } );

      case MIOPEN_REDUCE_TENSOR_MIN:
           return( [&](compType a_, compType b_) { return (a_> b_)? b_ : a_; } );

      case MIOPEN_REDUCE_TENSOR_MAX:
           return( [&](compType a_, compType b_) { return (a_> b_)? a_ : b_; } );
    }
};

template <typename compType>
static compType  ReduceOpZeroVal(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
      case MIOPEN_REDUCE_TENSOR_ADD:
           return( static_cast<compType>(0.0) );

      case MIOPEN_REDUCE_TENSOR_MUL:
           return( static_cast<compType>(1.0) );

      case MIOPEN_REDUCE_TENSOR_MIN:
           return( std::numeric_limits<compType>::max() );

      case MIOPEN_REDUCE_TENSOR_MAX:
           return( std::numeric_limits<compType>::min() );
    }
};

#define binop_with_nan_check(nanOpt,opReduce,accuVal,currVal)  \
     {                                                         \
        if ( nanOpt == MIOPEN_NOT_PROPAGATE_NAN )              \
             accuVal = opReduce(accuVal,currVal);              \
        else {                                                 \
             if ( ::isnan(currVal) )                           \
                accuVal = currVal;                             \
             else                                              \
                accuVal = opReduce(accuVal,currVal);           \
        };                                                     \
     }

#define binop_with_nan_check2(nanOpt,opReduce,accuVal,currVal,accuIndex,currIndex)  \
     {                                                           \
        if ( nanOpt == MIOPEN_NOT_PROPAGATE_NAN )   {            \
             auto accuVal_new = opReduce(accuVal,currVal);       \
             if ( !miopen::float_equal(accuVal,accuVal_new) ) {  \
                  accuIndex = currIndex;                         \
		  accuVal = accuVal_new;                         \
	     };	                                                 \
        }                                                        \
        else {                                                   \
             decltype(accuVal) accuVal_new;                      \
             if ( ::isnan(currVal) )                             \
                  accuVal_new = currVal;                         \
             else                                                \
                accuVal_new = opReduce(accuVal,currVal);         \
                                                                 \
             if ( !miopen::float_equal(accuVal,accuVal_new) ) {  \
                  accuIndex = currIndex;                         \
		  accuVal = accuVal_new;                         \
	     };	                                                 \
        };                                                       \
     }

template <class T>
struct verify_reduce
{
    miopen::ReduceTensorDescriptor reduce;
    tensor<T> input;
    tensor<T> output; 
    tensor<T> workspace; 
    //tensor<int> indices; 
    T alpha; 
    T beta; 

    miopenReduceTensorOp_t reduceOp;
    miopenDataType_t compTypeVal;
    miopenNanPropagation_t nanOpt;
    miopenReduceTensorIndices_t indicesOpt;
    miopenIndicesType_t indicesType;

    verify_reduce(miopen::ReduceTensorDescriptor& reduce_, const tensor<T>& input_,  tensor<T>& output_,  tensor<T> & workspace_,  /*tensor<int> & indices_,*/
		                                                 T alpha_, T beta_)
    {
        reduce   = reduce_;
        input = input_; 
        output = output_; 
	workspace = workspace_; 
        // indices = indices_; 
	alpha = alpha_; 
	beta = beta_; 

        reduceOp = reduce.reduceTensorOp_; 
	compTypeVal = reduce.reduceTensorCompType_; 
	nanOpt = reduce.reduceTensorNanOpt_; 
	indicesOpt = reduce.reduceTensorIndices_; 
	indicesType = reduce.reduceTensorIndicesType_; 

	bool need_indices = ( indicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES ) && (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX);

        if ( need_indices ) {
             alpha = static_cast<T>(1.0f); 
       	     beta = static_cast<T>(0.0f); 
        };	
    }
  
    tensor<T> cpu() 
    {
        if ( compTypeVal == miopenFloat ) {
             if ( std::is_same<T, double>::value ) 
	          return( cpuImpl<double>() ); 
             else 	     
                  return( cpuImpl<float>() ); 
        }	
        else
        if ( compTypeVal == miopenHalf ) {
             if ( std::is_same<T, double>::value || std::is_same<T, float>::value )
                  return( cpuImpl<T>() ); 
             else 
                  return( cpuImpl<float16>() ); 
        }; 	

        return( tensor<T>{} ); 
    }; 

    template <typename compType> 
    tensor<T> cpuImpl() const  
    {
        auto inLengths = input.desc.GetLengths(); 
	auto outLengths = output.desc.GetLengths(); 
	auto inStrides = input.desc.GetStrides(); 
	auto outStrides = output.desc.GetStrides(); 

        // replicate 
        auto res = output; 
        //auto res_indices = indices; 

        std::vector<std::size_t> invariantLengths;
        std::vector<std::size_t> toReduceLengths;

        std::vector<int> invariantDims;
        std::vector<int> toReduceDims; 

        for (int i=0; i < inLengths.size(); i++)
             if ( inLengths[i] == outLengths[i] )
                  invariantDims.push_back(i);
             else 
		  toReduceDims.push_back(i); 

        for (int i=0; i < invariantDims.size(); i++)
             invariantLengths.push_back( inLengths[ invariantDims[i] ] );

        for (int i=0; i < toReduceDims.size(); i++)
             toReduceLengths.push_back( inLengths[ toReduceDims[i] ] );

        bool reduceAllDims = invariantDims.empty() ? true : false;

        auto opReduce = ReduceOpFn<compType>( reduceOp );
        bool need_indices = ( indicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES ) && (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX);

        if ( reduceAllDims ) {
             std::vector< std::vector<std::size_t> > indexes_1;

             get_all_indexes(inLengths, 0, indexes_1);

             compType accuVal = ReduceOpZeroVal<compType>( reduceOp );
             int accuIndex = 0; 

             // go through indexes of the invariant dimensions
             for (std::size_t i1=0; i1 < indexes_1.size(); i1++) {
                  std::vector<std::size_t> & src_index = indexes_1[i1];

                  auto src_offset = get_offset_from_index(inStrides, src_index);

                  auto currVal = static_cast<compType>( input.data[src_offset] );

                  if ( need_indices) {
                       int currIndex = get_flatten_offset(inLengths, src_index);  
                       binop_with_nan_check2(nanOpt,opReduce,accuVal,currVal,accuIndex,currIndex);
                  }
                  else 
                       binop_with_nan_check(nanOpt,opReduce,accuVal,currVal);
             };

             // scale the accumulated value
             if (  !miopen::float_equal(alpha, static_cast<T>(1.0)) )
                  accuVal = accuVal * static_cast<compType>(alpha);

             // scale the prior dst value and add it to the accumulated value
             if (  !miopen::float_equal(beta, static_cast<T>(0.0)) ) {
                  auto priorDstValue = static_cast<T>( res.data[0] );

                  accuVal += static_cast<compType>(priorDstValue * static_cast<T>(beta));
             };

             // store the reduced value to dst location
             res.data[0] = static_cast<T>(accuVal);
             //res_indices.data[0] = accuIndex; 
        }
        else {
             std::vector< std::vector<std::size_t> > indexes_1, indexes_2;

             get_all_indexes(invariantLengths, 0, indexes_1);
             get_all_indexes(toReduceLengths, 0, indexes_2);

             // go through indexes of the invariant dimensions
             for (std::size_t i1=0; i1 < indexes_1.size(); i1++) {
                  auto & index_1 = indexes_1[i1];
                  std::vector<std::size_t> src_index;
                  std::vector<std::size_t> dst_index;

                  src_index.resize( inLengths.size() );
                  dst_index.resize( inLengths.size() );

                  for (int k=0; k < dst_index.size(); k++)
                       dst_index[k] = 0;

                  for (int k=0; k < invariantDims.size(); k++)
                       dst_index[ invariantDims[k] ] = index_1[k];

                  int dst_offset = get_offset_from_index(outStrides, dst_index);

                  // generate the part of the index belonging to the invariant dims
                  for (int k=0; k < invariantDims.size(); k++)
                       src_index[ invariantDims[k] ] = index_1[k];

                  compType accuVal = ReduceOpZeroVal<compType>( reduceOp );
                  int accuIndex = 0; 

                  // go through indexes of the toReduce dimensions
                  for (std::size_t i2=0; i2 < indexes_2.size(); i2++) {
                       auto & index_2 = indexes_2[i2];

                       // generate the part of the index belonging to the toReduce dims
                       for (int k=0; k < toReduceDims.size(); k++)
                            src_index[ toReduceDims[k] ] = index_2[k];

                       auto src_offset = get_offset_from_index(inStrides, src_index);

                       auto currVal = static_cast<compType>( input.data[src_offset] );

                       if ( need_indices ) {
                            auto currIndex = get_flatten_offset(toReduceLengths, index_2);   
                            binop_with_nan_check2(nanOpt,opReduce,accuVal,currVal,accuIndex,currIndex);
                       }
                       else 
                            binop_with_nan_check(nanOpt,opReduce,accuVal,currVal);

                       // scale the accumulated value
                       if (  !miopen::float_equal(alpha, static_cast<T>(1.0)) )
                            accuVal = accuVal * static_cast<compType>(alpha);

                       // scale the prior dst value and add it to the accumulated value
                       if (  !miopen::float_equal(beta, static_cast<T>(0.0)) ) {
                            auto priorDstValue = static_cast<T>( res.data[dst_offset] );

                            accuVal += static_cast<compType>(priorDstValue * static_cast<T>(beta));
                       };
                  };

                  // store the reduced value to dst location
                  res.data[dst_offset] = static_cast<T>(accuVal);
                  //res_indices.data[dst_offset] = accuIndex;            // store the index
             };   // end of if
        };
	
        return(res); 
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto input_dev = handle.Write(input.data); 	
        auto output_dev = handle.Write(output.data); 

        // replicate 
        auto res = output; 

        auto workspace_dev = handle.Write(workspace.data); 
        //auto indices_dev = handle.Write(indices.data); 

	std::size_t ws_sizeInBytes = workspace.desc.GetElementSize() * sizeof(T); 


        reduce.ReduceTensor(get_handle(), /*indices_dev.get(),*/nullptr, static_cast<std::size_t>(0), workspace_dev.get(), ws_sizeInBytes,
                                                   static_cast<const void*>(&alpha), input.desc, input_dev.get(),
                                                   static_cast<const void*>(&beta), output.desc, output_dev.get()); 
        res.data = handle.Read<T>(output_dev, res.data.size());

        return(res); 
    }

    void fail(int) const
    {
        std::cout << "verify_reduce" << std::endl;
        std::cout << "Input Tensor" << " " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct reduce_driver : test_driver
{
    int reduceOp;       //  miopenReduceTensorOp_t reduceOp;
    int compTypeVal;    //  miopenDataType_t compTypeVal;
    int nanOpt;         //  miopenNanPropagation_t nanOpt;
    int indicesOpt;     //  miopenReduceTensorIndices_t indicesOpt;
    miopenIndicesType_t indicesType = MIOPEN_32BIT_INDICES;

    std::vector<std::size_t> inLengths;              // the lengths of the input tensor's dimensions
    std::vector<int> toReduceDims;           // the indexes of the dimensions to be reduced in the input tensor

    float alpha; 
    float beta; 

    std::vector<std::vector<std::size_t>> get_tensor_lengths()
    {
        return {{64, 3, 120, 120},
                {12, 3, 80, 8}, 
		{128, 4, 200, 200},
                {801, 4, 20, 20},
		{4, 4, 60, 300},
               };
    }

    std::vector<std::vector<int>> get_toreduce_dims()
    {
        return {{0}, 
                {0,2,3},
	        {1,2,3},
	        {0,1,2,3}, 
	        {1,2,3}
               };
    }

    reduce_driver()
    {
        add(inLengths, "D", generate_data(get_tensor_lengths()) );
        add(toReduceDims, "R", generate_data(get_toreduce_dims()) );  
        add(reduceOp, "ReduceOp", generate_data({0,2}) ); 
        add(compTypeVal, "CompType", generate_data({1}) ); 
        add(nanOpt, "N", generate_data({0,1}) ); 		
        add(indicesOpt, "I", generate_data({0}) ); 

        add(alpha, "alpha", generate_data({1.0f}) );
        add(beta, "beta", generate_data({0.0f}) ); 	

        auto&& handle = get_handle();
        handle.EnableProfiling();
    }

    void run()
    {
        miopen::ReduceTensorDescriptor reduceDesc(static_cast<miopenReduceTensorOp_t>(reduceOp), static_cast<miopenDataType_t>(compTypeVal), 
				                  static_cast<miopenNanPropagation_t>(nanOpt), static_cast<miopenReduceTensorIndices_t>(indicesOpt), indicesType); 

        auto outLengths = this->inLengths; 	

        assert( toReduceDims.size() <= outLengths.size() ); 
        for (int i=0; i < toReduceDims.size(); i++) 
	     assert( toReduceDims[i] < outLengths.size() ); 

        // set the lengths of the dimensions to be reduced to 1 to represent the output Tensor 
        for (int i=0; i < toReduceDims.size(); i++)
             outLengths[ toReduceDims[i] ] = static_cast<std::size_t>(1);

        unsigned long max_value = miopen_type<T>{} == miopenHalf ? 5 : miopen_type<T>{} == miopenInt8 ? 127 : 17;

        auto inputTensor = tensor<T>{this->inLengths}.generate(tensor_elem_gen_integer{max_value});
        auto outputTensor = tensor<T>{outLengths}.generate(tensor_elem_gen_integer{max_value});

        auto workspace_size = reduceDesc.GetWorkSpaceSize(get_handle(), inputTensor.desc, outputTensor.desc) / sizeof(T); 
        //auto indices_size = reduceDesc.GetIndicesSize(get_handle(), inputTensor.desc, outputTensor.desc) / sizeof(int); 
	 

        std::vector<std::size_t> wsLengths = {static_cast<std::size_t>(workspace_size), 1}; 
        //std::vector<std::size_t> indicesLengths = { static_cast<std::size_t>(indices_size), 1}; 

        auto workspaceTensor = tensor<T>{wsLengths}.generate(tensor_elem_gen_integer{max_value});
        //auto indicesTensor = tensor<int>{indicesLengths}.generate(tensor_elem_gen_integer{max_value}); 

        verify( verify_reduce<T>(reduceDesc, inputTensor, outputTensor, workspaceTensor, /*indicesTensor,*/ static_cast<T>(alpha), static_cast<T>(beta)) ); 
    };
};

int main(int argc, const char* argv[]) { test_drive<reduce_driver>(argc, argv); };
