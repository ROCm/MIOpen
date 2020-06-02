#ifndef GUARD_MIOPEN_REDUCTION_HOST_HPP_
#define GUARD_MIOPEN_REDUCTION_HOST_HPP_

#include <vector>
#include <functional>
#include <limits>
#include <type_traits>
#include <cassert>
#include <cmath>

#include <miopen/float_equal.hpp>

#include "tensor_driver.hpp"

static void get_all_indexes(const std::vector<int> & dimLengths, int dim, std::vector< std::vector<int> > & indexes)
{
   if ( dim < dimLengths.size() ) { 
        std::vector< std::vector<int> > updated_indexes; 	

        if ( dim == 0 ) {
             for (int i=0; i < dimLengths[dim]; i++) {
		  std::vector<int> index = { i }; 
	     
		  updated_indexes.push_back(index); 
             };  
        }
	else {
             // go through all the current indexes
             for (int k=0; k < indexes.size(); k++) 
                  for (int i=0; i < dimLengths[dim]; i++) {
                       std::vector<int> index = indexes[k]; 

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

static int get_offset_from_index(const std::vector<int> & strides, const std::vector<int> & index)
{
   int offset = 0; 

   assert( index.size() == strides.size() ); 
   for (int i=0; i < index.size(); i++) 
        offset += strides[i] * index[i]; 

   return(offset); 
}; 

static int get_flatten_offset(const std::vector<int> & lengths, const std::vector<int> & index)
{
   int offset = 0; 

   assert( lengths.size() == index.size() && lengths.size() > 0 ); 

   int len = lengths.size(); 
   int stride = 1; 

   // for len==1, the loop is not executed
   for (int i=len-1; i > 0; i--) {
        offset += stride * index[i]; 

        stride *= lengths[i]; 	     
   }; 	

   offset += stride * index[0]; 
   
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
	   return( [&](compType a_, compType b_) { return (a_ > b_)? b_ : a_; } );  // a is selected when they are equal

      case MIOPEN_REDUCE_TENSOR_MAX:
	   return( [&](compType a_, compType b_) { return (a_ < b_)? b_ : a_; } );  // a is selected when they are equal
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
             };                                                  \
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
             };                                                  \
        };                                                       \
     }

template <typename Tgpu, typename Tref>
class miopenReductionHost
{
public:
    miopenReductionHost() = default; 
    miopenReductionHost(const miopenReduceTensorDescriptor_t reduceDesc,
                        miopenTensorDescriptor_t inDesc, miopenTensorDescriptor_t outDesc,
                        const std::vector<int>& invariantDims_, const std::vector<int>& toReduceDims_) 
    {
        miopenGetReduceTensorDescriptor(reduceDesc, &reduceOp, &compTypeVal, &nanOpt, &indicesOpt, &indicesType);

        this->inLengths = GetTensorLengths(inDesc);
        this->outLengths = GetTensorLengths(outDesc);
        this->inStrides = GetTensorStrides(inDesc);
        this->outStrides = GetTensorStrides(outDesc);

        this->invariantDims = invariantDims_;
        this->toReduceDims = toReduceDims_;

        assert( this->inLengths.size() == this->outLengths ); 
        assert( !this->toReduceDims.empty() );

        for (int i=0; i < this->invariantDims.size(); i++)
             this->invariantLengths.push_back( this->inLengths[ this->invariantDims[i] ] );

        for (int i=0; i < this->toReduceDims.size(); i++)
             toReduceLengths.push_back( this->inLengths[ this->toReduceDims[i] ] );

        this->reduceAllDims = this->invariantDims.empty() ? true: false;
    }; 

    ~miopenReductionHost() {};  
 
    void Run(Tgpu alpha, const Tgpu *in_data, Tgpu beta, Tref *out_data, int *indices)
    {
        if ( compTypeVal == miopenFloat ) {
             if ( std::is_same<Tref, double>::value ) 
                  RunImpl<double>(alpha, in_data, beta, out_data, indices);
             else 	
                  RunImpl<float>(alpha, in_data, beta, out_data, indices);
        }	
        else 
        if ( compTypeVal == miopenHalf ) { 
             if ( std::is_same<Tref, double>::value || std::is_same<Tref, float>::value ) 
                  RunImpl<Tref>(alpha, in_data, beta, out_data, indices);
             else 	
                  RunImpl<float16>(alpha, in_data, beta, out_data, indices);
        }; 

        return; 	
    }; 

private: 
    miopenReduceTensorOp_t reduceOp;
    miopenDataType_t compTypeVal;
    miopenNanPropagation_t nanOpt;
    miopenReduceTensorIndices_t indicesOpt;
    miopenIndicesType_t indicesType;

    std::vector<int> inLengths;
    std::vector<int> outLengths; 
    std::vector<int> inStrides;
    std::vector<int> outStrides;

    std::vector<int> invariantLengths;
    std::vector<int> toReduceLengths; 

    std::vector<int> invariantDims; 
    std::vector<int> toReduceDims; 

    bool reduceAllDims; 

    template <typename compType> 
    void RunImpl(Tgpu alpha, const Tgpu *in_data, Tgpu beta, Tref *out_data, int *indices)
    {
        auto opReduce = ReduceOpFn<compType>( this->reduceOp ); 
        bool need_indices = ( indicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES ) && (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX);

        if ( reduceAllDims ) {
             std::vector< std::vector<int> > indexes_1; 
 
             get_all_indexes(inLengths, 0, indexes_1); 	 // generate the input indexes space  

             auto accuVal = ReduceOpZeroVal<compType>( this->reduceOp );  
             int accuIndex = 0; 

             // go through indexes of the invariant dimensions
             for (int i1=0; i1 < indexes_1.size(); i1++) {
                  std::vector<int> & src_index = indexes_1[i1]; 

                  auto src_offset = get_offset_from_index(this->inStrides, src_index);  

                  auto currVal = static_cast<compType>( in_data[src_offset] );

                  if ( need_indices ) {
                       auto currIndex = get_flatten_offset(inLengths, src_index);  
                       binop_with_nan_check2(nanOpt,opReduce,accuVal,currVal,accuIndex,currIndex);
                  }
                  else 
                       binop_with_nan_check(nanOpt,opReduce,accuVal,currVal);
             }; 

             // scale the accumulated value
             if (  !miopen::float_equal(alpha, static_cast<Tgpu>(1.0)) ) 
                  accuVal = accuVal * static_cast<compType>(alpha);

             // scale the prior dst value and add it to the accumulated value
             if (  !miopen::float_equal(beta, static_cast<Tgpu>(0.0)) ) {
                  auto priorDstValue = static_cast<Tref>( out_data[0] );

                  accuVal += static_cast<compType>(priorDstValue * static_cast<Tref>(beta));
             };

             // store the reduced value to dst location
             out_data[0] = static_cast<Tref>(accuVal); 
             if ( need_indices )
                  indices[0] = accuIndex; 
        }
        else {
             std::vector< std::vector<int> > indexes_1, indexes_2; 

	     get_all_indexes(this->invariantLengths, 0, indexes_1);  // generate the invariant indexes space
	     get_all_indexes(this->toReduceLengths, 0, indexes_2);   // generate the toReduce indexes space

             // go through indexes of the invariant dimensions
             for (int i1=0; i1 < indexes_1.size(); i1++) {
                  auto & index_1 = indexes_1[i1]; 
                  std::vector<int> src_index; 
                  std::vector<int> dst_index; 

	          src_index.resize( this->inLengths.size() ); 
	          dst_index.resize( this->inLengths.size() ); 

                  // generate the srd index 
                  for (int k=0; k < dst_index.size(); k++)
		       dst_index[k] = 0; 

	          for (int k=0; k < invariantDims.size(); k++)
		       dst_index[ invariantDims[k] ] = index_1[k]; 

                  int dst_offset = get_offset_from_index(this->outStrides, dst_index); 

                  // generate the part of src index belonging to invariant dims
                  for (int k=0; k < invariantDims.size(); k++)
		       src_index[ invariantDims[k] ] = index_1[k]; 

                  compType accuVal = ReduceOpZeroVal<compType>( this->reduceOp );  
                  int accuIndex = 0; 

                  // go through indexes of the toReduce dimensions
                  for (int i2=0; i2 < indexes_2.size(); i2++) {
                       auto & index_2 = indexes_2[i2]; 

                       // generate the part of src index belonging to toReduce dims
                       for (int k=0; k < toReduceDims.size(); k++)
		  	    src_index[ toReduceDims[k] ] = index_2[k];

                       auto src_offset = get_offset_from_index(this->inStrides, src_index); 

                       auto currVal = static_cast<compType>( in_data[src_offset] );

                       if ( need_indices ) {
                            auto currIndex = get_flatten_offset(toReduceLengths, index_2); 
                            binop_with_nan_check2(nanOpt,opReduce,accuVal,currVal,accuIndex,currIndex);
                       }
                       else
                            binop_with_nan_check(nanOpt,opReduce,accuVal,currVal);

                       // scale the accumulated value
                       if (  !miopen::float_equal(alpha, static_cast<Tgpu>(1.0)) ) 
                            accuVal = accuVal * static_cast<compType>(alpha);

		       // scale the prior dst value and add it to the accumulated value
                       if (  !miopen::float_equal(beta, static_cast<Tgpu>(0.0)) ) {
                            auto priorDstValue = static_cast<Tref>( out_data[dst_offset] );

                            accuVal += static_cast<compType>(priorDstValue * static_cast<Tref>(beta));
                       }; 
	          }; 

                  // store the reduced value to dst location
                  out_data[dst_offset] = static_cast<Tref>(accuVal); 
                  if ( need_indices )
                       indices[dst_offset] = accuIndex; 
             };   
        }; 
   };  // end of RunImpl()

}; 

#endif 
