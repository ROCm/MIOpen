
#include <miopen/miopen.h>
#include "test.hpp"
#include <array>
#include <iterator>
#include <memory>
#include <utility>
#include <iostream>
#include <miopen/tensor.hpp>
#include <miopen/batch_norm.hpp>
#include <limits>
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "driver.hpp"


struct deriveSpatialTensorTest{

  
    miopenTensorDescriptor_t ctensor;      
    miopenTensorDescriptor_t derivedTensor; 
    
    deriveSpatialTensorTest(){
        miopenCreateTensorDescriptor(&ctensor);
        miopenCreateTensorDescriptor(&derivedTensor);
        miopenSet4dTensorDescriptor(
                ctensor,
                miopenFloat,
                100,
                32,
                8,
                16);
            
        

    }
    
    void run(){
        std::array<int, 4> lens;
        miopenDataType_t dt;
        
        miopenDeriveBNTensorDescriptor(derivedTensor, ctensor, miopenBNSpatial);
        miopenGetTensorDescriptor(
                    derivedTensor,
                   &dt,
                    lens.data(),
                    nullptr);
        EXPECT(dt == miopenFloat);
        EXPECT(lens.size() == 4);
        EXPECT(lens[0] == 1);
        EXPECT(lens[1] == 32);
        EXPECT(lens[2] == 1);
        EXPECT(lens[3] == 1);
        
        
        
        
    }
    
    ~deriveSpatialTensorTest(){
        miopenDestroyTensorDescriptor(ctensor);
        miopenDestroyTensorDescriptor(derivedTensor);
    }
};


struct derivePerActTensorTest{

  
    miopenTensorDescriptor_t ctensor;      
    miopenTensorDescriptor_t derivedTensor; 
    
    derivePerActTensorTest(){
        miopenCreateTensorDescriptor(&ctensor);
        miopenCreateTensorDescriptor(&derivedTensor);
        miopenSet4dTensorDescriptor(
                ctensor,
                miopenFloat,
                100,
                32,
                8,
                16);
            
        

    }
    
    void run(){
        std::array<int, 4> lens;
        miopenDataType_t dt;
        
        miopenDeriveBNTensorDescriptor(derivedTensor, ctensor, miopenBNPerActivation);
        miopenGetTensorDescriptor(
                    derivedTensor,
                   &dt,
                    lens.data(),
                    nullptr);
        EXPECT(dt == miopenFloat);
        EXPECT(lens.size() == 4);
        EXPECT(lens[0] == 1);
        EXPECT(lens[1] == 32);
        EXPECT(lens[2] == 8);
        EXPECT(lens[3] == 16);
        
        
        
        
    }
    
    ~derivePerActTensorTest(){
        miopenDestroyTensorDescriptor(ctensor);
        miopenDestroyTensorDescriptor(derivedTensor);
    }
};



int main() { 
    run_test<deriveSpatialTensorTest>();
    run_test<derivePerActTensorTest>();
    
}
