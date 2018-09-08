Fusion API: Getting Started
===========================
## Introduction
With the increase in the depth of deep learning networks and a requirement for faster kernels it is imperative that more ways be sought to improve the performance of GPU hardware. One mechanism to achieve higher efficiency is to _fuse_ separate kernels into a single kernel to reduce off-chip memory access and avoid kernel launch overhead. This document outlines the proposed addition of a Fusion API to the MIOpen library. The fusion API would allow users to specify operators that he/she wants to fuse in a single kernel, compile it and then launch the kernel. While not all combinations might be supported by the library, the API is flexible enough to allow the specification of many operations in any order from a finite set of supported operators. All combinations of operators might not be supported, therefore the API provides a mechanism to report combinations that are not supported.

Let us assume that a user wishes to fuse a convolution and activation operation together, the following list outlines the steps required 

- Create a fusion plan
- Create and add the convolution and activation operators
- Compile the Fusion Plan 
- If the above succeeds, execute the fusion plan

The above steps assume that an MIOpen handle object has already been initialized. Moreover, the order in which operators are created is important, since it represents the order of operations on the data itself. Therefore a fusion plan with convolution created before activation is a different fusion plan as opposed to if activation was added before convolution. 

The following sections further elaborate the above steps as well as give code examples to make these ideas concrete.

## Intended Audience
The primary consumer of the fusion API are high level frameworks such as TensorFlow/XLA etc.

 
## Fusion Plan
A **Fusion Plan** is the uber data structure which holds all the metadata about the users fusion intent as well as logic to **Compile** and **Execute** a fusion plan. As mentioned earlier, a fusion plan holds the order in which different opertions would be applied on the data, but it also specifies the _axis_ of fusion as well. Therefore, a user might wish to fuse operations in a **vertical** (sequential) directions such as the convolution/ activation fusion mentioned in the introduction. Alternatively, the API supports the specification of **horizontal** (parallel) operations fusions. While vertical fusions are more ubiquitous, horizontal fusions might be useful in networks such as inception, where different operation operate on the same data. The current version of the API only supports vertical fusions.

A fusion plan is created using the API call:

```cpp
miopenStatus_t
miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
const miopenFusionDirection_t fuseDirection,const miopenTensorDescriptor_t inputDesc);
``` 
The *input descriptor* specifies the geometry of the incoming data. Since the data geometry of the intermediate operations can be derived from the input tensor, therefore only the input tensor is required for the fusion plan and not for the individual operations.

Once the fusion plan descriptor is created, different operators can be added to it by using the individual operator creation API calls. Creation of an operator might fail if the API does not support the fusion of the operations being added and report back immediately to the user.

Following the operator addition, the user would compile the fusion plan, to populate the MIOpen kernel cache with the fused kernel and make it ready for execution. The API call that accomplishes this is:

```cpp
miopenStatus_t
miopenCompileFusionPlan(miopenHandle_t handle, miopenFusionPlanDescriptor_t fusePlanDesc);
```
In order to compile the fusion plan, the user is assumed to have acquired an MIOpen handle object. While a fusion plan itself is not bound to a handle object, an instance of a fusion plan that is compiled with a particular handle is bound to the same handle. It may be noted that compilation of a fusion plan might fail for a number of reasons, moreover it is not assured that a fused version of the kernel would offer any performance improvement over the separately run kernels.

Finally, the compiled fusion plan may be executed with the API call given below passing it the actual data to be processed.

```cpp
miopenStatus_t
miopenExecuteFusionPlan(const miopenHandle_t handle,
                        const miopenFusionPlanDescriptor_t fusePlanDesc,
                        const miopenTensorDescriptor_t inputDesc,
                        const void* input,
                        const miopenTensorDescriptor_t outputDesc,
                        void* output,
                        miopenOperatorArgs_t args);
```
It may be noted that it is an error to attempt to execute a fusion plan that is either not compiled or is invalid. 

The *args* parameter would be discussed in a later section. The same fusion plan in its compiled state may be executed again and again with different data to amortize the compilation cost. This would become clearer with the discussion of the *args* parameter. Once the user is finished with the fusion plan it may be destroyed using the `miopenDestroyFusionPlan` call.

While the fusion plan forms the glue for the different fused operations, the following section outlines the currently supported operations providing more detail.

## Operators
The fusion API introduces the notion of **operators** which represent different operations that are intended to be fused together by the API consumer. Currently, the API supports the following operators:

*    Convolution Forward
*    Activation Forward
*    BatchNorm Inference
*    Bias Forward

Notice that _Bias_ is a separate operator, although it is typically only used with convolution. This list is expected to grow as support for more operators is added to the API, moreover, operators for backward passes are in the works as well.

The fusion API provides calls for the creation of the supported operators, here we would describe the process for the convolution operator, details for other operators may be found in the [miopen header file][1]. 

The forward convolution fusion operator may be created using the API call:

```cpp
miopenStatus_t
miopenCreateOpConvForwardAlgo(miopenFusionPlanDescriptor_t fusePlanDesc,
                              miopenFusionOpDescriptor_t* convOp,
                              miopenConvolutionDescriptor_t convDesc,
                              miopenConvFwdAlgorithm_t fwdAlgo,
                              const miopenTensorDescriptor_t wDesc);
```
It may be noted that the fusion operator requires the regular MIOpen Convolution Descriptor (`convDesc`) as well as the MIOpen convolution algorithm (`fwdAlgo`). The only supported convolution algorithm supported is `miopenConvolutionFwdAlgoDirect`. This API call not only creates the Fusion Operator Descriptor `convOp` but also adds it to the fusion plan as mentioned above. The operator derives its input tensor geometry from its input own descriptor as well as the output tensor geometry of the preceeding operator in the fusion plan.

### Operator Arguments
While the underlying MIOpen descriptor of the fusion operator specifies the data geometry and parameters, the fusion plan still needs access to the data to execute a successfully compiled fusion plan. The arguments mechanism in the Fusion API provides such data before a fusion plan may be executed. For example the convolution operator requires *weights* to carry out the convolution computation, a bias operator requires the actual bias values etc. Therefore, before a fusion plan may be executed, arguments required by each fusion operator need to be specified. In our running example, the forward convolution operator requires the convolution weights argument which is supplied using the API call:
```cpp
miopenStatus_t
miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
								const miopenFusionOpDescriptor_t convOp,
                           const void* alpha,
                           const void* beta,
                           const void* w);
```

The `args` parameter in the above call is the fusion args object which holds arguments for all the operators in the fusion plan and must be created before any operator related call using the API call:
```cpp
miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args);
```
and destroyed afterwards using the the appropriate call. 

This separation between the fusion plan and the arguments required by each operator allows better reuse of the fusion plan with different argument as well as avoids the necessity of recompiling the fusion plan to run the same combination of operators with different arguments.

Once the operator arguments are set, the fusion plan may be executed. Finally, the operator descriptor would be destroyed when no longer needed.

##Convolution + Activation Fusion Example
The following example assumes that MIOpen as well as the regular MIOpen Convolution and Activation descriptors have been initialized using the requisite API calls and only focuses on the Fusion part of the process.

We begin by creating the fusion plan descriptor as well as the arguments for the fusion plan :

```cpp
        miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, inputTensor);
        miopenCreateOperatorArgs(&fusionArgs);
```
Here we specify a veritcal fusion plan and specify the input tensor as `inputTensor`. Now we can begin adding the different operators in the required order:


```cpp
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    miopenCreateOpConvForwardAlgo(fusePlanDesc,
                                  &convoOp,
                                  convDesc,
                                  miopenConvolutionFwdAlgoDirect,
                                  weightTensor);
    miopenCreateOpActivationForward(fusePlanDesc, &activOp, miopenActivationRELU);
    miopenError = miopenCompileFusionPlan(GetHandle(), fusePlanDesc);
    if(miopenError != miopenStatusSuccess)
    {
        std::cerr << "ConvBiasActivInference plan not supported." << std::endl;
        exit(EXIT_FAILURE);
    }
```
Which creates the Forward Convolution and Activation (`miopenActivationRELU`) operators. `convDesc` is the MIOpen convolution descriptor and `weightTensor` is the tensor representing the weights for the convolution operation, however, the actual weight values are not part of the operator itself and would be added to the argument object as mentioned before. Finally, the fusion plan is compiled making sure that the fusion plan is supported by the implementation.

If the plan compiles successfully, arguments for the operator are set and the plan is launched. 

```cpp
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);
    miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, wei_dev->GetMem());
    miopenSetOpArgsActivForward(
        fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
    
    for(int it = 0; it < iters; it++)
    {
        startTiming();
        miopenExecuteFusionPlan(GetHandle(),
                                fusePlanDesc,
                                inputTensor,
                                in_dev->GetMem(),
                                outputTensor,
                                out_dev->GetMem(),
                                fusionArgs);
        finishTiming(it);
    }
```
Where the helper function `GetHandle()` returns the MIOpen Handle and `in_dev` and `out_dev` represent device memory pointer holder. The `outputTensor` is the tensor descriptor for the output of the fusion plan. 

Also note that the `miopenExecuteFusionPlan` may be called again and again with differnt `fusionArgs` without the need to recompile the fusion plan.

The above example is from the `CBAInfer` driver provided in the `fusion-dev-core` branch and supplies other examples of supported fusion plans. A test script `fusion_tests.sh` is also provided in the root directory to execute all the fusions currently supported through the same driver.
