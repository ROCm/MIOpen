Fusion API: Getting Started
===========================
## Introduction
Increasing depth of deep learning networks necessitate the need for novel mechanisms to improve performance on GPUs. One mechanism to achieve higher efficiency is to _fuse_ separate kernels into a single kernel to reduce off-chip memory access and avoid kernel launch overhead. This document outlines the addition of a Fusion API to the MIOpen library. The fusion API would allow users to specify operators that they wants to fuse in a single kernel, compile it and then launch the kernel. While not all combinations might be supported by the library, the API is flexible enough to allow the specification of many operations in any order from a finite set of supported operations. The API provides a mechanism to report unsupported combinations.

A complete example of the Fusion API in the context of MIOpen is given [here](https://github.com/ROCmSoftwarePlatform/MIOpenExamples/tree/master/fusion). We will use code from the example project as we go along. The example project creates a fusion plan to merge the convolution, bias and activation operations. For a list of supported fusion operations and associated constraints please refer to the [Supported Fusions](#supported_fusions) section. The example depicts bare-bones code without any error checking or even populating the tensors with meaningful data in the interest of simplicity.

The following list outlines the steps required 

- Create a fusion plan
- Create and add the convolution, bias and activation operators
- Compile the Fusion Plan 
- Set the runtime arguments for each operator
- Execute the fusion plan
- Cleanup

The above steps assume that an MIOpen handle object has already been initialized. Moreover, the order in which operators are created is important, since it represents the order of operations on the data itself. Therefore a fusion plan with convolution created before activation is a different fusion plan as opposed to if activation was added before convolution. 

The following sections further elaborate the above steps as well as give code examples to make these ideas concrete.

### Intended Audience
The primary consumers of the fusion API are high level frameworks such as TensorFlow/XLA or PyTorch etc.

## Create a Fusion Plan
A **Fusion Plan** is the data structure which holds all the metadata about the users fusion intent as well as logic to **Compile** and **Execute** a fusion plan. As mentioned earlier, a fusion plan holds the order in which different opertions would be applied on the data, but it also specifies the _axis_ of fusion as well. Currently only **vertical** (sequential) fusions are supported implying the flow of data between operations is sequential.

A fusion plan is created using the API call `miopenCreateFusionPlan` with the signature:

```cpp
miopenStatus_t
miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
const miopenFusionDirection_t fuseDirection,const miopenTensorDescriptor_t inputDesc);
``` 

The *input tensor descriptor* specifies the geometry of the incoming data. Since the data geometry of the intermediate operations can be derived from the *input tensor descriptor*, therefore only the *input tensor descriptor* is required for the fusion plan and not for the individual operations. In our fusion example the following lines of code accomplish this:
```cpp
miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, input.desc);
```
Where `fusePlanDesc` is an object of type `miopenFusionPlanDescriptor_t` and `input.desc` is the `miopenTensorDescriptor_t` object.

## Create and add Operators
The fusion API introduces the notion of **operators** which represent different operations that are intended to be fused together by the API consumer. Currently, the API supports the following operators:

*    Convolution Forward
*    Activation Forward
*    BatchNorm Inference
*    Bias Forward

Notice that _Bias_ is a separate operator, although it is typically only used with convolution. This list is expected to grow as support for more operators is added to the API, moreover, operators for backward passes are in the works as well.

The fusion API provides calls for the creation of the supported operators, here we would describe the process for the convolution operator, details for other operators may be found in the [miopen header file](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/fusion.html) 

Once the fusion plan descriptor is created, two or more operators can be added to it by using the individual operator creation API calls. Creation of an operator might fail if the API does not support the fusion of the operations being added and report back immediately to the user. For our example we need to add the Convolution, Bias and Activation operations to our freshly minted fusion plan. This is done using the following calls for the Convolution, Bias and Activation operations respectively:

```cpp
miopenStatus_t
miopenCreateOpConvForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                          miopenFusionOpDescriptor_t* convOp,
                          miopenConvolutionDescriptor_t convDesc,
                          const miopenTensorDescriptor_t wDesc);
miopenStatus_t
miopenCreateOpBiasForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                          miopenFusionOpDescriptor_t* biasOp,
                          const miopenTensorDescriptor_t bDesc);

miopenStatus_t
miopenCreateOpActivationForward(miopenFusionPlanDescriptor_t fusePlanDesc,
                                miopenFusionOpDescriptor_t* activOp,
                                miopenActivationMode_t mode);
```

The following lines in the fusion example project use these API calls to create and insert the operators in the fusion plan:

```cpp
miopenCreateOpConvForward(fusePlanDesc, &convoOp, conv_desc, weights.desc);
miopenCreateOpBiasForward(fusePlanDesc, &biasOp, bias.desc);
miopenCreateOpActivationForward(fusePlanDesc, &activOp, miopenActivationRELU);
```

It may be noted that `conv_desc` is the regular MIOpen Convolution descriptor and is created in the standard way before it is referenced here. For more details on creating and setting the convolution descriptor please refer to the example code as well as the [MIOpen documentation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/convolution.html). In the above snippet `weights.desc` refers to the `miopenTensorDescriptor_t` for the convolution operations and `bias.desc` refers to the object of the same type for the bias operation. The order of insertion of operators indicates the order in which the operations would be performed on the data. Therefore, the above code implies that the convolution operation would be the first operation to execute on the incoming data, followed by the bias and activation operations. 

During this process, it is important that the returned codes be checked to make sure that the operations as well as their order is supported. The operator insertion might fail for a number of reasons such as unsupported sequence of operations, unsupported dimensions of the input or in case of convolution unsupported dimensions for the filters. In the above example, these aspects are ignored for the sake of simplicity.

## <a name="compile_fusion"></a>Compile the Fusion Plan

Following the operator addition, the user would compile the fusion plan, to populate the MIOpen kernel cache with the fused kernel and make it ready for execution. The API call that accomplishes this is:

```cpp
miopenStatus_t
miopenCompileFusionPlan(miopenHandle_t handle, miopenFusionPlanDescriptor_t fusePlanDesc);
```

The corresponding code snippet in the example is as follows:

```cpp
auto status = miopenCompileFusionPlan(mio::handle(), fusePlanDesc);
if (status != miopenStatusSuccess) {
return -1;
}
```
In order to compile the fusion plan, the user is assumed to have acquired an MIOpen handle object, in the example code above this is accomplished using the `mio::handle()` helper function. While a fusion plan itself is not bound to a MIOpen handle object, it would however need to be recompiled for each handle separately. It may be noted that compilation of a fusion plan might fail for a number of reasons, moreover it is not assured that a fused version of the kernel would offer any performance improvement over the separately run kernels.

Compiling a fusion plan is a costly operation in terms of run-time. Therefore, it is recommended that a fusion plan should only be compiled once and may be reused for execution with different runtime parameters as described in the next section. 

## Set the runtime arguments

While the underlying MIOpen descriptor of the fusion operator specifies the data geometry and parameters, the fusion plan still needs access to the data to execute a successfully compiled fusion plan. The arguments mechanism in the Fusion API provides such data before a fusion plan may be executed. For example the convolution operator requires *weights* to carry out the convolution computation, a bias operator requires the actual bias values etc. Therefore, before a fusion plan may be executed, arguments required by each fusion operator need to be specified. To begin, we create the `miopenOperatorArgs_t` object using:

```cpp
miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args);
``` 

Once created, runtime arguments for each operation may be set. In our running example, the forward convolution operator requires the convolution weights argument which is supplied using the API call:

```cpp
miopenStatus_t
miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
								const miopenFusionOpDescriptor_t convOp,
                           const void* alpha,
                           const void* beta,
                           const void* w);
```

Similarly the parameters for bias and activation are given by:

```cpp
miopenStatus_t miopenSetOpArgsBiasForward(miopenOperatorArgs_t args,
                                          const miopenFusionOpDescriptor_t biasOp,
                                          const void* alpha,
                                          const void* beta,
                                          const void* bias);
                                          
miopenStatus_t miopenSetOpArgsActivForward(miopenOperatorArgs_t args,
                                           const miopenFusionOpDescriptor_t activOp,
                                           const void* alpha,
                                           const void* beta,
                                           double activAlpha,
                                           double activBeta,
                                           double activGamma);
```

In our example code, we set the arguments for the operations as follows:

```cpp
miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, weights.data);
miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta, activ_alpha,
                          activ_beta, activ_gamma);
miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, bias.data);
```

This separation between the fusion plan and the arguments required by each operator allows better reuse of the fusion plan with different argument as well as avoids the necessity of recompiling the fusion plan to run the same combination of operators with different arguments. 

As mentioned in the section [Compile the Fusion Plan](#compile_fusion) earlier, the compilation step for a fusion plan might be costly, therefore a fusion plan should only be compiled once in its lifetime. A fusion plan needs not be recompiled if the input desciptor or any of the parameters to the `miopenCreateOp*` API calls are different, otherwise a compiled fusion plan may be reused again and again with a different set of arguments. In our example this is demonstrated in lines 77 - 85 of `main.cpp`. 

## Execute a Fusion Plan

Once the fusion plan has been compiled and arguments set for each operator, it may be executed with the API call given below passing it the actual data to be processed.

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

The following code snippet in the example accomplishes the fusion plan execution:

```cpp
miopenExecuteFusionPlan(mio::handle(), fusePlanDesc, input.desc, input.data,
                        output.desc, output.data, fusionArgs);
```

It may be noted that it is an error to attempt to execute a fusion plan that is either not compiled or has been invalidated by changing the input tensor descriptor or any of the operation parameters. 


## Cleanup
Once the application is done with the fusion plan, the fusion plan and the fusion args objects may be destroyed using the API calls:

```cpp
miopenStatus_t miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc);

miopenStatus_t miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc);
```
Once the fusion plan object is destroyed, all the operations created are destroyed automatically and do not need any special cleanup.


## <a name="supported_fusions"></a> Supported Fusions
The table below outlines the supported fusions as well as any applicable constraints. Currently, only convolutions with unit stride and unit dilation are supported. Currently, the fusion API is in the initial phases of development and may change.

<table border=0 cellpadding=0 cellspacing=0 width=713 style='border-collapse:
 collapse;table-layout:fixed;width:534pt'>
 <col width=108 style='mso-width-source:userset;mso-width-alt:3456;width:81pt'>
 <col width=87 style='width:65pt'>
 <col width=221 style='mso-width-source:userset;mso-width-alt:7082;width:166pt'>
 <col width=87 style='width:65pt'>
 <col width=123 style='mso-width-source:userset;mso-width-alt:3925;width:92pt'>
 <col width=87 style='width:65pt'>
 <tr height=45 style='height:34.0pt'>
  <td height=45 class=xl65 width=108 style='height:34.0pt;width:81pt'>Combination</td>
  <td class=xl65 width=87 style='width:65pt'>Conv Algo</td>
  <td class=xl65 width=221 style='width:166pt'>Filter Dims</td>
  <td class=xl65 width=87 style='width:65pt'>BN Mode</td>
  <td class=xl65 width=123 style='width:92pt'>Activations</td>
  <td class=xl65 width=87 style='width:65pt'>Other Constraints</td>
 </tr>
 <tr height=45 style='height:34.0pt'>
  <td height=45 class=xl66 width=108 style='height:34.0pt;width:81pt'>CBNA</td>
  <td class=xl66 width=87 style='width:65pt'>Direct</td>
  <td class=xl66 width=221 style='width:166pt'>1x1, 3x3, 5x5, 7x7, 9x9, 11x11</td>
  <td class=xl66 width=87 style='width:65pt'>All</td>
  <td class=xl66 width=123 style='width:92pt'>All</td>
  <td class=xl66 width=87 style='width:65pt'>Padding not supported</td>
 </tr>
 <tr height=23 style='height:17.0pt'>
  <td rowspan=2 height=46 class=xl67 width=108 style='height:34.0pt;width:81pt'>CBA</td>
  <td class=xl66 width=87 style='width:65pt'>Direct</td>
  <td class=xl66 width=221 style='width:166pt'>1x1, 3x3, 5x5, 7x7, 9x9, 11x11</td>
  <td class=xl66 width=87 style='width:65pt'></td>
  <td class=xl66 width=123 style='width:92pt'>All</td>
  <td class=xl66 width=87 style='width:65pt'></td>
 </tr>
 <tr height=23 style='height:17.0pt'>
  <td height=23 class=xl66 width=87 style='height:17.0pt;width:65pt'>Winograd</td>
  <td class=xl66 width=221 style='width:166pt'>3x3</td>
  <td class=xl66 width=87 style='width:65pt'>N/A</td>
  <td class=xl66 width=123 style='width:92pt'>Relu, Leaky Relu</td>
  <td class=xl66 width=87 style='width:65pt'>c &gt;= 18</td>
 </tr>
 <tr height=45 style='height:34.0pt'>
  <td height=45 class=xl66 width=108 style='height:34.0pt;width:81pt'>NA</td>
  <td class=xl66 width=87 style='width:65pt'>-</td>
  <td class=xl66 width=221 style='width:166pt'>-</td>
  <td class=xl66 width=87 style='width:65pt'>All</td>
  <td class=xl66 width=123 style='width:92pt'>All</td>
  <td class=xl66 width=87 style='width:65pt'>Padding not supported</td>
 </tr>
</table>


## <a name="supported_fusions"></a> Performance Comparison to Non-Fused Kernels
The table below shows some of the tested configurations and the respective increase in performance. Other supported configurations are not shown here.

All configurations have a batch size of 64.

<table border=0 cellpadding=0 cellspacing=0 width=894 style='border-collapse:
 collapse;table-layout:fixed;width:672pt'>
 <col width=87 style='mso-width-source:userset;mso-width-alt:2377;width:65pt'>
 <col width=89 span=8 style='mso-width-source:userset;mso-width-alt:2450;
 width:67pt'>
 <col width=95 style='mso-width-source:userset;mso-width-alt:2596;width:71pt'>
 <tr class=xl65 height=24 style='height:18.0pt'>
  <td height=24 class=xl71 width=87 style='height:18.0pt;width:65pt;font-size:
  12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl66 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl66 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl66 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl66 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl66 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl66 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl66 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl74 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border-top:.5pt solid windowtext;
  border-right:none;border-bottom:.5pt solid windowtext;border-left:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>Speedup</td>
  <td class=xl77 width=95 style='width:71pt;font-size:12.0pt;color:black;
  font-weight:700;text-decoration:none;text-underline-style:none;text-line-through:
  none;font-family:Candara, sans-serif;border-top:.5pt solid windowtext;
  border-right:.5pt solid windowtext;border-bottom:.5pt solid windowtext;
  border-left:none;background:#D9D9D9;mso-pattern:#D9D9D9 none'>Speedup</td>
 </tr>
 <tr height=76 style='height:57.0pt'>
  <td height=76 class=xl72 width=87 style='height:57.0pt;border-top:none;
  width:65pt;font-size:12.0pt;color:black;font-weight:700;text-decoration:none;
  text-underline-style:none;text-line-through:none;font-family:Candara, sans-serif;
  border:.5pt solid windowtext'>Fusion Mode</td>
  <td class=xl67 width=89 style='border-top:none;border-left:none;width:67pt'>Input
  Channels</td>
  <td class=xl67 width=89 style='border-top:none;border-left:none;width:67pt'>Height</td>
  <td class=xl67 width=89 style='border-top:none;border-left:none;width:67pt'>Width</td>
  <td class=xl67 width=89 style='border-top:none;border-left:none;width:67pt'>Conv.
  Channels</td>
  <td class=xl67 width=89 style='border-top:none;border-left:none;width:67pt'>Filter
  Height and Width</td>
  <td class=xl67 width=89 style='border-top:none;border-left:none;width:67pt'>kernel
  time (ms)</td>
  <td class=xl67 width=89 style='border-top:none;border-left:none;width:67pt'>wall
  time (ms)</td>
  <td class=xl67 width=89 style='border-top:none;border-left:none;width:67pt'>kernel
  time</td>
  <td class=xl75 width=95 style='border-top:none;border-left:none;width:71pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>wall
  time</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1024</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>14</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>14</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>256</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.949</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.705</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.218</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.188</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>128</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>28</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>28</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>128</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>3</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.793</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.897</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.287</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.369</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>128</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>28</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>28</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>512</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.066</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.684</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.991</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.369</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2048</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>7</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>7</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>512</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.156</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.714</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.010</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.787</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>256</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>14</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>14</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1024</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.217</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.721</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.371</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.930</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>256</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>14</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>14</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>256</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>3</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.640</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.706</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.173</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.239</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>256</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>64</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.881</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.622</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.653</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.858</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>512</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>28</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>28</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>128</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.999</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.629</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.276</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.849</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>512</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>7</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>7</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2048</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.247</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.755</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.152</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.855</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>512</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>7</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>7</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>512</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>3</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.731</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.811</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.089</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.140</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>64</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>256</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.222</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.712</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.698</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.732</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>64</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>55</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>55</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>64</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.319</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.384</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.683</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.968</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-B-A</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>64</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>64</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>3</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.893</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.464</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.550</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.945</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl68 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl68 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl68 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl68 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl68 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl70 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl66 style='border-top:none;border-left:none'>Average</td>
  <td class=xl69 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.550</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.171</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=87 style='width:65pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=95 style='width:71pt'></td>
 </tr>
 <![endif]>
</table>


<table border=0 cellpadding=0 cellspacing=0 width=894 style='border-collapse:
 collapse;table-layout:fixed;width:672pt'>
 <col width=87 style='mso-width-source:userset;mso-width-alt:2377;width:65pt'>
 <col width=89 span=8 style='mso-width-source:userset;mso-width-alt:2450;
 width:67pt'>
 <col width=95 style='mso-width-source:userset;mso-width-alt:2596;width:71pt'>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 width=87 style='height:18.0pt;width:65pt;font-size:
  12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt'>&nbsp;</td>
  <td class=xl74 width=89 style='border-left:none;width:67pt'>Speedup</td>
  <td class=xl77 width=95 style='width:71pt;font-size:12.0pt;color:black;
  font-weight:700;text-decoration:none;text-underline-style:none;text-line-through:
  none;font-family:Candara, sans-serif;border-top:.5pt solid windowtext;
  border-right:.5pt solid windowtext;border-bottom:.5pt solid windowtext;
  border-left:none'>Speedup</td>
 </tr>
 <tr height=76 style='height:57.0pt'>
  <td height=76 class=xl72 width=87 style='height:57.0pt;border-top:none;
  width:65pt;font-size:12.0pt;color:black;font-weight:700;text-decoration:none;
  text-underline-style:none;text-line-through:none;font-family:Candara, sans-serif;
  border:.5pt solid windowtext;background:#D9D9D9;mso-pattern:#D9D9D9 none'>Fusion
  Mode</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>Input Channels</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>Height</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>Width</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>Conv. Channels</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>Filter Height and Width</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>kernel time (ms)</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>wall time (ms)</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>kernel time</td>
  <td class=xl75 width=95 style='border-top:none;border-left:none;width:71pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>wall time</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1024</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.941</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.359</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.132</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.427</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>128</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>128</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>3</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.791</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.889</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.135</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.248</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>128</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.047</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.186</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.477</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.891</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2048</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.153</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.272</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.955</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.059</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1024</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.209</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.345</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.106</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.195</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>3</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.640</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.705</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.063</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.233</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.879</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.005</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.339</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.381</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>128</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.991</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.137</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.144</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.194</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2048</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.250</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.367</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.008</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.100</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>3</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.728</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.818</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.017</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.134</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.216</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.358</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.760</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.187</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.319</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.375</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.809</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.013</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>C-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>3</td>
  <td class=xl70 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.8950</td>
  <td class=xl70 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.0372</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.258</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.334</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl73 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl67 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl67 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl67 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl67 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl67 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl69 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl65 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>Average</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.246</td>
  <td class=xl76 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.415</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=87 style='width:65pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=95 style='width:71pt'></td>
 </tr>
 <![endif]>
</table>


<table border=0 cellpadding=0 cellspacing=0 width=894 style='border-collapse:
 collapse;table-layout:fixed;width:672pt'>
 <col width=87 style='mso-width-source:userset;mso-width-alt:2377;width:65pt'>
 <col width=89 span=8 style='mso-width-source:userset;mso-width-alt:2450;
 width:67pt'>
 <col width=95 style='mso-width-source:userset;mso-width-alt:2596;width:71pt'>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl69 width=87 style='height:18.0pt;width:65pt;font-size:
  12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl65 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>&nbsp;</td>
  <td class=xl72 width=89 style='border-left:none;width:67pt;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border-top:.5pt solid windowtext;
  border-right:none;border-bottom:.5pt solid windowtext;border-left:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>Speedup</td>
  <td class=xl80 width=95 style='width:71pt;font-size:12.0pt;color:black;
  font-weight:700;text-decoration:none;text-underline-style:none;text-line-through:
  none;font-family:Candara, sans-serif;border-top:.5pt solid windowtext;
  border-right:.5pt solid windowtext;border-bottom:.5pt solid windowtext;
  border-left:none;background:#D9D9D9;mso-pattern:#D9D9D9 none'>Speedup</td>
 </tr>
 <tr height=76 style='height:57.0pt'>
  <td height=76 class=xl70 width=87 style='height:57.0pt;border-top:none;
  width:65pt;font-size:12.0pt;color:black;font-weight:700;text-decoration:none;
  text-underline-style:none;text-line-through:none;font-family:Candara, sans-serif;
  border:.5pt solid windowtext'>Fusion Mode</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt'>Input
  Channels</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt'>Height</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt'>Width</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt'>Conv.
  Channels</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt'>Filter
  Height and Width</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt'>kernel
  time (ms)</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt'>wall
  time (ms)</td>
  <td class=xl66 width=89 style='border-top:none;border-left:none;width:67pt'>kernel
  time</td>
  <td class=xl73 width=95 style='border-top:none;border-left:none;width:71pt;
  font-size:12.0pt;color:black;font-weight:700;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>wall
  time</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1024</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.269</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.322</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.374</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.577</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>128</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>128</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>3</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.135</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.185</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.209</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.615</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>128</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.135</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.185</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.336</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.763</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2048</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.135</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.183</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>3.250</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>3.452</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1024</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.072</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.122</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.065</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.824</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>14</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>3</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.073</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.122</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1.826</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.667</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.090</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1.241</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.338</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.610</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>28</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>128</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.557</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.617</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.680</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>4.144</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2048</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.046</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.093</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.467</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>3.129</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>7</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>512</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>3</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.047</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.101</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.091</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.414</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>256</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.264</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.315</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.432</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.655</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>1</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.264</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>0.312</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.462</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.678</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl71 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>N-A</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>55</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>64</td>
  <td class=xl67 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>3</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.265</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>0.315</td>
  <td class=xl68 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.376</td>
  <td class=xl74 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext;
  background:#D9D9D9;mso-pattern:#D9D9D9 none'>2.593</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl75 style='height:18.0pt;border-top:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl76 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl76 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl76 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl76 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl76 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl76 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:400;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>&nbsp;</td>
  <td class=xl77 style='border-top:none;border-left:none;font-size:12.0pt;
  color:black;font-weight:700;text-decoration:none;text-underline-style:none;
  text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>Average</td>
  <td class=xl78 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.377</td>
  <td class=xl79 align=right style='border-top:none;border-left:none;
  font-size:12.0pt;color:black;font-weight:400;text-decoration:none;text-underline-style:
  none;text-line-through:none;font-family:Candara, sans-serif;border:.5pt solid windowtext'>2.855</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=87 style='width:65pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=89 style='width:67pt'></td>
  <td width=95 style='width:71pt'></td>
 </tr>
 <![endif]>
</table>


