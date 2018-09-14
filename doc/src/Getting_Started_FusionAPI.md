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
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl64 width=108 style='height:16.0pt;width:81pt'>Combination <legend>C: Convolution <br/> B: Bias <br/> N: Batch Norm <br/> A: Activation</legend> </td>
  <td class=xl64 width=87 style='width:65pt'>Conv Algo</td>
  <td class=xl64 width=221 style='width:166pt'>Filter Dims</td>
  <td class=xl64 width=87 style='width:65pt'>BN Mode</td>
  <td class=xl64 width=123 style='width:92pt'>Activations</td>
  <td class=xl64 width=87 style='width:65pt'>Other Constr<span
  style='display:none'>aints</span></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 style='height:16.0pt'>C-B-N-A</td>
  <td>Direct</td>
  <td>1x1, 3x3, 5x5, 7x7, 9x9, 11x11</td>
  <td>All</td>
  <td>All</td>
  <td></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=2 height=42 class=xl63 style='height:32.0pt'>C-B-A</td>
  <td>Direct</td>
  <td>1x1, 3x3, 5x5, 7x7, 9x9, 11x11</td>
  <td>--</td>
  <td>All</td>
  <td></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 style='height:16.0pt'>Winograd</td>
  <td>3x3</td>
  <td>--</td>
  <td>Relu, Leaky Relu</td>
  <td>c &gt;= 18</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 style='height:16.0pt'>N-A</td>
  <td>-</td>
  <td>-</td>
  <td>All</td>
  <td>All</td>
  <td></td>
 </tr>
</table>

## <a name="supported_fusions"></a> Performance Comparison to Non-Fused Kernels
The table below shows some of the tested configurations and the respective increase in performance. Other supported configurations are not shown here.

<table border=0 cellpadding=0 cellspacing=0 width=1159 style='border-collapse:
 collapse;table-layout:fixed;width:869pt'>
 <col width=99 style='mso-width-source:userset;mso-width-alt:2706;width:74pt'>
 <col width=79 style='mso-width-source:userset;mso-width-alt:2157;width:59pt'>
 <col width=113 style='mso-width-source:userset;mso-width-alt:3108;width:85pt'>
 <col width=56 style='mso-width-source:userset;mso-width-alt:1536;width:42pt'>
 <col width=52 style='mso-width-source:userset;mso-width-alt:1426;width:39pt'>
 <col width=117 style='mso-width-source:userset;mso-width-alt:3218;width:88pt'>
 <col width=175 style='mso-width-source:userset;mso-width-alt:4790;width:131pt'>
 <col width=123 style='mso-width-source:userset;mso-width-alt:3364;width:92pt'>
 <col width=107 style='mso-width-source:userset;mso-width-alt:2925;width:80pt'>
 <col width=73 style='mso-width-source:userset;mso-width-alt:2011;width:55pt'>
 <col width=93 style='mso-width-source:userset;mso-width-alt:2560;width:70pt'>
 <col width=72 style='mso-width-source:userset;mso-width-alt:1974;width:54pt'>
 <tr class=xl65 height=24 style='height:18.0pt'>
  <td height=24 class=xl65 width=99 style='height:18.0pt;width:74pt'></td>
  <td class=xl65 width=79 style='width:59pt'></td>
  <td class=xl65 width=113 style='width:85pt'></td>
  <td class=xl65 width=56 style='width:42pt'></td>
  <td class=xl65 width=52 style='width:39pt'></td>
  <td class=xl65 width=117 style='width:88pt'></td>
  <td class=xl65 width=175 style='width:131pt'></td>
  <td class=xl65 width=123 style='width:92pt'></td>
  <td class=xl65 width=107 style='width:80pt'></td>
  <td class=xl65 width=73 style='width:55pt'></td>
  <td colspan=2 class=xl66 width=165 style='width:124pt'>Speedup</td>
 </tr>
 <tr class=xl66 height=24 style='height:18.0pt'>
  <td height=24 class=xl66 style='height:18.0pt'>Fusion Mode</td>
  <td class=xl66>Batch Size</td>
  <td class=xl66>Input Channels</td>
  <td class=xl66>Height</td>
  <td class=xl66>Width</td>
  <td class=xl66>Conv. Channels</td>
  <td class=xl66>Filter Height and Width</td>
  <td class=xl66>kernel time (ms)</td>
  <td class=xl66>wall time (ms)</td>
  <td class=xl66></td>
  <td class=xl66>kernel time</td>
  <td class=xl66>wall time</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>1024</td>
  <td align=right>14</td>
  <td align=right>14</td>
  <td align=right>256</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.9495</td>
  <td class=xl68 align=right>1.7053</td>
  <td></td>
  <td class=xl68 align=right>1.2182</td>
  <td class=xl68 align=right>1.1880</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>128</td>
  <td align=right>28</td>
  <td align=right>28</td>
  <td align=right>128</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.7927</td>
  <td class=xl68 align=right>0.8974</td>
  <td></td>
  <td class=xl68 align=right>1.2869</td>
  <td class=xl68 align=right>1.3688</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>128</td>
  <td align=right>28</td>
  <td align=right>28</td>
  <td align=right>512</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.0656</td>
  <td class=xl68 align=right>1.6839</td>
  <td></td>
  <td class=xl68 align=right>1.9908</td>
  <td class=xl68 align=right>1.3686</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>2048</td>
  <td align=right>7</td>
  <td align=right>7</td>
  <td align=right>512</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.1559</td>
  <td class=xl68 align=right>1.7141</td>
  <td></td>
  <td class=xl68 align=right>1.0103</td>
  <td class=xl68 align=right>0.7869</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>256</td>
  <td align=right>14</td>
  <td align=right>14</td>
  <td align=right>1024</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.2170</td>
  <td class=xl68 align=right>1.7212</td>
  <td></td>
  <td class=xl68 align=right>1.3705</td>
  <td class=xl68 align=right>0.9296</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>256</td>
  <td align=right>14</td>
  <td align=right>14</td>
  <td align=right>256</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.6397</td>
  <td class=xl68 align=right>0.7055</td>
  <td></td>
  <td class=xl68 align=right>1.1730</td>
  <td class=xl68 align=right>1.2388</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>256</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>64</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.8807</td>
  <td class=xl68 align=right>1.6217</td>
  <td></td>
  <td class=xl68 align=right>1.6530</td>
  <td class=xl68 align=right>0.8581</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>512</td>
  <td align=right>28</td>
  <td align=right>28</td>
  <td align=right>128</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.9987</td>
  <td class=xl68 align=right>1.6292</td>
  <td></td>
  <td class=xl68 align=right>1.2763</td>
  <td class=xl68 align=right>0.8494</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>512</td>
  <td align=right>7</td>
  <td align=right>7</td>
  <td align=right>2048</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.2472</td>
  <td class=xl68 align=right>1.7552</td>
  <td></td>
  <td class=xl68 align=right>1.1523</td>
  <td class=xl68 align=right>0.8553</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>512</td>
  <td align=right>7</td>
  <td align=right>7</td>
  <td align=right>512</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.7313</td>
  <td class=xl68 align=right>0.8112</td>
  <td></td>
  <td class=xl68 align=right>1.0893</td>
  <td class=xl68 align=right>1.1402</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>64</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>256</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.2223</td>
  <td class=xl68 align=right>1.7120</td>
  <td></td>
  <td class=xl68 align=right>2.6976</td>
  <td class=xl68 align=right>1.7321</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>64</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>64</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.3186</td>
  <td class=xl68 align=right>0.3838</td>
  <td></td>
  <td class=xl68 align=right>2.6831</td>
  <td class=xl68 align=right>1.9685</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-B-A</td>
  <td align=right>64</td>
  <td align=right>64</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>64</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.8934</td>
  <td class=xl68 align=right>1.4642</td>
  <td></td>
  <td class=xl68 align=right>1.5497</td>
  <td class=xl68 align=right>0.9451</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 colspan=7 style='height:18.0pt;mso-ignore:colspan'></td>
  <td class=xl67></td>
  <td class=xl67></td>
  <td class=xl65>Average</td>
  <td class=xl68 align=right>1.5501</td>
  <td class=xl68 align=right>1.1715</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>1024</td>
  <td align=right>14</td>
  <td align=right>14</td>
  <td align=right>256</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.9412</td>
  <td class=xl68 align=right>1.3590</td>
  <td></td>
  <td class=xl68 align=right>1.1321</td>
  <td class=xl68 align=right>1.4267</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>128</td>
  <td align=right>28</td>
  <td align=right>28</td>
  <td align=right>128</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.7912</td>
  <td class=xl68 align=right>0.8894</td>
  <td></td>
  <td class=xl68 align=right>1.1345</td>
  <td class=xl68 align=right>1.2481</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>128</td>
  <td align=right>28</td>
  <td align=right>28</td>
  <td align=right>512</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.0475</td>
  <td class=xl68 align=right>1.1856</td>
  <td></td>
  <td class=xl68 align=right>1.4766</td>
  <td class=xl68 align=right>1.8913</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>2048</td>
  <td align=right>7</td>
  <td align=right>7</td>
  <td align=right>512</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.1529</td>
  <td class=xl68 align=right>1.2716</td>
  <td></td>
  <td class=xl68 align=right>0.9554</td>
  <td class=xl68 align=right>1.0590</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>256</td>
  <td align=right>14</td>
  <td align=right>14</td>
  <td align=right>1024</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.2095</td>
  <td class=xl68 align=right>1.3446</td>
  <td></td>
  <td class=xl68 align=right>1.1057</td>
  <td class=xl68 align=right>1.1946</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>256</td>
  <td align=right>14</td>
  <td align=right>14</td>
  <td align=right>256</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.6400</td>
  <td class=xl68 align=right>0.7053</td>
  <td></td>
  <td class=xl68 align=right>1.0633</td>
  <td class=xl68 align=right>1.2330</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>256</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>64</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.8790</td>
  <td class=xl68 align=right>1.0053</td>
  <td></td>
  <td class=xl68 align=right>1.3394</td>
  <td class=xl68 align=right>1.3806</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>512</td>
  <td align=right>28</td>
  <td align=right>28</td>
  <td align=right>128</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.9915</td>
  <td class=xl68 align=right>1.1367</td>
  <td></td>
  <td class=xl68 align=right>1.1437</td>
  <td class=xl68 align=right>1.1935</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>512</td>
  <td align=right>7</td>
  <td align=right>7</td>
  <td align=right>2048</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.2499</td>
  <td class=xl68 align=right>1.3670</td>
  <td></td>
  <td class=xl68 align=right>1.0084</td>
  <td class=xl68 align=right>1.1005</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>512</td>
  <td align=right>7</td>
  <td align=right>7</td>
  <td align=right>512</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.7276</td>
  <td class=xl68 align=right>0.8181</td>
  <td></td>
  <td class=xl68 align=right>1.0173</td>
  <td class=xl68 align=right>1.1338</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>64</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>256</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.2159</td>
  <td class=xl68 align=right>1.3582</td>
  <td></td>
  <td class=xl68 align=right>1.7601</td>
  <td class=xl68 align=right>2.1865</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>64</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>64</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.3188</td>
  <td class=xl68 align=right>0.3746</td>
  <td></td>
  <td class=xl68 align=right>1.8092</td>
  <td class=xl68 align=right>2.0135</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>C-A</td>
  <td align=right>64</td>
  <td align=right>64</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>64</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.8950</td>
  <td class=xl68 align=right>1.0372</td>
  <td></td>
  <td class=xl68 align=right>1.2576</td>
  <td class=xl68 align=right>1.3338</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 colspan=7 style='height:18.0pt;mso-ignore:colspan'></td>
  <td class=xl67></td>
  <td class=xl67></td>
  <td class=xl65>Average</td>
  <td class=xl68 align=right>1.2464</td>
  <td class=xl68 align=right>1.4150</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>1024</td>
  <td align=right>14</td>
  <td align=right>14</td>
  <td align=right>256</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.2690</td>
  <td class=xl68 align=right>0.3220</td>
  <td></td>
  <td class=xl68 align=right>2.3744</td>
  <td class=xl68 align=right>2.5766</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>128</td>
  <td align=right>28</td>
  <td align=right>28</td>
  <td align=right>128</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.1348</td>
  <td class=xl68 align=right>0.1850</td>
  <td></td>
  <td class=xl68 align=right>2.2086</td>
  <td class=xl68 align=right>2.6150</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>128</td>
  <td align=right>28</td>
  <td align=right>28</td>
  <td align=right>512</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.1349</td>
  <td class=xl68 align=right>0.1854</td>
  <td></td>
  <td class=xl68 align=right>2.3359</td>
  <td class=xl68 align=right>2.7634</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>2048</td>
  <td align=right>7</td>
  <td align=right>7</td>
  <td align=right>512</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.1347</td>
  <td class=xl68 align=right>0.1834</td>
  <td></td>
  <td class=xl68 align=right>3.2505</td>
  <td class=xl68 align=right>3.4520</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>256</td>
  <td align=right>14</td>
  <td align=right>14</td>
  <td align=right>1024</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.0723</td>
  <td class=xl68 align=right>0.1217</td>
  <td></td>
  <td class=xl68 align=right>2.0648</td>
  <td class=xl68 align=right>2.8245</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>256</td>
  <td align=right>14</td>
  <td align=right>14</td>
  <td align=right>256</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.0728</td>
  <td class=xl68 align=right>0.1222</td>
  <td></td>
  <td class=xl68 align=right>1.8264</td>
  <td class=xl68 align=right>2.6666</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>256</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>64</td>
  <td align=right>1</td>
  <td class=xl68 align=right>1.0904</td>
  <td class=xl68 align=right>1.2405</td>
  <td></td>
  <td class=xl68 align=right>2.3377</td>
  <td class=xl68 align=right>2.6101</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>512</td>
  <td align=right>28</td>
  <td align=right>28</td>
  <td align=right>128</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.5569</td>
  <td class=xl68 align=right>0.6174</td>
  <td></td>
  <td class=xl68 align=right>2.6798</td>
  <td class=xl68 align=right>4.1437</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>512</td>
  <td align=right>7</td>
  <td align=right>7</td>
  <td align=right>2048</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.0465</td>
  <td class=xl68 align=right>0.0935</td>
  <td></td>
  <td class=xl68 align=right>2.4666</td>
  <td class=xl68 align=right>3.1287</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>512</td>
  <td align=right>7</td>
  <td align=right>7</td>
  <td align=right>512</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.0465</td>
  <td class=xl68 align=right>0.1006</td>
  <td></td>
  <td class=xl68 align=right>2.0905</td>
  <td class=xl68 align=right>2.4140</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>64</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>256</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.2644</td>
  <td class=xl68 align=right>0.3154</td>
  <td></td>
  <td class=xl68 align=right>2.4319</td>
  <td class=xl68 align=right>2.6550</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>64</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>64</td>
  <td align=right>1</td>
  <td class=xl68 align=right>0.2636</td>
  <td class=xl68 align=right>0.3118</td>
  <td></td>
  <td class=xl68 align=right>2.4621</td>
  <td class=xl68 align=right>2.6778</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 style='height:18.0pt'>N-A</td>
  <td align=right>64</td>
  <td align=right>64</td>
  <td align=right>55</td>
  <td align=right>55</td>
  <td align=right>64</td>
  <td align=right>3</td>
  <td class=xl68 align=right>0.2645</td>
  <td class=xl68 align=right>0.3154</td>
  <td></td>
  <td class=xl68 align=right>2.3763</td>
  <td class=xl68 align=right>2.5929</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 colspan=9 style='height:18.0pt;mso-ignore:colspan'></td>
  <td class=xl65>Average</td>
  <td class=xl68 align=right>2.3773</td>
  <td class=xl68 align=right>2.8554</td>
 </tr>
</table>

