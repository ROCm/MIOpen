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


| ﻿            |            |                |        |       |                |                         |                  |                | Speedup     | Speedup   |
|-------------|------------|----------------|--------|-------|----------------|-------------------------|------------------|----------------|-------------|-----------|
| Fusion Mode | Batch Size | Input Channels | Height | Width | Conv. Channels | Filter Height and Width | kernel time (ms) | wall time (ms) | kernel time | wall time |
| C-B-A       | 64         | 1024           | 14     | 14    | 256            | 1                       | 0.9495           | 1.7053         | 1.2182      | 1.1880    |
| C-B-A       | 64         | 128            | 28     | 28    | 128            | 3                       | 0.7927           | 0.8974         | 1.2869      | 1.3688    |
| C-B-A       | 64         | 128            | 28     | 28    | 512            | 1                       | 1.0656           | 1.6839         | 1.9908      | 1.3686    |
| C-B-A       | 64         | 2048           | 7      | 7     | 512            | 1                       | 1.1559           | 1.7141         | 1.0103      | 0.7869    |
| C-B-A       | 64         | 256            | 14     | 14    | 1024           | 1                       | 1.2170           | 1.7212         | 1.3705      | 0.9296    |
| C-B-A       | 64         | 256            | 14     | 14    | 256            | 3                       | 0.6397           | 0.7055         | 1.1730      | 1.2388    |
| C-B-A       | 64         | 256            | 55     | 55    | 64             | 1                       | 0.8807           | 1.6217         | 1.6530      | 0.8581    |
| C-B-A       | 64         | 512            | 28     | 28    | 128            | 1                       | 0.9987           | 1.6292         | 1.2763      | 0.8494    |
| C-B-A       | 64         | 512            | 7      | 7     | 2048           | 1                       | 1.2472           | 1.7552         | 1.1523      | 0.8553    |
| C-B-A       | 64         | 512            | 7      | 7     | 512            | 3                       | 0.7313           | 0.8112         | 1.0893      | 1.1402    |
| C-B-A       | 64         | 64             | 55     | 55    | 256            | 1                       | 1.2223           | 1.7120         | 2.6976      | 1.7321    |
| C-B-A       | 64         | 64             | 55     | 55    | 64             | 1                       | 0.3186           | 0.3838         | 2.6831      | 1.9685    |
| C-B-A       | 64         | 64             | 55     | 55    | 64             | 3                       | 0.8934           | 1.4642         | 1.5497      | 0.9451    |
|             |            |                |        |       |                |                         |                  | Average        | 1.5501      | 1.1715    |
| C-A         | 64         | 1024           | 14     | 14    | 256            | 1                       | 0.9412           | 1.3590         | 1.1321      | 1.4267    |
| C-A         | 64         | 128            | 28     | 28    | 128            | 3                       | 0.7912           | 0.8894         | 1.1345      | 1.2481    |
| C-A         | 64         | 128            | 28     | 28    | 512            | 1                       | 1.0475           | 1.1856         | 1.4766      | 1.8913    |
| C-A         | 64         | 2048           | 7      | 7     | 512            | 1                       | 1.1529           | 1.2716         | 0.9554      | 1.0590    |
| C-A         | 64         | 256            | 14     | 14    | 1024           | 1                       | 1.2095           | 1.3446         | 1.1057      | 1.1946    |
| C-A         | 64         | 256            | 14     | 14    | 256            | 3                       | 0.6400           | 0.7053         | 1.0633      | 1.2330    |
| C-A         | 64         | 256            | 55     | 55    | 64             | 1                       | 0.8790           | 1.0053         | 1.3394      | 1.3806    |
| C-A         | 64         | 512            | 28     | 28    | 128            | 1                       | 0.9915           | 1.1367         | 1.1437      | 1.1935    |
| C-A         | 64         | 512            | 7      | 7     | 2048           | 1                       | 1.2499           | 1.3670         | 1.0084      | 1.1005    |
| C-A         | 64         | 512            | 7      | 7     | 512            | 3                       | 0.7276           | 0.8181         | 1.0173      | 1.1338    |
| C-A         | 64         | 64             | 55     | 55    | 256            | 1                       | 1.2159           | 1.3582         | 1.7601      | 2.1865    |
| C-A         | 64         | 64             | 55     | 55    | 64             | 1                       | 0.3188           | 0.3746         | 1.8092      | 2.0135    |
| C-A         | 64         | 64             | 55     | 55    | 64             | 3                       | 0.8950           | 1.0372         | 1.2576      | 1.3338    |
|             |            |                |        |       |                |                         |                  | Average        | 1.2464      | 1.4150    |
| N-A         | 64         | 1024           | 14     | 14    | 256            | 1                       | 0.2690           | 0.3220         | 2.3744      | 2.5766    |
| N-A         | 64         | 128            | 28     | 28    | 128            | 3                       | 0.1348           | 0.1850         | 2.2086      | 2.6150    |
| N-A         | 64         | 128            | 28     | 28    | 512            | 1                       | 0.1349           | 0.1854         | 2.3359      | 2.7634    |
| N-A         | 64         | 2048           | 7      | 7     | 512            | 1                       | 0.1347           | 0.1834         | 3.2505      | 3.4520    |
| N-A         | 64         | 256            | 14     | 14    | 1024           | 1                       | 0.0723           | 0.1217         | 2.0648      | 2.8245    |
| N-A         | 64         | 256            | 14     | 14    | 256            | 3                       | 0.0728           | 0.1222         | 1.8264      | 2.6666    |
| N-A         | 64         | 256            | 55     | 55    | 64             | 1                       | 1.0904           | 1.2405         | 2.3377      | 2.6101    |
| N-A         | 64         | 512            | 28     | 28    | 128            | 1                       | 0.5569           | 0.6174         | 2.6798      | 4.1437    |
| N-A         | 64         | 512            | 7      | 7     | 2048           | 1                       | 0.0465           | 0.0935         | 2.4666      | 3.1287    |
| N-A         | 64         | 512            | 7      | 7     | 512            | 3                       | 0.0465           | 0.1006         | 2.0905      | 2.4140    |
| N-A         | 64         | 64             | 55     | 55    | 256            | 1                       | 0.2644           | 0.3154         | 2.4319      | 2.6550    |
| N-A         | 64         | 64             | 55     | 55    | 64             | 1                       | 0.2636           | 0.3118         | 2.4621      | 2.6778    |
| N-A         | 64         | 64             | 55     | 55    | 64             | 3                       | 0.2645           | 0.3154         | 2.3763      | 2.5929    |
|             |            |                |        |       |                |                         |                  | Average        | 2.3773      | 2.8554    |
