.. meta::
  :description: Using the Fusion API
  :keywords: MIOpen, ROCm, API, documentation, fusion API

************************************************************************************************
Using the fusion API
************************************************************************************************

Increasing the depth of deep-learning networks requires novel mechanisms to improve GPU
performance. One mechanism to achieve higher efficiency is to *fuse* separate kernels into a single
kernel in order to reduce off-chip memory access and avoid kernel launch overhead.

Using MIOpen's fusion API, you can specify operators that you want to fuse into a single kernel,
compile that kernel, and then launch it. While not all combinations are supported, the API is flexible
enough to allow the specification of several operations, in any order, from the set of supported
operations. The API provides a mechanism to report unsupported combinations.

You can find a complete example of MIOpen's fusion API in the MIOpen GitHub repository
`example folder <https://github.com/ROCm/MIOpenExamples/tree/master/fusion>`_. The code
examples in this document are taken from this example project.

.. note::
  The example project creates a fusion plan to merge the convolution, bias, and activation operations.
  For a list of supported fusion operations and associated constraints, see the
  :ref:`Supported fusions <supported-fusions>` section. For simplicity, the example doesn't populate
  the tensors with meaningful data and only shows the basic code without any error checking.

After you've initialized an MIOpen handle object, the workflow for using the fusion API is:

* Create a fusion plan
* Create and add the convolution, bias, and activation operators
* Compile the fusion plan
* Set the runtime arguments for each operator
* Run the fusion plan
* Cleanup

The order in which you create operators is important because this order represents the order of
operations for the data. Therefore, a fusion plan where convolution is created before activation differs
from a fusion plan where activation is added before convolution.

.. note::
  The primary consumers of the fusion API are high-level frameworks, such as TensorFlow/XLA and
  PyTorch.

Creating a fusion plan
=================================================

A *fusion plan* is the data structure that holds all the metadata regarding fusion intent along with the
logic to compile and run a fusion plan. The fusion plan not only contains the order in which different
operations are applied on the data, but also specifies the *axis* of fusion. Currently, only *vertical*
(sequential) fusions are supported, implying the flow of data between operations is sequential.

You can create a fusion plan using ``miopenCreateFusionPlan``, as follows:

.. code:: cpp

  miopenStatus_t
  miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
  const miopenFusionDirection_t fuseDirection,const miopenTensorDescriptor_t inputDesc);

The *input tensor descriptor* specifies the geometry of the incoming data. Because the data geometry
of the intermediate operations can be derived from the input tensor descriptor, this is only required for
the fusion plan. The input tensor descriptor isn't required for the individual operations.

.. code:: cpp

  miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, input.desc);

In the previous example, ``fusePlanDesc`` is an object of type ``miopenFusionPlanDescriptor_t`` and ``input.desc`` is
the ``miopenTensorDescriptor_t`` object.

Creating and adding operators
=================================================

Operators represent the different operations to fuse. Currently, the API supports these
operators:

* Convolution forward
* Activation forward
* BatchNorm inference
* Bias forward

.. note::

  Although bias is a separate operator, it's typically only used with convolution.

MIOpen plans to add support for more operators, including operators for backward passes, in the future.

The fusion API provides calls for the creation of the supported operators. To learn more, refer to the
:doc:`Fusion <../doxygen/html/group___f_u_s_i_o_n>` API documentation.

After you've created the fusion plan descriptor, you can add two or more operators to it by using the
individual operator creation API calls. If the API doesn't support the fusion of the operations you add,
the creation might fail.

This example adds the convolution, bias, and activation operations to the newly created fusion
plan.

.. code:: cpp

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


``conv_desc`` is the regular MIOpen convolution descriptor. For more information on creating and
setting this descriptor, see the example code and the
:doc:`Convolution <../doxygen/html/group__convolutions>` API documentation.

``weights.desc`` refers to ``miopenTensorDescriptor_t`` for the convolution operations.
``bias.desc`` refers to the object of the same type for the bias operation.

In the preceding code, the convolution operation is the first operation to run on the incoming data,
followed by the bias, and then activation operations.

During this process, it is important to verify the return codes to ensure the operations and
sequence are supported. The operator insertion can fail for a number of reasons, such as an unsupported
operation sequence, unsupported input dimensions, or, in the case of convolution, unsupported filter
dimensions. In the preceding example, these aspects are ignored for the sake of simplicity.

Compiling the fusion plan
=================================================

Following the addition of the operators, you can compile the fusion plan. This populates the MIOpen kernel
cache with the fused kernel and gets it ready to run.

.. code:: cpp

  miopenStatus_t
  miopenCompileFusionPlan(miopenHandle_t handle, miopenFusionPlanDescriptor_t fusePlanDesc);


The corresponding code snippet in the example is:

.. code:: cpp

  auto status = miopenCompileFusionPlan(mio::handle(), fusePlanDesc);
  if (status != miopenStatusSuccess) {
  return -1;
  }

To compile the fusion plan, you must acquire an MIOpen handle object. In the
preceding code, this is accomplished using the ``mio::handle()`` helper function. While a fusion plan is
itself not bound to an MIOpen handle object, it must be recompiled separately for each handle.

Compiling a fusion plan is a costly operation in terms of run-time, and compilation can fail for a
number of reasons. Therefore, the recommendation is to only compile your fusion plan once and reuse it
with different runtime parameters, as described in the next section.

Setting runtime arguments
=================================================

While the fusion operator for the underlying MIOpen descriptor specifies the data geometry and
parameters, the fusion plan still needs access to the data to run a successfully compiled fusion plan.
The arguments mechanism in the fusion API provides this data before a fusion plan can be run. For
example, the convolution operator requires *weights* to carry out the convolution computation, and the
bias operator requires the actual bias values. Therefore, before you can run a fusion plan, you must
specify the arguments required by each fusion operator.

First create the ``miopenOperatorArgs_t`` object using this code:

.. code:: cpp

  miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args);

After it is created, you can set the runtime arguments for each operation. In this example, the forward
convolution operator requires the convolution weights argument, which is supplied using:

.. code:: cpp

  miopenStatus_t
  miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
                            const miopenFusionOpDescriptor_t convOp,
                            const void* alpha,
                            const void* beta,
                            const void* w);

Similarly, the parameters for bias and activation are supplied by:

.. code:: cpp

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

In the example code, the arguments for the operations are set as follows:

.. code:: cpp

  miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, weights.data);
  miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta, activ_alpha,
                            activ_beta, activ_gamma);
  miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, bias.data);

Having a separation between the fusion plan and the arguments required by each operator allows better
reuse of the fusion plan with different arguments. It also avoids the necessity to recompile the fusion
plan to run the same combination of operators with different arguments.

As previously mentioned, the compilation step for a fusion plan can be costly. Therefore, it is
recommended that you only compile a fusion plan once in its lifetime. A fusion plan doesn't need to be
recompiled if the input descriptor or any of the parameters in the ``miopenCreateOp*`` API calls are
different. You can repeatedly reuse a compiled fusion plan with a different set of arguments.

In the example, this is demonstrated in ``main.cpp``, lines 77 through 85.

Running a fusion plan
========================================================

Once you've compiled the fusion plan and set the arguments for each operator, you can run it as
follows:

.. code:: cpp

  miopenStatus_t
  miopenExecuteFusionPlan(const miopenHandle_t handle,
                          const miopenFusionPlanDescriptor_t fusePlanDesc,
                          const miopenTensorDescriptor_t inputDesc,
                          const void* input,
                          const miopenTensorDescriptor_t outputDesc,
                          void* output,
                          miopenOperatorArgs_t args);

The following code snippet runs the fusion plan:

.. code:: cpp

  miopenExecuteFusionPlan(mio::handle(), fusePlanDesc, input.desc, input.data,
                          output.desc, output.data, fusionArgs);

If you try to run a fusion plan that is not compiled, or has been invalidated by changing the input
tensor descriptor or any of the operation parameters, you'll get an error.

Cleanup
=================================================

After the application is finished with the fusion plan, you can destroy the fusion plan and the fusion ``args``
objects:

.. code:: cpp

  miopenStatus_t miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc);

After the fusion plan object is destroyed, all the operations are automatically destroyed. You don't
need to worry about additional cleanup.

.. _supported-fusions:

Supported fusions
=================================================

The following tables outline the supported fusions for ``FP32``, ``FP16``, and ``BFP16``, including any applicable
constraints.

.. note::

   Fusion Plans with grouped convolutions are supported in the inference direction for
   convolution, bias, and activation.

The following abbreviations apply to the combination column in the following tables:

*  **C**: Convolution
*  **B**: Bias
*  **N**: Batch Normalization
*  **A**: Activation

For example, CBA refers to convolution plus bias plus activation.

Convolution-based FP32 fusion for inference
-------------------------------------------

The following table applies to single-precision floating point.

.. csv-table::
   :header: "Combination","Conv algo","Stride","Filter dims","N mode","Activations","Other constraints"
   :widths: 15, 15, 15, 20, 12, 20, 20

   "CBNA","Direct","1 and 2","3x3, 5x5, 7x7, 9x9, 11x11","All","All","stride and padding must be either 1 or 2"
   "CBA","Direct","--","1x1","--","All","stride and padding not supported"
   "CBA","Winograd","1","1x1, 2x2","N/A","Relu, Leaky Relu","c >= 18"
   "CBA","Winograd","1","3x3","--","Relu, Leaky Relu","c >= 18 and c is even"
   "CBA","Winograd","1","4x4, 5x5, 6x6","--","Relu, Leaky Relu","4 x c >= 18"
   "CBA","Winograd","1","7x7, 8x8, 9x9","--","Relu, Leaky Relu","12 x c >= 18"
   "CBA","Winograd","1","10x10, 11x11, 12x12","--","Relu, Leaky Relu","16 x c >= 18"
   "CBA","Winograd","1","larger filter sizes","--","Relu, Leaky Relu","none"
   "CBA","Winograd","2","1x1","--","Relu, Leaky Relu","2 x c >= 18"
   "CBA","Winograd","2","2x2, 3x3, 4x4, 5x5, 6x6","--","Relu, Leaky Relu","4 x c >= 18"
   "CBA","Winograd","2","7x7","--","Relu, Leaky Relu","12 x c >= 18"
   "CBA","Winograd","2","8x8, 9x9, 10x10, 11x11, 12x12","--","Relu, Leaky Relu","16 x c >= 18"
   "CBA","Winograd","2","larger filter sizes","--","Relu, Leaky Relu","none"
   "CBA","CK","--","--","--","Relu, Clipped Relu, CLAMP","none"
   "NA","--","--","--","All","All","padding not supported"
   "CA","Direct","--","1x1","--","All","stride and padding not supported"
   "CA","CK","--","--","--","Relu, Clipped Relu, CLAMP","none"

.. note::

   N mode is either spatial or per activation. For CBA, other asymmetric kernels are supported but for brevity are not enumerated here.


Convolution-based FP16 fusion for inference
-------------------------------------------

The following table applies to half-precision floating point.

.. csv-table::
   :header: "Combination","Conv algo","Stride","Filter dims","N mode","Activations","Other constraints"
   :widths: 15, 15, 15, 20, 12, 20, 20

   "CBNA","Direct","1 and 2","3x3, 5x5, 7x7, 9x9, 11x11","All","All","stride and padding must be either 1 or 2"
   "CBA","Direct","--","1x1","--","All","stride and padding not supported"
   "CBA","CK","--","--","--","Relu, Clipped Relu, CLAMP","none"
   "CA","Direct","--","1x1","--","All","stride and padding not supported"
   "CA","CK","--","--","--","Relu, Clipped Relu, CLAMP","none"

.. note::

   N mode is either spatial or per activation.


Convolution-based BFP16 fusion for inference
--------------------------------------------

The following table applies to half-precision block floating point.

.. csv-table::
   :header: "Combination","Conv algo","Stride","Filter dims","N mode","Activations","Other constraints"
   :widths: 15, 15, 15, 20, 12, 20, 20

   "CBNA","Direct","1 and 2","3x3, 5x5, 7x7, 9x9, 11x11","All","All","stride and padding must be either 1 or 2"
   "CBA","Direct","--","1x1","--","All","stride and padding not supported"
   "CBA","CK","--","--","--","Relu, Clipped Relu, CLAMP","none"
   "CA","Direct","--","1x1","--","All","stride and padding not supported"
   "CA","CK","--","--","--","Relu, Clipped Relu, CLAMP","none"

.. note::

   N mode is either spatial or per activation.

Batch Normalization-based fusion for FP32, BFP16, and FP16 for inference and training
-------------------------------------------------------------------------------------

The following table applies to both full-precision and half-precision floating point.

.. csv-table::
   :header: "Combination","N mode","Activations","Constraints"
   :widths: 30, 15, 15, 15

   "NA for inference","All","All","None"
   "NA forward training","All","All","None"
   "NA backward training","All","All","None"

.. note::

   N mode is either spatial or per activation.

Comparing performance with non-fused kernels
=================================================

The following graph depicts the speedup gained for a fused convolution+bias+activation over a
non-fused version. All configurations have a batch size of 64:

.. image:: ../data/how-to/cba.png
  :width: 800
  :alt: convolution-bias-activation graph

The following graph depicts the speedup obtained by fusing BatchNorm (in spatial mode) with activation:

.. image:: ../data/how-to/bn_activ_fused.png
  :width: 800
  :alt: BatchNorm activation fusion
