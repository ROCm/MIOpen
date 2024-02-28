Set runtime arguments
-----------------------------

While the underlying MIOpen descriptor of the fusion operator specifies the data geometry and parameters, the fusion plan still needs access to the data to execute a successfully compiled fusion plan. The arguments mechanism in the Fusion API provides such data before a fusion plan may be executed. For example the convolution operator requires weights to carry out the convolution computation, a bias operator requires the actual bias values etc. Therefore, before a fusion plan may be executed, arguments required by each fusion operator need to be specified. To begin, we create the miopenOperatorArgs_t object using:

.. code-block:: 

    miopenStatus_t miopenCreateOperatorArgs(miopenOperatorArgs_t* args);

Once created, runtime arguments for each operation may be set. In our running example, the forward convolution operator requires the convolution weights argument which is supplied using the API call:

.. code-block:: 

    miopenStatus_t
    miopenSetOpArgsConvForward(miopenOperatorArgs_t args,
                               const miopenFusionOpDescriptor_t convOp,
                               const void* alpha,
                               const void* beta,
                               const void* w);

Similarly the parameters for bias and activation are given by:

.. code-block:: 

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


In our example code, we set the arguments for the operations as follows:

.. code-block:: 

    miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, weights.data);
    miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta, activ_alpha,
                              activ_beta, activ_gamma);
    miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, bias.data);


This separation between the fusion plan and the arguments required by each operator allows better reuse of the fusion plan with different arguments as well as avoids the necessity of recompiling the fusion plan to run the same combination of operators with different arguments.

As mentioned in the section Compile the Fusion Plan earlier, the compilation step for a fusion plan might be costly, therefore a fusion plan should only be compiled once in its lifetime. A fusion plan needs not be recompiled if the input desciptor or any of the parameters to the miopenCreateOp* API calls are different, otherwise a compiled fusion plan may be reused again and again with a different set of arguments. In our example this is demonstrated in lines 77 - 85 of main.cpp.
