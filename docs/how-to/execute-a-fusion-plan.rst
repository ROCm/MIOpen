Execute a Fusion Plan
-----------------------
  
Once the fusion plan has been compiled and arguments set for each operator, it may be executed with the API call given below passing it the actual data to be processed.

.. code-block:: 

    miopenStatus_t
    miopenExecuteFusionPlan(const miopenHandle_t handle,
                            const miopenFusionPlanDescriptor_t fusePlanDesc,
                            const miopenTensorDescriptor_t inputDesc,
                            const void* input,
                            const miopenTensorDescriptor_t outputDesc,
                            void* output,
                            miopenOperatorArgs_t args);

The following code snippet in the example accomplishes the fusion plan execution:

.. code-block:: 

    miopenExecuteFusionPlan(mio::handle(), fusePlanDesc, input.desc, input.data,
                        output.desc, output.data, fusionArgs);

It may be noted that it is an error to attempt to execute a fusion plan that is either not compiled or has been invalidated by changing the input tensor descriptor or any of the operation parameters.
