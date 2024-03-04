Compile a Fusion Plan
-------------------------

Following the operator addition, the user would compile the fusion plan, to populate the MIOpen kernel cache with the fused kernel and make it ready for execution. 
The API call that accomplishes this is:

.. code-block:: 

    miopenStatus_t
    miopenCompileFusionPlan(miopenHandle_t handle, miopenFusionPlanDescriptor_t fusePlanDesc);

The corresponding code snippet in the example is as follows:

.. code-block:: 

    auto status = miopenCompileFusionPlan(mio::handle(), fusePlanDesc);
    if (status != miopenStatusSuccess) {
    return -1;
    }

In order to compile the fusion plan, the user is assumed to have acquired an MIOpen handle object, in the example code above this is accomplished using the mio::handle() helper function. While a fusion plan itself is not bound to a MIOpen handle object, it would however need to be recompiled for each handle separately. It may be noted that compilation of a fusion plan might fail for a number of reasons, moreover it is not assured that a fused version of the kernel would offer any performance improvement over the separately run kernels.

Compiling a fusion plan is a costly operation in terms of run-time. Therefore, it is recommended that a fusion plan should only be compiled once and may be reused for execution with different runtime parameters as described in the next section.
