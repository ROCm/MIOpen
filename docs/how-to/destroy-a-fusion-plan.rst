
Destroy a Fusion plan
----------------------

Once the application is done with the fusion plan, the fusion plan and the fusion args objects may be destroyed using the API calls:

.. code-block:: 
  
    miopenStatus_t miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc);

Once the fusion plan object is destroyed, all the operations created are destroyed automatically and do not need any special cleanup.
