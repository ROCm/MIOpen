
Create a Fusion plan
----------------------
A **Fusion Plan** is the data structure which holds all the metadata about the users fusion intent as well as logic to **Compile** and **Execute** a fusion plan. As mentioned earlier, a fusion plan holds the order in which different opertions would be applied on the data, but it also specifies the _axis_ of fusion as well. Currently only **vertical** (sequential) fusions are supported implying the flow of data between operations is sequential.

A fusion plan is created using the API call `miopenCreateFusionPlan` with the signature:

.. code-block:: 
        
            cpp
        miopenStatus_t
        miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
        const miopenFusionDirection_t fuseDirection,const miopenTensorDescriptor_t inputDesc);


The *input tensor descriptor* specifies the geometry of the incoming data. Since the data geometry of the intermediate operations can be derived from the *input tensor descriptor*, therefore only the *input tensor descriptor* is required for the fusion plan and not for the individual operations. In our fusion example the following lines of code accomplish this:

  .. code-block:: 
  
       cpp
 miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, input.desc);

Where `fusePlanDesc` is an object of type `miopenFusionPlanDescriptor_t` and `input.desc` is the `miopenTensorDescriptor_t` object.
