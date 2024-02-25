.. meta::
  :description: MIOpen documentation and API reference library
  :keywords: MIOpen, ROCm, API, documentation

.. _MIOpen-docs-home:

********************************************************************
MIOpen documentation
********************************************************************

MIOpen, AMD's open-source deep learning primitives library for GPUs, provides highly optimized implementations of such operators, shielding researchers from internal implementation details and hence, accelerating the time to discovery.

MIOpen innovates on several fronts, such as implementing fusion to optimize for memory bandwidth and GPU launch overheads, providing an auto-tuning infrastructure to overcome the large design space of problem configurations, and implementing different algorithms to optimize convolutions for different filter and input sizes. MIOpen is one of the first libraries to publicly support the bfloat16 data-type for convolutions, allowing efficient training at lower precision without the loss of accuracy.

Our documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :doc:`Installation <install/install>`
    * `Build the driver <https://github.com/ROCm/rocDecode/blob/master/docs/install/driver.html>`_
    * `Embedd the driver <https://github.com/ROCm/rocDecode/blob/master/docs/install/embed.html>`_
  
  .. grid-item-card:: Concepts

    * `Getting started with Fusion API <https://github.com/ROCm/rocDecode/blob/master/docs/conceptual/Getting_Started_FusionAPI.html>`_
    * `MI200 alternate implementation <https://github.com/ROCm/rocDecode/blob/master/docs/conceptual/MI200AlternateImplementation.html>`_
    * `Cache <https://github.com/ROCm/rocDecode/blob/master/docs/conceptual/cache.html>`_
    * `Find and immediate <https://github.com/ROCm/rocDecode/blob/master/docs/conceptual/find_and_immediate.html>`_
    * `Find database <https://github.com/ROCm/rocDecode/blob/master/docs/conceptual/finddb.html>`_
    * `Performance database <https://github.com/ROCm/rocDecode/blob/master/docs/conceptual/perfdatabase.html>`_
    * `MIOpen Porting Guide <https://github.com/ROCm/rocDecode/blob/master/docs/conceptual/MIOpen_Porting_Guide.html>`_
  
  ..  grid-item-card:: API Reference 

    * `Datatypes <https://rocm.docs.amd.com/projects/MIOpen/en/latest/datatypes.html>`_
    * `Handle <https://rocm.docs.amd.com/projects/MIOpen/en/latest/handle.html>`_
    * `Tensors <https://rocm.docs.amd.com/projects/MIOpen/en/latest/tensors.html>`_
    * `Activation <https://rocm.docs.amd.com/projects/MIOpen/en/latest/activation.html>`_
    * `Convolution <https://rocm.docs.amd.com/projects/MIOpen/en/latest/convolution.html>`_
    * `Recurrent neural networks <https://rocm.docs.amd.com/projects/MIOpen/en/latest/rnn.html>`_
    * `Batch normalization layer <https://rocm.docs.amd.com/projects/MIOpen/en/latest/batchnorm.html>`_
    * `Local response normalization layer <https://rocm.docs.amd.com/projects/MIOpen/en/latest/lrn.html>`_
    * `Pooling layer <https://rocm.docs.amd.com/projects/MIOpen/en/latest/pooling.html>`_
    * `Softmax layer <https://rocm.docs.amd.com/projects/MIOpen/en/latest/softmax.html>`_
    * `Layer fusion <https://rocm.docs.amd.com/projects/MIOpen/en/latest/fusion.html>`_
    * `Loss function layer <https://rocm.docs.amd.com/projects/MIOpen/en/latest/loss.html>`_
    * `Dropour layer <https://rocm.docs.amd.com/projects/MIOpen/en/latest/dropout.html>`_
    * `Reduction layer <https://rocm.docs.amd.com/projects/MIOpen/en/latest/reduction.html>`_
