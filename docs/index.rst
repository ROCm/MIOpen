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
    * :doc:`Build the driver <install/driver>`
    * :doc:`Build MIOpen for embedded systems <install/embed>`
  
  .. grid-item-card:: Conceptual

    * :doc:`Getting started with Fusion API <conceptual/Getting_Started_FusionAPI>`
    * :doc:`MI200 alternate implementation <conceptual/MI200AlternateImplementation>`
    * :doc:`Cache <conceptual/cache>`
    * :doc:`Find and immediate <conceptual/find_and_immediate>`
    * :doc:`Find database <conceptual/finddb>`
    * :doc:`Performance database <conceptual/perfdatabase>`
    * :doc:`MIOpen Porting Guide <conceptual/MIOpen_Porting_Guide>`
    
  
  ..  grid-item-card:: API reference 

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
