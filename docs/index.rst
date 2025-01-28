.. meta::
  :description: Documentation for MIOpen,
  :keywords: MIOpen, ROCm, API, documentation

********************************************************************
MIOpen documentation
********************************************************************

MIOpen is the AMD open-source, deep-learning primitives library for GPUs. It implements fusion to
optimize for memory bandwidth and GPU launch overheads, providing an auto-tuning infrastructure
to overcome the large design space of problem configurations. It also implements different algorithms
to optimize convolutions for different filter and input sizes.

MIOpen is one of the first libraries to publicly support the ``bfloat16`` data type for convolutions, which
allows for efficient training at lower precision without loss of accuracy.

The MIOpen public repository is located at `<https://github.com/ROCm/MIOpen>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`MIOpen prerequisites <./install/prerequisites>`
    * :doc:`Install MIOpen <./install/install>`
    * :doc:`Build MIOpen from source <./install/build-source>`
    * :doc:`Build MIOpen for embedded systems <./install/embed>`
    * :doc:`Build MIOpen using Docker <./install/docker-build>`
  
  .. grid-item-card:: Conceptual

    * :doc:`Find database <./conceptual/finddb>`
    * :doc:`Kernel cache <./conceptual/cache>`
    * :doc:`Performance database <./conceptual/perfdb>`
    * :doc:`MI200 alternate implementation <./conceptual/MI200-alt-implementation>`
    * :doc:`Porting to MIOpen <./conceptual/porting-guide>`

  .. grid-item-card:: How to

    * :doc:`Use the fusion API <./how-to/use-fusion-api>`
    * :doc:`Log and debug <./how-to/debug-log>`
    * :doc:`Use the find APIs and immediate mode <./how-to/find-and-immediate>`

  ..  grid-item-card:: Reference

    * :doc:`API library <reference/index>`

      * :doc:`Modules <./doxygen/html/modules>`
      * :doc:`Datatypes <reference/datatypes>`

For information on contributing to the MIOpen code base, see
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

For licensing information for all ROCm components, see
`ROCm licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_.
