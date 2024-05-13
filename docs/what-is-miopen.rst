
.. meta::
  :description: What is MIOpen?
  :keywords: MIOpen, ROCm, API, documentation

*************************************************************
What is MIOpen?
*************************************************************

MIOpen is AMD's open-source, deep-learning primitives library for GPUs. It implements fusion to
optimize for memory bandwidth and GPU launch overheads, providing an auto-tuning infrastructure
to overcome the large design space of problem configurations. It also implements different algorithms
to optimize convolutions for different filter and input sizes.

MIOpen is one of the first libraries to publicly support the bfloat16 datatype for convolutions, which
allows for efficient training at lower precision without loss of accuracy.
