.. meta::
  :description: Using NHWC Batchnorm in PyTorch
  :keywords: MIOpen, ROCm, API, documentation, NHWC Batchnorm, PyTorch

************************************************************************************************
Using NHWC Batchnorm in PyTorch
************************************************************************************************

This topic explains how to use NHWC Batchnorm for MIOpen operations in PyTorch. NHWC is
a deep learning memory format that has certain performance advantages over traditional
memory formats.

NHWC versus NCHW
=================================================

NHWC (also known as "Channels Last") and NCHW are two types of memory formats for deep learning. They describe how
multidimensional arrays (nD) are translated to a linear (1-dimensional) memory address space.

*  NCHW (Number of samples, channels, height, width): This is the default data layout in which channels
   are stored separately from one another. The height and width information is stored after
   the channels.
*  NHWC (Number of samples, height, width, channels): In this alternative format, channels are stored next
   to each other after the height and width information.

The performance of NHWC is better than NCHW and is close to that observed when using blocked memory format. NHWC is also
easier to work with for common operations.

For more information about these memory formats, see the
`PyTorch memory format documentation <https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_
and the `Intel® Extension for PyTorch GitHub <https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/nhwc.html>`_.

Batchnorm
=================================================

Batchnorm (Batch Normalization) enables higher learning rates and reduces initialization overhead by
normalizing layer inputs. Ordinarily, the distribution of the inputs to each layer changes as the
parameters to the previous layer change. This makes it more difficult to train deep learning models
and leads to lower learning rates. With Batchnorm, normalization is part of the architecture
and is performed for each training batch.

For more information on Batchnorm, see `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_.

Enabling NHWC Batchnorm for MIOpen using PyTorch
=================================================

The PyTorch open-source tensor library provides support for using NHWC Batchnorm with MIOpen.
In addition to Batchnorm, NHWC support is also available for convolution and other MIOpen features.

By default, NHWC Batchnorm support in MIOpen is disabled.
However, it can be enabled in a PyTorch environment using ROCm version 6.5 or later.
This configuration only supports 2D NHWC Batchnorm, with 3D NHWC Batchnorm planned for a future release.
1D Batchnorm is not applicable to the NHWC format.
To enable NHWC Batchnorm, use this command:

.. code:: shell

   PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM=1

.. note::

   In a future release, NHWC Batchnorm will be enabled by default and this command will be deprecated and
   removed.

PyTorch branch support
------------------------

Only the ``ROCm/pytorch`` PyTorch images support NHWC Batchnorm. ROCm version 6.5 or later is required.
The ``upstream`` images do not support this feature. The following PyTorch branches support the
NHWC Batchnorm feature:

*  `release/2.6 <https://github.com/ROCm/pytorch/tree/release/2.6>`_
*  `release/2.7 <https://github.com/ROCm/pytorch/tree/release/2.7>`_: Not yet available
*  `rocm6.4_internal_testing <https://github.com/ROCm/pytorch/tree/rocm6.4_internal_testing>`_
*  `rocm6.5_internal_testing <https://github.com/ROCm/pytorch/tree/rocm6.5_internal_testing>`_: Not yet available

For information on installing and using PyTorch on ROCm, see :doc:`PyTorch on ROCm <rocm-install-on-linux:install/3rd-party/pytorch-install>`.

Supported configurations
=================================================

The following table shows the Batchnorm support for NHWC and NCHW with various data types and modes.
It also indicates which backend is used with and without the ``PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM``
environment variable enabled.

.. csv-table::
   :header: "Input data type","Memory format","Mode","Default backend","Backend with variable enabled"
   :widths: 20, 20, 20, 20, 25

   "``float32``","NCHW","1D/2D","MIOpen","MIOpen"
   "``float32``","NHWC","2D","native","MIOpen"
   "``float16``","NCHW","1D/2D mixed","MIOpen","MIOpen"
   "``float16``","NCHW","1D/2D not mixed","native","native"
   "``float16``","NHWC","2D mixed","native","MIOpen"
   "``float16``","NHWC","2D not mixed","native","native"
   "``bfloat16``","NCHW","1D/2D mixed","MIOpen (*)","MIOpen (*)"
   "``bfloat16``","NCHW","1D/2D not mixed","native","native"
   "``bfloat16``","NHWC","2D mixed","native","MIOpen"
   "``bfloat16``","NHWC","2D not mixed","native","native"
   "any","any","3D","native","native"

(*) MIOpen is used with ROCm 6.5 and later. Otherwise, the native backend is used.

Verifying NHWC Batchnorm use with MIOpen
=================================================

For some operations, it can be difficult to determine the backend and memory format that were used.
To verify whether MIOpen is being used and whether the memory format is NHWC or NCHW, run your program
with the following environment variable:

.. code:: shell

   MIOPEN_ENABLE_LOGGING_CMD=1

Here is an example command:

.. code:: shell

   MIOPEN_ENABLE_LOGGING_CMD=1 python test_nn.py -v -k test_batchnorm_cudnn_nhwc

The output might look like this:

.. code:: shell

   MIOpen(HIP): Command [LogCmdBNorm] ./bin/MIOpenDriver bnorm -n 4 -c 8 -H 2 -W 2 -m 1 --forw 1 -b 0 -r 1 -s 1 --layout NHWC
   MIOpen(HIP): Command [LogCmdBNorm] ./bin/MIOpenDriver bnorm -n 4 -c 8 -H 2 -W 2 -m 1 --forw 0 -b 1 -s 1 --layout NHWC
   MIOpen(HIP): Command [LogCmdBNorm] ./bin/MIOpenDriver bnorm -n 4 -c 8 -H 2 -W 2 -m 1 --forw 1 -b 0 -r 1 -s 1 --layout NCHW
   MIOpen(HIP): Command [LogCmdBNorm] ./bin/MIOpenDriver bnorm -n 4 -c 8 -H 2 -W 2 -m 1 --forw 0 -b 1 -s 1 --layout NCHW
   MIOpen(HIP): Command [LogCmdBNorm] ./bin/MIOpenDriver bnorm -n 2 -c 8 -H 8 -W 1 -m 1 --forw 1 -b 0 -r 1 -s 1 --layout NHWC
   MIOpen(HIP): Command [LogCmdBNorm] ./bin/MIOpenDriver bnorm -n 2 -c 8 -H 8 -W 1 -m 1 --forw 0 -b 1 -s 1 --layout NHWC
   MIOpen(HIP): Command [LogCmdBNorm] ./bin/MIOpenDriver bnorm -n 2 -c 8 -H 8 -W 1 -m 1 --forw 1 -b 0 -r 1 -s 1 --layout NCHW
   MIOpen(HIP): Command [LogCmdBNorm] ./bin/MIOpenDriver bnorm -n 2 -c 8 -H 8 -W 1 -m 1 --forw 0 -b 1 -s 1 --layout NCHW

Each line corresponds to a different command or operation.
The ``./bin/MIOpenDriver`` string indicates that MIOpen was used for the operation.
The ``--layout`` parameter shows whether NHWC or NCHW was used, for example, ``--layout NHWC`` means the
NHWC memory format was used.
