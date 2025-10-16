.. meta::
  :description: Using NHWC Batch Normalization on PyTorch
  :keywords: MIOpen, ROCm, API, documentation, NHWC Batch Normalization, PyTorch

************************************************************************************************
Using NHWC Batch Normalization with PyTorch
************************************************************************************************

This topic explains how to use NHWC Batch Normalization for MIOpen operations in PyTorch. NHWC is
a deep-learning memory format that has certain performance advantages over traditional
memory formats.

For information about installing and using PyTorch with ROCm, see :doc:`PyTorch on ROCm <rocm-install-on-linux:install/3rd-party/pytorch-install>`.
For a list of the ROCm components and features that PyTorch supports, see :doc:`PyTorch compatibility <rocm:compatibility/ml-compatibility/pytorch-compatibility>`.
For more background on using PyTorch and ROCm for AI tasks, see
:doc:`Training a model with PyTorch for ROCm <rocm:how-to/rocm-for-ai/training/benchmark-docker/pytorch-training>`.

NHWC versus NCHW
=================================================

NHWC (also known as "Channels Last") and NCHW are two types of memory formats for deep learning. They describe how
multidimensional arrays (nD) are translated to a linear (one-dimensional) memory address space.

*  NCHW (Number of samples, channels, height, width): This is the default data layout in which channels
   are stored separately from one another. The height and width information is stored after
   the channels.
*  NHWC (Number of samples, height, width, channels): In this alternative format, channels are stored next
   to each other after the height and width information.

The performance of NHWC is better than that of NCHW and is close to that observed when using a blocked memory format. NHWC is also
easier to work with for common operations.

For more information about these memory formats, see the
`PyTorch memory format documentation <https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_
and the `Intel Extension for PyTorch GitHub <https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/nhwc.html>`_.

Batch Normalization
=================================================

Batch Normalization (also known as Batchnorm or BatchNorm) enables higher learning rates and reduces initialization overhead by
normalizing layer inputs. Ordinarily, the distribution of the inputs to each layer changes as the
parameters to the previous layer change. This makes it more difficult to train deep learning models
and leads to lower learning rates. With Batch Normalization, normalization is part of the architecture
and is performed for each training batch.

For more information on Batch Normalization, see `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_.

Enabling or disabling NHWC Batch Normalization for MIOpen using PyTorch
=======================================================================

The PyTorch open-source tensor library provides support for using NHWC Batch Normalization with MIOpen.
In addition to Batch Normalization, NHWC support is also available for convolution and other MIOpen features.

NHWC Batch Normalization support in MIOpen can be used in a PyTorch environment using ROCm 7.0 or later.
This configuration supports 2D and 3D NHWC Batch Normalization. 1D Batch Normalization is not applicable to the NHWC format.

PyTorch branch support
------------------------

The ``ROCm/pytorch`` PyTorch images support NHWC Batch Normalization. ROCm 7.0 or later is required.
The following PyTorch branches support the NHWC Batch Normalization feature:

*  `release/2.6 <https://github.com/ROCm/pytorch/tree/release/2.6>`_
*  `release/2.7 <https://github.com/ROCm/pytorch/tree/release/2.7>`_

In the ``release/2.7`` PyTorch branch, NHWC Batch Normalization support in MIOpen is enabled by default.
To use the native Batch Normalization approach with this image, use this command:

.. code:: shell

   PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM=0

In the ``release/2.6`` PyTorch branch, NHWC Batch Normalization support in MIOpen is disabled by default.
To enable NHWC Batch Normalization for this image, use this command:

.. code:: shell

   PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM=1

For information about installing and using PyTorch on ROCm, see :doc:`PyTorch on ROCm <rocm-install-on-linux:install/3rd-party/pytorch-install>`.

Supported configurations
=================================================

The following table shows the Batch Normalization support for NHWC and NCHW with various data types and modes.
It also indicates which backend is used with and without the ``PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM``
environment variable enabled.

.. note::

   Mixed mode means that the Batch Normalization module has a different type than the inputs, for example,
   an input or gradient data type of ``FP16`` or ``BF16`` and a Batch Normalization type of ``FP32``.
   If the Batch Normalization module has the same data type as the inputs, for instance, an
   input or gradient data type of ``FP32`` and a Batch Normalization module that is also ``FP32``, the mode is
   "not mixed".

.. csv-table::
   :header: "Input data type","Memory format","Mode","Mixed/not mixed","Backend with NHWC Batch Normalization enabled","Backend with NHWC Batch Normalization disabled"
   :widths: 20, 20, 15, 15, 25, 25

   "``float32``","NCHW","1D/2D/3D","not mixed","MIOpen","MIOpen"
   "``float32``","NHWC","2D/3D","not mixed","native","MIOpen"
   "``float16``","NCHW","1D/2D/3D","mixed","MIOpen","MIOpen"
   "``float16``","NCHW","1D/2D/3D","not mixed","native","native"
   "``float16``","NHWC","2D/3D","mixed","native","MIOpen"
   "``float16``","NHWC","2D/3D","not mixed","native","native"
   "``bfloat16``","NCHW","1D/2D/3D","mixed","MIOpen (*)","MIOpen (*)"
   "``bfloat16``","NCHW","1D/2D/3D","not mixed","native","native"
   "``bfloat16``","NHWC","2D/3D","mixed","native","MIOpen"
   "``bfloat16``","NHWC","2D/3D","not mixed","native","native"

(*) MIOpen is used with ROCm 6.4 and later. Otherwise, the native backend is used.


Disabling MIOpen for Batch Normalization in PyTorch
====================================================

In some situations, you might not want to use MIOpen as the backend for Batch Normalization operations.
To disable the use of MIOpen with Batch Normalization, add this code to your application.

.. code:: python

   inp = torch.randn(size, requires_grad=True)
   grad = torch.randn(size, requires_grad=False)
   mod = nn.BatchNorm2d(inp.size(1), device="cuda")

   with torch.backends.cudnn.flags(enabled=False): # this line disables MIOpen for the two lines below, native batchnorm will be used

      out = mod(inp)
      out.backward(grad)

Verifying NHWC Batch Normalization use with MIOpen
===================================================

For some operations, it can be difficult to determine the backend and memory format used.
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

Running Batch Normalization tests
==================================

Several test suites are available for Batch Normalization. To test Batch Normalization training using both NHWC and NCHW in 2D,
run the following command:

.. code:: shell

   python test_nn.py -v -k test_batchnorm_2D_train

To test Batch Normalization training using both NHWC and NCHW in 3D,
run the following command:

.. code:: shell

   python test_nn.py -v -k test_batchnorm_3D_train

To test Batch Normalization inference for 2D using both memory formats, use this command:

.. code:: shell

   python test_nn.py -v -k test_batchnorm_2D_inference

To test the same functionality for 3D, use this command:

.. code:: shell

   python test_nn.py -v -k test_batchnorm_3D_inference
