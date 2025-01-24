.. meta::
  :description: MI200 MFMA behavior specifics
  :keywords: MIOpen, ROCm, API, documentation, MI200, MFMA

***********************************************************************************
MI200 matrix fused multiply-add (MFMA) behavior specifics
***********************************************************************************

On the MI200, ``MFMA_F16``, ``MFMA_BF16``, and ``MFMA_BF16_1K`` flush subnormal input/output data to
zero. This behavior might affect the convolution operation in certain workloads due to the limited
exponent range of the half-precision floating-point datatypes.

MIOpen offers an alternate implementation for the half-precision datatype via conversion instructions
to utilize the larger exponent range of the ``BFloat16`` data type, albeit with reduced accuracy. The following
caveats apply to this alternate implementation:

* It's disabled by default in the forward convolution operations.

* It's enabled by default in the backward data and backward weights convolution operations.

* You can override the default MIOpen behavior by using the ``miopenSetConvolutionAttribute`` API
  call. To use the alternate implementation, pass the convolution descriptor for the appropriate
  convolution operation and the ``MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL`` convolution
  attribute (with a non-zero value).

* You can also override the behavior using the
  ``MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL`` environment variable. When set to ``1``,
  ``MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL``  engages the alternate implementation.
  When set to ``0``, it's disabled. This environment variable impacts the convolution
  operation in all directions.
