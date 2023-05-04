## MI200 MFMA Behavior Specifics

The MI200 MFMA_F16, MFMA_BF16 and MFMA_BF16_1K flush subnormal input/output data to zero. This behavior might affect the convolution operation in certain workloads due to the limited exponent range of the half-precision floating point datatypes.  

An alternate implementation for the half precision data-type is available in MIOpen which utilizes conversion instructions to utilizes the BFloat16 data-types larger exponent range, albeit with reduced accuracy. The following salients apply to this alternate implementation:  

* It is disabled by default in the Forward convolution operations. 

* It is enabled by default in the backward data and backward weights convolution operations. 

* The default MIOpen behaviors described above may be overridden using the `miopenSetConvolutionAttribute` API call and passing the convolution descriptor for the appropriate convolution operation and the `MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL` convolution attribute with a non-zero value to engage the alternate implementation. 

* The behavior might also be overridden using the `MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL` environment variable. The above variable when set to a value of `1` engages the alternate implementation while a value of `0` disables it. Keep in mind the environment variable impacts the convolution operation in all directions. 