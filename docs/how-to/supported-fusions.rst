Supported fusions 
-------------------

The tables below outlines the supported fusions for fp32 and fp16 as well as any applicable constraints. **(C = convolution, B = bias, N = batch normalization, A = activation)**
Fusion Plans with grouped convolutions are not supported.

<img title="Convolution based fp32 fusion" src="/docs/fp32fusions.png">


<img title="Convolution based fp16 fusion" src="/docs/fp16fusions.png">


Performance comparison to non-fused kernels
===========================================


The following graph depicts the speedup gained for a fused Convolution+Bias+Activation over a non-fused version, all configurations have a batch size of 64:

<img title="CBA Graph" src="/docs/cba.png">


Speedup obtained by fusing Batchnorm (spatial mode) with Activation are presented in the graph below:

<img title="Batchnorm activation fusion" src="/docs/na.png">
