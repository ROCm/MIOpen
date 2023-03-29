Find and Immediate Mode
=======================



## Find API

MIOpen contains several convolution algorithms for each stage of training or inference. Pre-MIOpen version 2.0 users needed to call Find methods in order generate a set of applicable algorithms.

A typical workflow for the find stage:

```
miopenConvolutionForwardGetWorkSpaceSize(handle, 
                                         weightTensorDesc, 
                                         inputTensorDesc, 
                                         convDesc, 
                                         outputTensorDesc, 
                                         &maxWorkSpaceSize);

// < allocate workspace >


// NOTE:
// miopenFindConvolution*() call is expensive in terms of execution time and required workspace.
// Therefore it is highly recommended to save off the selected algorithm and workspace required so that
// can be reused later within the lifetime of the same MIOpen handle object.
// In this way, there should be is no need to invoke miopenFind*() more than once per application lifetime.

miopenFindConvolutionForwardAlgorithm(handle, 
                                      inputTensorDesc, 
                                      input_device_mem, 
                                      weightTensorDesc, 
                                      weight_device_mem,
                                      convDesc,
                                      outputTensorDesc, 
                                      output_device_mem,,
                                      request_algo_count,
                                      &ret_algo_count,
                                      perf_results,
                                      workspace_device_mem,
                                      maxWorkSpaceSize,
                                      1);

// < select fastest algorithm >

// < free previously allocated workspace and allocate workspace required for the selected algorithm>

miopenConvolutionForward(handle, &alpha,
                         inputTensorDesc, 
                         input_device_mem, 
                         weightTensorDesc, 
                         weight_device_mem,
                         convDesc,
                         perf_results[0].fwd_algo, // use the fastest algo
                         &beta,
                         outputTensorDesc, 
                         output_device_mem,
                         workspace_device_mem,
                         perf_results[0].memory); //workspace size                                           
```


The results of Find() are returned in an array of `miopenConvAlgoPerf_t` structs in order of performance, with the fastest at index 0.

This call sequence is executed once per session as it is inherently expensive. Of those, `miopenFindConvolution*()` is the most expensive call. It caches its own results on disk, so the subsequent calls during the same MIOpen session will execute faster. However, it is better to remember results of `miopenFindConvolution*()` in the application, as recommended above. 

Internally MIOpen's Find calls will compile and benchmark a set of `solvers` contained in `miopenConvAlgoPerf_t` this is done in parallel per `miopenConvAlgorithm_t`. The level of parallelism can be controlled using an environment variable. See the debugging section [controlling parallel compilation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/DebugAndLogging.html#controlling-parallel-compilation) for more details.


## Immediate Mode API

MIOpen v2.0 introduces the immediate which removes the requirement for the `miopenFindConvolution*()` calls and their associated runtime costs. In this mode, the user can query the MIOpen runtime for all the supported _solutions_ for a given convolution configuration. These solutions may either be using the same algorithm or different ones. The sequence of operations for in immediate mode is similar to launching regular convolutions in MIOpen i.e. through the use of the `miopenFindConvolution*()` API. However, in this case the different APIs have much lower runtime cost. A typical convolution call would be similar to the following sequence of calls:

* The user constructs the MIOpen handle and relevant descriptors such as the convolution descriptor as usual. 
* With the above data structures, the user calls `miopenConvolution*GetSolutionCount` to get the **maximum** number of supported solutions for the convolution descriptor in question.
* The count obtained above is used to allocate memory for the `miopenConvSolution_t` structure introduced in MIOpen v2.0
* The user calls `miopenConvolution*GetSolution` to populate the `miopenConvSolution_t` structures allocated above. The returned list is ordered in the order of best performance, thus the first element would be the fastest. 
* While the above structure returns the amount of workspace required for an algorithm, the user may inquire the amount of a workspace required for a known solution id by using the `miopenConvolution*GetSolutionWorkspaceSize` API call. However, this is not a requirement, since the strucure returned by `miopenConvolution*GetSolution` would already have this information. 
* Now the user may initiate the convolution operation in _immediate_ mode by calling `miopenConvolution*Immediate`. Which would populate the output tensor descriptor with the respective convolution result. However, the first call to `miopenConvolution*Immediate` may consume more time since the kernel may not be present in the kernel cache and may need to be compiled.
* Optionally, the user may compile the solution of choice by calling `miopenConvolution*CompileSolution` which would ensure that the kernel represented by the chosen solution is populated in the kernel cache a priori, removing the necessity for compiling the kernel in question. 


```
miopenConvolutionForwardGetSolutionCount(handle, 
                                         weightTensorDesc,
                                         inputTensorDesc,
                                         convDesc,
                                         outputTensorDesc,
                                         &solutionCount);


// < allocate an array of miopenConvSolution_t of size solutionCount >


miopenConvolutionForwardGetSolution(handle,
                                    weightTensorDesc,
                                    inputTensorDesc,
                                    convDesc,
                                    outputTensorDesc,
                                    solutionCount,
                                    &actualCount,
                                    solutions);

// < select a solution from solutions array >

miopenConvolutionForwardGetSolutionWorkspaceSize(handle,
                                                 weightTensorDesc,
                                                 inputTensorDesc,
                                                 convDesc,
                                                 outputTensorDesc,
                                                 selected->solution_id,
                                                 &ws_size);
 
// < allocate solution workspace of size ws_size >


// This stage is optional
miopenConvolutionForwardCompileSolution(handle,  
                                        weightTensorDesc,
                                        inputTensorDesc,
                                        convDesc,
                                        outputTensorDesc,
                                        selected->solution_id);



 miopenConvolutionForwardImmediate(handle,
                                   weightTensor,
                                   weight_device_mem,
                                   inputTensorDesc,
                                   input_device_mem,
                                   convDesc,
                                   outputTensorDesc,
                                   output_device_mem,
                                   workspace_device_mem,
                                   ws_size,
                                   selected->solution_id);                                                   
```

## Immediate Mode Fall Back

The immediate mode is underpinned by the [Find-Db](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/finddb.html), however it may not contain every configuration of interest. Immediate mode's behavior when encountering a database miss is to fallback to a GEMM algorithm. The GEMM algorithm will handle most cases, however, if the user requires performance they should run the Find stage at least once. Fallback's `miopenConvolution*GetSolution` returns only one `miopenConvSolution_t` structure and its `time` member contains negative value. Future releases will implement a more robust heuristic based fallback, which is expected to provide better (but still non-optimal) performance.



## Limitations of Immediate Mode

### Architectual Limitations
The system Find-Db has only been populated for the following architectures:
 * gfx906 with 64 CUs
 * gfx906 with 60 CUs
 * gfx900 with 64 CUs
 * gfx900 with 56 CUs

If the user's architecture is not listed above they will need to run the Find API once on their system per application in order to take advantage of immediate mode's more efficient behavior.


### Backend Limitations

OpenCL support for immediate mode via the fallback is limited to fp32 datatypes. This is because this current release's fallback path goes through GEMM which on the OpenCL is serviced through MIOpenGEMM -- which itself only contains support for fp32. The HIP backend uses rocBLAS as its fallback path which contains a richer set of datatypes.


### Find Modes

MIOpen provides a set of Find modes which are used to accelerate the Find calls. The different modes are set by using the environment variable `MIOPEN_FIND_MODE`, and setting it to one of the values:

- `NORMAL`, or `1`: Normal Find: This is the full Find mode call, which will benchmark all the solvers and return a list.
- `FAST`, or `2`: Fast Find: Checks the [Find-Db](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/finddb.html) for an entry. If there is a Find-Db hit, use that entry. If there is a miss, utilize the Immediate mode fallback. If Start-up times are expected to be faster, but worse GPU performance.
- `HYBRID`, or `3`, or unset `MIOPEN_FIND_MODE`: Hybrid Find: Checks the [Find-Db](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/finddb.html) for an entry. If there is a Find-Db hit, use that entry. If there is a miss, use the existing Find machinery. Slower start-up times than Fast Find, but no GPU performance drop.
- `4`: This value is reserved and should not be used.
- `DYNAMIC_HYBRID`, or `5`: Dynamic Hybrid Find: Checks the [Find-Db](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/finddb.html) for an entry. If there is a Find-Db hit, uses that entry. If there is a miss, uses the existing Find machinery with skipping non-dynamic kernels. Faster start-up times than Hybrid Find, but GPU performance may be a bit worse.

 Currently, the default Find mode is `DYNAMIC_HYBRID`. To run the full `NORMAL` Find mode, set the environment as:
 ```
 export MIOPEN_FIND_MODE=NORMAL
 ```
 Or,
 ```
  export MIOPEN_FIND_MODE=1
 ```
 
