.. meta::
  :description: Find and immediate modes
  :keywords: MIOpen, ROCm, API, documentation

***********************************************************************************
Using the find APIs and immediate mode
***********************************************************************************

MIOpen contains several convolution algorithms for each stage of training or inference. Prior to
MIOpen version 2.0, you had to call find methods in order generate a set of applicable algorithms.

Here's a typical workflow for the find stage:

.. code:: cpp

  miopenConvolutionForwardGetWorkSpaceSize(handle,
                                          weightTensorDesc,
                                          inputTensorDesc,
                                          convDesc,
                                          outputTensorDesc,
                                          &maxWorkSpaceSize);

  // < allocate workspace >


  // NOTE:
  // The miopenFindConvolution*() call is expensive in terms of run time and required workspace.
  // Therefore, we highly recommend reserving the required algorithm and workspace so that you can
  // reuse them later (within the lifetime of the same MIOpen handle object).
  // With this approach, there should be no need to invoke miopenFind*() more than once per
  // application lifetime.

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

The results of `find` are returned in an array of ``miopenConvAlgoPerf_t`` structs in order of
performance, with the fastest at index 0.

This call sequence is only run once per session, as it's inherently expensive. Within the sequence,
``miopenFindConvolution*()`` is the most expensive call. ``miopenFindConvolution*()`` caches its own
results on disk so the subsequent calls during the same MIOpen session run faster.

Internally, MIOpen's find calls compile and benchmark a set of ``solvers`` contained in
``miopenConvAlgoPerf_t``. This is performed in parallel with ``miopenConvAlgorithm_t``. You can
control the level of parallelism using an environmental variable. Refer to the debugging section,
:ref:`controlling parallel compilation <control-parallel-compilation>` for more information.

Immediate mode
=====================================================

MIOpen v2.0 introduces immediate more, which removes the requirement for
``miopenFindConvolution*()`` calls, thereby reducing runtime costs. In this mode, you can query the
MIOpen runtime for all of the supported solutions for a given convolution configuration. The sequence
of operations for immediate mode is similar to launching regular convolutions in MIOpen (i.e., through
the use of the ``miopenFindConvolution*()`` API). However, in this case, the different APIs have a lower
runtime cost.

A typical convolution call is similar to the following sequence:

* You construct the MIOpen handle and relevant descriptors, such as the convolution descriptor.
* With the above data structures, you call ``miopenConvolution*GetSolutionCount`` to get the
  maximum number of supported solutions for the convolution descriptor.
* The obtained count is used to allocate memory for the ``miopenConvSolution_t`` structure,
  introduced in MIOpen v2.0.
* You call ``miopenConvolution*GetSolution`` to populate the ``miopenConvSolution_t`` structures
  allocated above. The returned list is in the order of best performance (where the first element is the
  fastest).
* While the above structure returns the amount of workspace required for an algorithm, you can
  inquire the amount of a workspace required for a known solution ID using
  ``miopenConvolution*GetSolutionWorkspaceSize``. However, this is not a requirement (because the
  structure returned by ``miopenConvolution*GetSolution`` already has this information).
* Now you can initiate the convolution operation in ``immediate`` mode by calling
  ``miopenConvolution*Immediate``. This populates the output tensor descriptor with the respective
  convolution result. However, the first call to ``miopenConvolution*Immediate`` may consume more
  time because if the kernel isn't present in the kernel cache, it would need to be compiled.
* Optionally, you can compile the solution of choice by calling ``miopenConvolution*CompileSolution``.
  This ensures that the kernel represented by the chosen solution is populated in the kernel cache,
  removing the need to compile the kernel in question.

.. code:: cpp

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


  // This stage is optional.
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

Immediate mode fallback
-----------------------------------------------------------------------------------------------

Although immediate mode is underpinned by :doc:`FindDb <../conceptual/finddb>`, it may not contain every
configuration of interest. If FindDb encounters a database miss, it has two fallback paths it can take,
depending on whether the CMake variable ``MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK`` is set to
``ON`` or ``OFF``.

If you require the best possible performance, run the find stage at least once.

AI-based heuristic fallback (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK`` is set to ``ON`` (default), the immediate mode
behavior upon encountering a database miss is to use an AI-based heuristic to pick the optimal
solution.

First, the applicability of the AI-based heuristic for the given configuration is checked. If the heuristic is
applicable, it feeds various parameters of the given configuration into a neural network that has been
tuned to predict the optimal solution with 90% accuracy.

Weighted throughput index-based fallback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK`` is set to ``OFF``, or the AI heuristic is not
applicable for the given convolution configuration, the immediate mode behavior upon encountering
a database miss is to use a weighted throughput index-based mechanism to estimate which solution
would be optimal (based on the convolution configuration parameters).

Limitations of immediate mode
-----------------------------------------------------------------------------------------------

System FindDb has only been populated for these architectures:

* gfx906 with 64 CUs
* gfx906 with 60 CUs
* gfx900 with 64 CUs
* gfx900 with 56 CUs

If your architecture isn't listed, you must run the find API on your system (once per application) in
order to take advantage of immediate mode's more efficient behavior.

Backend limitations
-----------------------------------------------------------------------------------------------

OpenCL support for immediate mode via the fallback is limited to FP32 datatypes. This is because the
current release's fallback path goes through GEMM, which is serviced through MIOpenGEMM (on
OpenCL). MIOpenGEMM only contains support for FP32.

The HIP backend uses rocBLAS as its fallback path, which contains a more robust set of datatypes.

.. _find_modes:

Find modes
============================================================

MIOpen provides a set of find modes that are used to accelerate find API calls. The different
modes are set by using the ``MIOPEN_FIND_MODE`` environment variable with one of these values:

* ``NORMAL``/``1`` (normal find): This is the full find mode call, which benchmarks all the solvers and
  returns a list.
* ``FAST``/``2`` (fast find): Checks :doc:`FindDb <../conceptual/finddb>` for an entry. If there's a FindDb
  hit, it uses that entry. If there's a miss, it uses the immediate mode fallback. Offers fast start-up times
  at the cost of GPU performance.
* ``HYBRID``/``3`` or unset ``MIOPEN_FIND_MODE`` (hybrid find): Checks
  :doc:`FindDb <../conceptual/finddb>` for an entry. If there's a FindDb hit, it uses that entry. If there's a
  miss, it uses the existing find machinery. Offers slower start-up times than fast find without the GPU
  performance drop.
* ``4``: This value is reserved and should not be used.
* ``DYNAMIC_HYBRID``/``5`` (dynamic hybrid find): Checks :doc:`FindDb <../conceptual/finddb>` for an
  entry. If there's a FindDb hit, it uses that entry. If there's a miss, it uses the existing find machinery
  (skipping non-dynamic kernels). It offers faster start-up times than hybrid find, but GPU performance
  may decrease.

The default find mode is ``DYNAMIC_HYBRID``. To run the full ``NORMAL`` find mode, use
``export MIOPEN_FIND_MODE=NORMAL`` or ``export MIOPEN_FIND_MODE=1``.
