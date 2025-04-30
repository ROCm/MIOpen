.. meta::
  :description: Using the MIOpen Find Database
  :keywords: MIOpen, ROCm, API, documentation

********************************************************************
Using the find database
********************************************************************

MIOpen 2.0 introduced :doc:`immediate mode <../how-to/find-and-immediate>`, which
is based on a find database called FindDb. This database contains the results of calls to the legacy ``Find()`` stage.

.. note::

   Prior to MIOpen 2.0, you could use calls (such as ``miopenFindConvolution*Algorithm()``) to gather a
   set of convolution algorithms in the form of an array of ``miopenConvSolution_t`` structs. This process
   is time consuming because it requires online benchmarking of competing algorithms.

FindDb consists of two parts:

*  **System FindDb**: A system-wide storage that holds pre-run values for the most applicable
   configurations.
*  **User FindDb**: A per-user storage that holds results for arbitrary user-run configurations. It also
   serves as a cache for the ``Find()`` stage.

The User FindDb *always takes precedence* over the System FindDb.

By default, System FindDb resides within the MIOpen install location, while User FindDb resides in your
home directory.

.. note::

   *  The System FindDb is *not* modified upon installation of MIOpen.
   *  There are separate Find databases for the HIP and OpenCL backends.

Populating User FindDb
=============================================================

MIOpen collects FindDb information during the following API calls:

*  ``miopenFindConvolutionForwardAlgorithm()``
*  ``miopenFindConvolutionBackwardDataAlgorithm()``
*  ``miopenFindConvolutionBackwardWeightsAlgorithm()``

During the call, find data entries are collected for one specific "problem configuration", which is implicitly
defined by the tensor descriptors and convolution descriptor passed to the API function.

Updating MIOpen and User FindDb
=============================================================

When you install a new version of MIOpen, the new version ignores old User FindDb files. Therefore,
you don't need to move or delete the old User FindDb files.

To collect the previous information again into the new User FindDb, use the same steps you
followed in the previous version. Re-collecting information keeps immediate mode optimized.

Disabling FindDb
=============================================================

To disable FindDb, set the ``MIOPEN_DEBUG_DISABLE_FIND_DB`` environmental variable to ``1``:

.. code:: bash

   export MIOPEN_DEBUG_DISABLE_FIND_DB=1

.. note::

   System FindDb can be cached into memory, which might dramatically increase performance. To disable
   this option, set the ``DMIOPEN_DEBUG_FIND_DB_CACHING`` CMake configuration flag to off.

   .. code:: bash

      -DMIOPEN_DEBUG_FIND_DB_CACHING=Off
