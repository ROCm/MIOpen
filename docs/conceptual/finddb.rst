.. meta::
  :description: Find-Db Database
  :keywords: MIOpen, ROCm, API, documentation

********************************************************************
Find-Db database
********************************************************************

Prior to MIOpen 2.0, you could use calls (such as ``miopenFindConvolution*Algorithm()``) to gather a
set of convolution algorithms in the form of an array of ``miopenConvSolution_t`` structs. This process
is time-consuming because it requires online benchmarking of competing algorithms.

As of MIOpen 2.0, we introduced an :doc:`immediate mode <../reference/find-and-immediate>`, which
is based on a database that contains the results of calls to the legacy ``Find()`` stage. We refer to this
database as Find-Db.

Find-Db consists of two parts:

* **System Find-Db**: A system-wide storage that holds pre-run values for the most applicable
  configurations
* **User Find-Db**: A per-user storage that is intended to hold results for arbitrary user-run
  configurations. It also serves as a cache for the ``Find()`` stage.

User Find-Db *always takes precedence* over System Find-Db.

By default, System Find-Db resides within MIOpen's install location, while User Find-Db resides in your
home directory.

Note that:

 * The System Find-Db is *not* modified upon installation of MIOpen.
 * There are separate Find databases for HIP and OpenCL backends.

Populating User Find-Db
=============================================================

MIOpen collects Find-db information during the following API calls:

* ``miopenFindConvolutionForwardAlgorithm()``
* ``miopenFindConvolutionBackwardDataAlgorithm()``
* ``miopenFindConvolutionBackwardWeightsAlgorithm()``

During the call, find data entries are collected for one `problem configuration`, which is implicitly
defined by the tensor descriptors and convolution descriptor passed to API function.


Updating MIOpen and User Find-Db
=============================================================

When you install a new version of MIOpen, this new version ignores old User Find-Db files. Therefore,
you don't need to move or delete the old User Find-Db files.

If you want to re-collect the information into the new User Find-Db, you can use the same steps you
followed in the previous version. Re-collecting information keeps immediate mode optimized.


Disabling Find-Db
=============================================================

You can disable Find-Db by setting the ``MIOPEN_DEBUG_DISABLE_FIND_DB`` environmental variable
to 1:

.. code:: bash

  export MIOPEN_DEBUG_DISABLE_FIND_DB=1


.. note::

  System Find-Db can be cached into memory and may dramatically increase performance. To disable
  this option, set the ``DMIOPEN_DEBUG_FIND_DB_CACHING`` CMake configuration flag to off.

.. code:: bash

  -DMIOPEN_DEBUG_FIND_DB_CACHING=Off
