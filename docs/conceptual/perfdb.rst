.. meta::
  :description: Using the MIOpen performance database
  :keywords: MIOpen, ROCm, API, documentation, performance database

************************************************************************************************
Using the performance database
************************************************************************************************

Many MIOpen kernels have parameters that affect their performance. Setting these parameters to
optimal values allows for the best possible throughput. The optimal values depend on many factors,
including the network configuration, GPU type, clock frequencies, and ROCm version.

Due to the large number of possible configurations and settings, MIOpen provides a set of pre-tuned
values for the "most applicable" network configurations and a method for expanding the set of
optimized values. MIOpen's performance database (PerfDb) contains these pre-tuned parameter values
in addition to any user-optimized parameters.

The PerfDb consists of two parts:

* **System PerfDb**: A system-wide storage that holds pre-run values for the most applicable
  configurations.
* **User PerfDb**: A per-user storage that holds optimized values for arbitrary configurations.

The User PerfDb *always takes precedence* over System PerfDb.

MIOpen also has auto-tuning functionality, which is able to find optimized kernel parameter values for
a specific configuration. The auto-tune process might take a long time, but after the optimized values are
found, they're stored in the User PerfDb. MIOpen then automatically reads and uses these parameter
values.

By default, System PerfDb resides within the MIOpen install location, while User PerfDb resides in your
home directory. For more information, see :ref:`setting up locations <setting-up-locations>`.

The System PerfDb is not modified during the MIOpen installation.

Auto-tuning kernels
==========================================================

MIOpen performs auto-tuning during the these API calls:

* ``miopenFindConvolutionForwardAlgorithm()``
* ``miopenFindConvolutionBackwardDataAlgorithm()``
* ``miopenFindConvolutionBackwardWeightsAlgorithm()``

Auto-tuning is performed for only one "problem configuration", which is implicitly defined by the
tensor descriptors that are passed to the API function.

In order for auto-tuning to begin, the following conditions must be met:

* The applicable kernels must have tuning parameters
* The value of the ``exhaustiveSearch`` parameter is set to ``true``
* Neither the System nor User PerfDb can contain values for the relevant "problem configuration".

You can override the latter two conditions and force the search using either the API call
``miopenSetTuningPolicy()`` or the ``-MIOPEN_FIND_ENFORCE`` environment variable. In addition to
controlling the auto-tuning behaviour of convolutions, both ``miopenSetTuningPolicy()`` and
``-MIOPEN_FIND_ENFORCE`` can be used to control the tuning for batch normalization.
See the following section for more details.

To optimize performance, MIOpen provides several find modes to accelerate find API calls.
These modes include:

*  normal find
*  fast find
*  hybrid find
*  dynamic hybrid find

For more information about the MIOpen find modes, see :ref:`Find modes <find_modes>`.

Using MIOPEN_FIND_ENFORCE or miopenSetTuningPolicy() to control auto-tuning
----------------------------------------------------------------------------------------------------------

``MIOPEN_FIND_ENFORCE`` supports case-insensitive symbolic and numeric values. The possible values
are:

* ``NONE``/``(1)``: No change in the default behavior.
* ``DB_UPDATE``/``(2)``: Do not skip auto-tune (even if PerfDb already contains optimized values). If you
  request auto-tune via API, MIOpen performs it and updates PerfDb. You can use this mode for
  fine-tuning the MIOpen installation on your system. However, this mode slows down the processes.
* ``SEARCH``/``(3)``: Perform auto-tune even if not requested via API. In this case, the library behaves as
  if the ``exhaustiveSearch`` parameter is set to ``true``. If PerfDb already contains optimized values,
  auto-tune is not performed. You can use this mode to tune applications that don't anticipate any means
  of getting the best performance from MIOpen. When in this mode, your application's first run might
  take substantially longer than expected.
* ``SEARCH_DB_UPDATE``/``(4)``: A combination of ``DB_UPDATE`` and ``SEARCH``. MIOpen performs
  auto-tune and updates User PerfDb on each ``miopenFindConvolution*()`` call. This mode is
  only recommended for debugging purposes.
* ``DB_CLEAN``/``(5)``: Removes optimized values related to the "problem configuration" from User
  PerfDb. Auto-tune is blocked, even if explicitly requested. System PerfDb is left intact.

  .. caution::

      Use the ``DB_CLEAN`` option with care.

Note that the API call miopenSetTuningPolicy() can be used to set the same modes as
``MIOPEN_FIND_ENFORCE``.  For example, to set the ``SEARCH`` mode, code like the following could be used:
.. code-block:: c

    miopenSetTuningPolicy(handle, miopenTuningPolicySearch);
    miopenBatchNorm*()
    miopenSetTuningPolicy(handle, miopenTuningPolicyNone);

Note that this API method is supported for both convolutions and batchnorms, although batchnorm does
not support a policy of ``DB_UPDATE`` (this will be a no-op and the user should specify ``SEARCH_DB_UPDATE``
instead if they want ``DB_UPDATE`` behavior).

If both the API method and environment variable are used, then the API method takes precedence.

Updating MIOpen and User PerfDb
==========================================================

If you install a new version of MIOpen, it is recommended that you move or delete your old User
PerfDb file. This prevents older database entries from affecting configurations within the newer system
database. The User PerfDb is named ``miopen.udb`` and can be found at the User PerfDb path location.
