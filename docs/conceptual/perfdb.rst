.. meta::
  :description: Using the performance database
  :keywords: MIOpen, ROCm, API, documentation, performance database

************************************************************************************************
Using the performance database
************************************************************************************************

Many MIOpen kernels have parameters that affect their performance. Setting these parameters to
optimal values allows for the best possible throughput. Optimal values depend on many factors,
including network configuration, GPU type, clock frequencies, and ROCm version.

Due to the large number of possible configurations and settings, MIOpen provides a set of pre-tuned
values for the `most applicable` network configurations and a means for expanding the set of
optimized values. MIOpen's performance database (PerfDb) contains these pre-tuned parameter values
in addition to any user-optimized parameters.

The PerfDb consists of two parts:

* **System PerfDb**: A system-wide storage that holds pre-run values for the most applicable
  configurations.
* **User PerfDb**: A per-user storage that holds optimized values for arbitrary configurations.

User PerfDb `always takes precedence` over System PerfDb.

MIOpen also has auto-tuning functionality, which is able to find optimized kernel parameter values for
a specific configuration. The auto-tune process may take a long time, but once optimized values are
found, they're stored in the User PerfDb. MIOpen then automatically reads and uses these parameter
values.

By default, System PerfDb resides within MIOpen's install location, while User PerfDb resides in your
home directory. See :ref:`setting up locations <setting-up-locations>` for more information.

System PerfDb is not modified during MIOpen installation.

Auto-tuning kernels
==========================================================

MIOpen performs auto-tuning during the these API calls:

* ``miopenFindConvolutionForwardAlgorithm()``
* ``miopenFindConvolutionBackwardDataAlgorithm()``
* ``miopenFindConvolutionBackwardWeightsAlgorithm()``

Auto-tuning is performed for only one `problem configuration`, which is implicitly defined by the
tensor descriptors that are passed to the API function.

In order for auto-tuning to begin, the following conditions must be met:

* The applicable kernels have tuning parameters
* The value of the ``exhaustiveSearch`` parameter is ``true``
* Neither System nor User PerfDb can contain values for the relevant `problem configuration`.

You can override the latter two conditions by enforcing the search using the
``- MIOPEN_FIND_ENFORCE`` environment variable. You can also use this variable to remove values
from User PerfDb, as described in the following section.

To optimize performance, MIOpen provides several find modes to accelerate find API calls.
These modes include:

*  normal find
*  fast find
*  hybrid find
*  dynamic hybrid find
 
For more information about MIOpen find modes, see :ref:`Find modes <find_modes>`.

Using MIOPEN_FIND_ENFORCE
----------------------------------------------------------------------------------------------------------

``MIOPEN_FIND_ENFORCE`` supports symbolic (case-insensitive) and numeric values. Possible values
are:

* ``NONE``/``(1)``: No change in the default behavior.
* ``DB_UPDATE/``(2)``: Do not skip auto-tune (even if PerfDb already contains optimized values). If you
  request auto-tune via API, MIOpen performs it and updates PerfDb. You can use this mode for
  fine-tuning the MIOpen installation on your system. However, this mode slows down processes.
* ``SEARCH``/``(3)``: Perform auto-tune even if not requested via API. In this case, the library behaves as
  if the ``exhaustiveSearch`` parameter set to ``true``. If PerfDb already contains optimized values,
  auto-tune is not performed. You can use this mode to tune applications that don't anticipate means
  for getting the best performance from MIOpen. When in this mode, your application's first run may
  take substantially longer than expected.
* ``SEARCH_DB_UPDATE``/``(4)``: A combination of ``DB_UPDATE`` and ``SEARCH``. MIOpen performs
  auto-tune (and updates User PerfDb) on each ``miopenFindConvolution*()`` call. This mode is
  recommended only for debugging purposes.
* ``DB_CLEAN``/``(5)``: Removes optimized values related to the `problem configuration` from User
  PerfDb. Auto-tune is blocked, even if explicitly requested. System PerfDb is left intact. **Use this
  option with care.**

Updating MIOpen and User PerfDb
==========================================================

If you install a new version of MIOpen, we strongly recommend moving or deleting your old User
PerfDb file. This prevents older database entries from affecting configurations within the newer system
database. The User PerfDb is named ``miopen.udb`` and is located at the User PerfDb path.
