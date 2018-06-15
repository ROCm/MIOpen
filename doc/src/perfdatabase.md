Performance Database
====================

Many of MIOpen kernels have parameters which affect their performance. Setting these parameters to optimal values allows to reach the best possible throughput. These optimal values depend on many things, including network configuration, GPU type, clock frequencies, ROCm version etc. Because of these dependencies and also due to enormous number of possible network configurations, it is virtually impossible to supply all values that users may need together with the library. Instead, MIOpen provides a set pre-tuned values for the _most applicable_ network configurations, **and** also means for expanding the set of optimized values. That is what the performance database is involved in. 

The performance database consists of two parts:
- **System Performace Database**, a system-wide storage which holds the pre-tuned values for the most applicable configurations,
- **User Performance Database**, a per-user storage which is intended to hold optimized values for arbitray configurations.

MIOpen also has the so-called "auto-tune" functionality, which is able to find optimized values. The auto-tune process may take substantial time. Once optimized values found, these are stored in User PerfDb. Then, the library automatically reads and uses these values when needed.

By default, System PerfDb resides within MIopen install location, while User PerfDb resides in the user's home directory. See [Setting up locations](https://github.com/ROCmSoftwarePlatform/MIOpen/blob/master/README.md#setting-up-locations) for more information.

User PerfDb always takes precedence over System PerfDb.

The System PerfDB is not modified upon installation of MIOpen.

## Auto-tuning the kernels.

MIOpen performs auto-tuning during the following MIOpen API calls:
- `miopenFindConvolutionForwardAlgorithm()`
- `miopenFindConvolutionBackwardDataAlgorithm()`
- `miopenFindConvolutionBackwardWeightsAlgorithm()`

During the call, auto-tuning is performed only for one _problem configuration_ (implicitly defined by the tensor descriptors passed to APi function).

The following conditions must be met for the auto-tune to begin:
- The applicable kernel(s) has tuning parameters.
- The passed value of `exhaustiveSearch` parameter is `true`, and
- Both System and User PerfDb do not yet contain values for the relevant _problem configuration_.

The latter two conditions may be overriden by _enforcing_ the search by means of the following environment variables:
- `MIOPEN_FIND_ENFORCE`
- `MIOPEN_FIND_ENFORCE_SCOPE`

These variables may also be used for _removing_ values from User PerfDb, see below.

### MIOPEN_FIND_ENFORCE

Both symbolic and numeric values are supported.

**NONE (1)**

No changes w.r.t default bahavior.

**DB_UPDATE (2)**

Auto-tune will not be skipped even if PerfDb already contains optimized values. If auto-tune is requested via API, then MIOpen will perform it and update PerfDb.

This mode can be used for fine-tuning the MIOpen installation on the user's system. When MIOpen is in this mode, the applications that use it may take quite long to finish.

**SEARCH (3)**

MIOpen will perform auto-tune even if not requested via MIOpen API. In other words, the library will behave as if `exhaustiveSearch` parameter set to `true` even this is not really so. If optimized values already reside in PerfDb, then auto-tune will not be performed.

This mode allows for tuning the apps that do not anticipate means for getting the best performance from MIOpen. When MIOpen is in this mode, the first run of the user's app may take substantially longer time than expected.

**SEARCH_DB_UPDATE (4)**

A combination of SEARCH and DB_UPDATE. MIOpen performs auto-tune (and updates User PerfDb) on each `miopenFindConvolution*()` call. It is not recommended to use this mode except for debugging purposes.

**DB_CLEAN (5)**

Use with care. MIOpen **removes** optimized values related to given _problem configuration_ from the User PerfDb. Auto-tune is blocked, even is explicitly requested. System PerfDb left intact. 

### MIOPEN_FIND_ENFORCE_SCOPE

This variable allows to limit the scope of `MIOPEN_FIND_ENFORCE`, so that only forward, backward data or backward weights convolutions will be affected. Both symbolic and numeric values are supported, as shown below.

**ALL (1)** `MIOPEN_FIND_ENFORCE` affects all convolutions. This is the default.
		
**CONV_FWD (2)** `MIOPEN_FIND_ENFORCE` affects only Forward convolutions.

**CONV_BWD (3)** `MIOPEN_FIND_ENFORCE` affects only Backward Data convolutions.

**CONV_WRW (4)** `MIOPEN_FIND_ENFORCE` affects only Backward With Regard to Weights (a.k.a WRW) convolutions.
