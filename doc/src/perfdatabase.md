Performance Database Search
===========================

Users can now update the PerfDb file on their systems by setting the appropriate value of **MIOPEN_FIND_ENFORCE** environemnt variable. Both symbolic and numeric values are supported.

This behavior is currently only supported in the `BUILD_DEV` mode while building MIOpen.

**NONE (1)**

MIOpen performs Exhaustive Search only if explicitly requested via MIOpen API and there is no record containing optimized solution(s) for the given problem configuration in the PerfDb. If relevant record exists in the PerfDb, then MIOpen reads optimized config from it, thus saving time.

The optimized solution found during the successful Search process is written into the PerfDb for future re-use. That is why MIOpen will not Search for optimized solution more than once for a given problem in this mode.

See documentation about miopenFind*() API calls for more info on how Search can be explicitly requested.


**DB_UPDATE (2)**

Similar to NONE, but Search will NOT be skipped if PerfDb contains relevant record. If Search is requested via MIOpen API, then MIOpen will perform the Search and update PerfDb.

Note: This mode is intended for tuning the MIOpen installation. When MIOpen is in this mode, the real-life applications that use it can take too long to finish. The reason is that a lengthy Search process may be run every time when miopenFind*() gets invoked.

**SEARCH (3)**

Similar to NONE, but performs Search even if not requested via MIOpen API explicitly. Like in the NONE mode, MIOpen will not Search for optimized solution more than once. If relevant record exists in the PerfDb, then MIOpen just uses it and skips the Search process.

This mode allows to auto-tune MIOpen even for applications that do not anticipate means for getting the best performance from MIOpen.

If MIOpen works is in this mode, and the user application run for the first time, then the user app may take substantially longer time than expected.

**SEARCH_DB_UPDATE (4)**

A combination of SEARCH and DB_UPDATE. MIOpen performs the Search on each miopenFind*() call and then updates the PerfDb.

Note: This mode is intended for tuning the MIOpen installation. When MIOpen is in this mode, the real-life apps may take too long to finish. The reason is that a lengthy Search process may be run every time when miopenFind*() gets invoked.

**DB_CLEAN (5)**

MIOpen removes relevant records from the PerfDb instead of just reading and using those. Search is blocked, even if explicitly requested.

## MIOPEN_FIND_ENFORCE_SCOPE

This variable allows to limit the scope of `MIOPEN_FIND_ENFORCE`, so that only forward, backward data or backward weights convolutions will be affected. Both symbolic and numeric values are supported, as shown below.

**ALL (1)** `MIOPEN_FIND_ENFORCE` affects all convolutions. This is the default.
		
**CONV_FWD (2)** `MIOPEN_FIND_ENFORCE` affects only Forward convolutions.

**CONV_BWD (3)** `MIOPEN_FIND_ENFORCE` affects only Backward Data convolutions.

**CONV_WRW (4)** `MIOPEN_FIND_ENFORCE` affects only Backward With Regard to Weights (a.k.a WRW) convolutions.
