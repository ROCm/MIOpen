Performance Database Search
===========================

Supported values of **MIOPEN_FIND_ENFORCE**

Both symbolic and numeric values are supported.


**NONE (1)**

MIOpen performs Exhaustive Search only if explicitly requested via MIOpen API and there is there is no record containing optimized solution(s) for the given problem configuration in the PerfDb. If relevant record exists in the Perf Db, then MIOpen reads optimized config from it, thus saving time.

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

MIOpen removes relevant records from the PerfDb instead of just reading and using those. Search is blocked, even is explicitly requested.
