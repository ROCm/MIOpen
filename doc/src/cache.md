Kernel Cache
============

MIOpen will cache binary kernels to disk, so they don't need to be compiled the next time the application is run. This cache is stored by default in `$HOME/.cache/miopen`. This location can be customized at build time by setting the `MIOPEN_CACHE_DIR` cmake variable. 

Clear the cache
---------------

The cache can be cleared by simply deleting the cache directory (i.e., `$HOME/.cache/miopen`). This should only be needed for development purposes or to free disk space. The cache does not need to be cleared when upgrading MIOpen.

Disabling the cache
-------------------

The are several ways to disable the cache. This is generally useful for development purposes. The cache can be disabled during build by either setting `MIOPEN_CACHE_DIR` to an empty string, or setting `BUILD_DEV=ON` when configuring cmake. The cache can also be disabled at runtime by setting the `MIOPEN_DISABLE_CACHE` environment variable to true.

Updating MIOpen and removing the cache
--------------------------------------
If the compiler changes, or the user modifies the kernels then the cache must be deleted for the MIOpen version in use; e.g., `rm -rf ~/.cache/miopen/<miopen-version-number>`.
    