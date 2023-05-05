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
For MIOpen version 2.3 and earlier, if the compiler changes, or the user modifies the kernels then the cache must be deleted for the MIOpen version in use; e.g., `rm -rf $HOME/.cache/miopen/<miopen-version-number>`. More information about the cache can be found [here](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/cache.html).

For MIOpen version 2.4 and later, MIOpen's kernel cache directory is versioned so that users' cached kernels will not collide when upgrading from earlier version.

Installing pre-compiled kernels
-------------------------------
GPU architecture-specific pre-compiled kernel packages are available in the ROCm package repositories, to reduce the startup latency of MIOpen kernels. In essence, these packages have the kernel cache file mentioned above and install them in the ROCm installation directory along with other MIOpen artifacts. Thus, when launching a kernel, MIOpen will first check for the existence of a kernel in the kernel cache installed in the MIOpen installation directory. If the file does not exist or the required kernel is not found, the kernel is compiled and placed in the user's kernel cache.

These packages are optional for the functioning of MIOpen and must be separately installed from MIOpen. Users who wish to conserve disk space may choose not to install these packages at the cost of higher startup latency. Users have the flexibility to only install kernel packages for installed device architecture, thus minimizing disk space usage.

If MIOpen kernels package is not installed, or if we do not deliver the kernels suitable for the user's GPU, then the user will get warning message like this:
> MIOpen(HIP): Warning [SQLiteBase] Missing system database file:gfx906_60.kdb Performance may degrade

The performance degradation mentioned in the warning only affects the network start-up time (aka "initial iteration time") and thus can be safely ignored.

Please refer to the MIOpen installation instructions: [installing MIOpen kernels package](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/install.html#installing-miopen-kernels-package) for guidance on installing the MIOpen kernels package.
