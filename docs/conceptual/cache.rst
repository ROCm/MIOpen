.. meta::
  :description: Using kernel cache
  :keywords: MIOpen, ROCm, API, documentation, kernel cache

********************************************************************
Kernel cache
********************************************************************

MIOpen caches binary kernels to disk so they don't need to be compiled the next time you run the
application. This cache is stored in ``$HOME/.cache/miopen`` by default, but you can change this at
build time by setting the ``MIOPEN_CACHE_DIR`` CMake variable.

Clear the cache
====================================================

You can clear the cache by deleting the cache directory (e.g., ``$HOME/.cache/miopen``). We
recommend that you only do this for development purposes or to free disk space. You don't need to
clear the cache when upgrading MIOpen.

Disabling the cache
====================================================

Disabling the cache is generally useful for development purposes. You can disable the cache:

* During **build**, either by setting ``MIOPEN_CACHE_DIR`` to an empty string or setting
  ``BUILD_DEV=ON`` when configuring CMake
* At **runtime** by setting the ``MIOPEN_DISABLE_CACHE`` environment variable to ``true``.

Updating MIOpen and removing the cache
===============================================================

For MIOpen version 2.3 and earlier, if the compiler changes or you modify the kernels. then you must
delete the cache for the existing MIOpen version
(e.g., ``rm -rf $HOME/.cache/miopen/<miopen-version-number>``).

For MIOpen version 2.4 and later, MIOpen's kernel cache directory is versioned, so cached kernels
won't collide when upgrading.

Installing pre-compiled kernels
====================================================

GPU architecture-specific, pre-compiled kernel packages are available in the ROCm package
repositories. These reduce the startup latency of MIOpen kernels (they contain the kernel cache file
and install it in the ROCm installation directory along with other MIOpen artifacts). When launching a
kernel, MIOpen first checks for a kernel in the kernel cache within the MIOpen installation directory. If
the file doesn't exist, or the required kernel isn't found, the kernel is compiled and placed in your
kernel cache.

These packages are optional and must be separately installed from MIOpen. If you want to conserve
disk space, you may choose not to install these packages (though without them, you'll have higher
startup latency). You also have the option to only install kernel packages for your device architecture,
which helps save disk space.

If the MIOpen kernels package is not installed, or if the kernel doens't match the GPU, you'll get a
warning message similar to:

.. code:: bash

  > MIOpen(HIP): Warning [SQLiteBase] Missing system database file:gfx906_60.kdb Performance may degrade

The performance degradation mentioned in the warning only affects the network start-up time (the
"initial iteration time") and can be safely ignored.

Refer to the :doc:`installation instructions <../install/install>` for guidance on installing the MIOpen
kernels package.
