.. meta::
  :description: Using the MIOpen kernel cache
  :keywords: MIOpen, ROCm, API, documentation, kernel cache

********************************************************************
Kernel cache
********************************************************************

MIOpen caches binary kernels to disk so they don't need to be compiled the next time you run the
application. This cache is stored in ``$HOME/.cache/miopen`` by default, but you can change this at
build time by setting the ``MIOPEN_CACHE_DIR`` CMake variable.

Clear the cache
====================================================

You can clear the cache by deleting the cache directory (for example, ``$HOME/.cache/miopen``). However,
you should only do this for development purposes or to free disk space. You don't need to
clear the cache when upgrading MIOpen.

Disabling the cache
====================================================

Disabling the cache is generally useful for development purposes. You can disable the cache
in this following situations:

*  During the build, either set ``MIOPEN_CACHE_DIR`` to an empty string or set
   ``BUILD_DEV=ON`` when configuring CMake.
*  At runtime, set the ``MIOPEN_DISABLE_CACHE`` environment variable to ``true``.

Updating MIOpen and removing the cache
===============================================================

For MIOpen version 2.4 and later, MIOpen's kernel cache directory is versioned, so any existing cached kernels
won't collide when upgrading.

.. note::

   For MIOpen version 2.3 and earlier, if the compiler changes or you modify the kernels, then you must
   delete the cache for the existing MIOpen version using the command
   ``rm -rf $HOME/.cache/miopen/<miopen-version-number>``.

Installing precompiled kernels
====================================================

GPU architecture-specific, precompiled kernel packages are available in the ROCm package
repositories. These packages reduce the startup latency of MIOpen kernels. They contain a kernel cache file,
which they install in the ROCm installation directory along with other MIOpen artifacts. When MIOpen launches a
kernel, it first checks for a kernel in the kernel cache within the MIOpen installation directory. If
the file doesn't exist, or the required kernel isn't found, it compiles the kernel and places it in the
kernel cache.

These packages are optional and must be separately installed from MIOpen. To conserve
disk space, you can choose not to install these packages (which would result in higher
startup latency). You also have the option to only install kernel packages for your device architecture,
which helps save disk space.

If the MIOpen kernels package is not installed, or if the kernel doesn't match the GPU, you'll get a
warning message similar to:

.. code:: bash

   > MIOpen(HIP): Warning [SQLiteBase] Missing system database file:gfx906_60.kdb Performance may degrade

The performance degradation mentioned in the warning only affects the network start-up time (the
"initial iteration time") and can be safely ignored.

Refer to the :doc:`installation instructions <../install/install>` for guidance on installing the MIOpen
kernels package.
