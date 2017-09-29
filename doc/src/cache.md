Kernel Cache
============

MIOpen will cache binary kernels to disk, so they don't need to be compiled the next time the application is run. This cache is stored by default in `$HOME/.cache/miopen`. This location can be customized at build time by setting the `MIOPEN_CACHE_DIR` cmake variable. 

Clear the cache
---------------

The cache can be cleared by simply deleting the cache directory(ie `$HOME/.cache/miopen`). This should only be needed for development purposes or to free disk space. The cache does not need to be cleared when upgrading MIOpen.
