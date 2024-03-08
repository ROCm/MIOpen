## Building the driver

MIOpen provides an [application-driver](https://github.com/ROCm/MIOpen/tree/master/driver) which can be used to execute any one particular layer in isolation and measure performance and verification of the library. 

The driver can be built using the `MIOpenDriver` target:

` cmake --build . --config Release --target MIOpenDriver ` **OR** ` make MIOpenDriver `

Documentation on how to run the driver is [here](https://github.com/ROCm/MIOpen/blob/develop/driver/README.md). 