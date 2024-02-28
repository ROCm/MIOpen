## Build the driver

MIOpen provides an [application-driver](https://github.com/ROCm/MIOpen/tree/master/driver) which can be used to execute any one particular layer in isolation and measure performance and verification of the library. 

The driver can be built using the `MIOpenDriver` target:

` cmake --build . --config Release --target MIOpenDriver ` **OR** ` make MIOpenDriver `

For documentation on how to run the driver, refer to [Run the driver](https://github.com/ROCm/MIOpen/blob/develop/driver/README.md)

