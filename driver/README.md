# MIOpen Driver

The driver enables to test forward/backward of any particular layer in MIOpen. 

To build the driver, type `make MIOpenDriver` from the `build` directory.

All the supported layers in MIOpen can be found by the supported `base_args` here:

` ./bin/MIOpenDriver --help `

To execute from the build directory: `./bin/MIOpenDriver *base_arg* *layer_specific_args*`

Sample runs:
* Convoluton with search on - 
`./bin/MIOpenDriver conv -c 32 -H 8 -W 8 -k 64 -x 5 -y 5 -p 1 -q 1` 
* Forward convolution with search off - 
`./bin/MIOpenDriver conv -c 32 -H 8 -W 8 -k 64 -x 5 -y 5 -s 0 -F 1 -p 1 -q 1`
* Pooling with default parameters
`./bin/MIOpenDriver pool`
* LRN with default parameters and timing on -
`./bin/MIOpenDriver lrn -t 1`
* Printout layer specific input arguments -
`./bin/MIOpenDriver *base_arg* -?` or `./bin/MIOpenDriver *base_arg* -h (--help)`
