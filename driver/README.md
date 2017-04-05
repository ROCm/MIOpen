# MLOpen Driver

The driver enables to test forward/backward of any particular layer in MIOpen. 

To build the driver, type `make MLOpenDriver` from the `build` directory.

All the supported layers in MIOpen can be found by the supported `base_args` here:

` ./driver MLOpenDriver --help `

To execute from the build directory: `./driver/MLOpenDriver *base_arg* *layer_specific_args*`

Sample runs:
* Convoluton with search on - 
`./driver/MLOpenDriver conv -c 32 -H 8 -W 8 -k 64 -x 5 -y 5 -p 1 -q 1` 
* Forward convolution with search off - 
`./driver/MLOpenDriver conv -c 32 -H 8 -W 8 -k 64 -x 5 -y 5 -s 0 -F 1 -p 1 -q 1`
* Pooling with default parameters
`./driver/MLOpenDriver pool`
* LRN with default parameters and timing on -
`./driver/MLOpenDriver lrn -t 1`
* Printout layer specific input arguments -
`./driver/MLOpenDriver *base_arg* -?` or `./driver/MLOpenDriver *base_arg* -h (--help)`
