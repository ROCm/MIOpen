# MIOpenDriver

The `MIOpenDriver` enables the user to test the functionality of any particular 
layer in MIOpen in both the forward and backward direction. 
MIOpenDriver can be build by typing:

```make MIOpenDriver``` from the ```build``` directory.

All the supported layers in MIOpen can be found by the supported `base_args` here:

``` ./bin/MIOpenDriver --help ```

To execute from the build directory: 

```./bin/MIOpenDriver *base_arg* *layer_specific_args*```

Or to execute the default configuration simpily run: 

```./bin/MIOpenDriver *base_arg*```

MIOpenDriver example usages:

- Convolution with search on:

```./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2```   

- Forward convolution with search off:

```./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1```  

- Pooling with default parameters:

```./bin/MIOpenDriver pool```  

- LRN with default parameters and timing on:

```./bin/MIOpenDriver lrn -t 1```

- Batch normalization with spatial fwd train, saving mean and variance tensors:

```./bin/MIOpenDriver bnorm -F 1 -n 32 -c 512 -H 16 -W 16 -m 1 -s 1```  

- RNN with forward and backwards pass, no bias, bi-directional and LSTM mode

```./bin/MIOpenDriver rnn -n 4,4,4,3,3,3,2,2,2,1 -k 10 -H 512 -W 1024 -l 3 -F 0 -b 0 -r 1 -m lstm```

- Printout layer specific input arguments:

`./bin/MIOpenDriver *base_arg* -?` **OR**  `./bin/MIOpenDriver *base_arg* -h (--help)`

