# MIOpenDriver runner script

Bash script `run_miopen_driver.sh` can be used to run diffrent configurations recorded in the JSON config file.
The default `config.json` is located in the same folder as the runner script. One can provide also the locations of the configuration file as 
an argument to the runner script. The runner script has one required argument -- the configuration name. Other argument are optional and can found by running the 
the runner script without any arguments or with flag `--help`.

# Filtering runner script logs

Information about KTN actions can be found by searching tags 

- ModelSetParams
- RunParameterPredictionModel