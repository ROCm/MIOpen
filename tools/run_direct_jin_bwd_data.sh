#!/bin/bash

# Start time measurement
START_TIME=$(date +%s)

./MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 1344 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1344 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 960 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 960 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 480 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 2304 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 2304 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 3840 -H 7 -W 7 -k 3840 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 3840 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 2304 -H 7 -W 7 -k 2304 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 2304 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 960 -H 14 -W 14 -k 960 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 960 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 288 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 64 -H 112 -W 112 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 64 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 32 -H 112 -W 112 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 32 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 1344 -H 14 -W 14 -k 1344 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 1344 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 480 -H 28 -W 28 -k 480 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 480 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 288 -H 56 -W 56 -k 288 -y 5 -x 5 -p 2 -q 2 -u 2 -v 2 -l 1 -j 1 -m conv -g 288 -F 2 -t 1
./MIOpenDriver convfp16 -n 128 -c 192 -H 112 -W 112 -k 192 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 192 -F 2 -t 1
# End time measurement
END_TIME=$(date +%s)

# Calculate the elapsed time
ELAPSED_TIME=$((END_TIME - START_TIME))

# Output the elapsed time in seconds
echo "Elapsed time: $ELAPSED_TIME seconds"

