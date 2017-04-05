set B=%1
set S=%2
set V=%3
set T=%4
cd ../../build
PATH=.\src\Debug;%PATH%
echo "AlexNet"
echo "conv1 10x3x227x227x96x11x11"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -c 3 -k 96 -H 227 -W 227 -x 11 -y 11 -p 5 -q 5 -u 4 -v 4 -F 1 -t %T% -s %S% -V %V%
echo "conv2 10x96x27x27x256x5x5"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 27 -H 27 -c 96 -k 256 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "conv3 10x256x13x13x384x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 13 -H 13 -c 256 -k 384 -x 3 -y 3 -p 1 -q 1 -F 1 -t %T% -s %S% -V %V%
echo "conv4 10x384x13x13x384x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 13 -H 13 -c 384 -k 384 -x 3 -y 3 -p 1 -q 1 -F 1 -t %T% -s %S% -V %V%
echo "conv5 10x384x13x13x256x3x3"
.\driver\Debug\MIOpenDriver.exe conv -i 1 -n %B% -W 13 -H 13 -c 384 -k 256 -x 3 -y 3 -p 1 -q 1 -F 1 -t %T% -s %S% -V %V%
:end
cd ../src/OCL