set S=%1
set V=%2
set T=%3
cd ../../build
PATH=.\src\Debug;%PATH%
echo "CIFAR-10"
echo "conv1 100x3x32x32x32x5x5"
.\driver\Debug\MLOpenDriver.exe conv -n 100 -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "conv1 100x32x16x16x32x5x5"
.\driver\Debug\MLOpenDriver.exe conv -n 100 -W 16 -H 16 -c 32 -k 32 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "conv1 100x32x8x8x64x5x5"
.\driver\Debug\MLOpenDriver.exe conv -n 100 -W 8 -H 8 -c 32 -k 64 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "conv1 200x3x32x32x32x5x5"
rem .\driver\Debug\MLOpenDriver.exe conv -n 200 -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "conv1 200x32x16x16x32x5x5"
rem .\driver\Debug\MLOpenDriver.exe conv -n 200 -W 16 -H 16 -c 32 -k 32 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "conv1 200x32x8x8x64x5x5"
rem .\driver\Debug\MLOpenDriver.exe conv -n 200 -W 8 -H 8 -c 32 -k 64 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "conv1 1x3x32x32x32x5x5"
.\driver\Debug\MLOpenDriver.exe conv -n 1 -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "conv1 1x32x16x16x32x5x5"
.\driver\Debug\MLOpenDriver.exe conv -n 1 -W 16 -H 16 -c 32 -k 32 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
echo "conv1 1x32x8x8x64x5x5"
.\driver\Debug\MLOpenDriver.exe conv -n 1 -W 8 -H 8 -c 32 -k 64 -x 5 -y 5 -p 2 -q 2 -F 1 -t %T% -s %S% -V %V%
:end
cd ../src/OCL