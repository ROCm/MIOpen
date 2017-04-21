rem arguments:
rem D - MiOpenDriver location.
rem V - turn on/off the verification.
set D=%1
set V=%2
echo "FullConvNet"
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 480 -H 48 -c 1 -n 16 -k 16 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 240 -H 24 -c 16 -n 16 -k 32 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 120 -H 12 -c 32 -n 16 -k 64 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 60 -H 6 -c 64 -n 16 -k 128 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 108 -H 108 -c 3 -n 8 -k 64 -x 3 -y 3 -p 1 -q 1 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 54 -H 54 -c 64 -n 8 -k 64 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 27 -H 27 -c 128 -n 8 -k 128 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 14 -H 14 -c 128 -n 8 -k 256 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 7 -H 7 -c 256 -n 8 -k 512 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 224 -H 224 -c 3 -n 8 -k 64 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 112 -H 112 -c 64 -n 8 -k 128 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 56 -H 56 -c 128 -n 8 -k 256 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 28 -H 28 -c 256 -n 8 -k 512 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 14 -H 14 -c 512 -n 8 -k 512 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 7 -H 7 -c 512 -n 8 -k 512 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 224 -H 224 -c 3 -n 16 -k 64 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 112 -H 112 -c 64 -n 16 -k 128 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 56 -H 56 -c 128 -n 16 -k 256 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 28 -H 28 -c 256 -n 16 -k 512 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 14 -H 14 -c 512 -n 16 -k 512 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 7 -H 7 -c 512 -n 16 -k 512 -x 3 -y 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 224 -H 224 -c 3 -n 16 -k 64 -x 7 -y 7 -p 3 -q 3 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 28 -H 28 -c 192 -n 16 -k 32 -x 5 -y 5 -p 2 -q 2 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 28 -H 28 -c 192 -n 16 -k 64 -x 1 -y 1 -p 0 -q 0 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 14 -H 14 -c 512 -n 16 -k 48 -x 5 -y 5 -p 2 -q 2 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 14 -H 14 -c 512 -n 16 -k 192 -x 1 -y 1 -p 0 -q 0 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 7 -H 7 -c 832 -n 16 -k 256 -x 1 -y 1 -p 0 -q 0 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 7 -H 7 -c 832 -n 16 -k 128 -x 5 -y 5 -p 2 -q 2 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 700 -H 161 -c 1 -n 4 -k 32 -y 5  -x 20 -p 0 -q 0 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 700 -H 161 -c 1 -n 8 -k 32 -y 5  -x 20 -p 0 -q 0 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 700 -H 161 -c 1 -n 16 -k 32 -y 5 -x 20 -p 0 -q 0 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 700 -H 161 -c 1 -n 32 -k 32 -y 5 -x 20 -p 0 -q 0 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 341 -H 79 -c 32 -n 4 -k 32 -y 5 -x 10 -p 0 -q 0 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 341 -H 79 -c 32 -n 8 -k 32 -y 5 -x 10 -p 0 -q 0 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 341 -H 79 -c 32 -n 16 -k 32 -y 5 -x 10 -p 0 -q 0 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 341 -H 79 -c 32 -n 32 -k 32 -y 5 -x 10 -p 0 -q 0 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 480 -H 48 -c 1 -n 16 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 240 -H 24 -c 16 -n 16 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 120 -H 12 -c 32 -n 16 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 60 -H 6 -c 64 -n 16 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 108 -H 108 -c 3 -n 8 -k 64 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 54 -H 54 -c 64 -n 8 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 27 -H 27 -c 128 -n 8 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 14 -H 14 -c 128 -n 8 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 7 -H 7 -c 256 -n 8 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 224 -H 224 -c 3 -n 8 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 112 -H 112 -c 64 -n 8 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 56 -H 56 -c 128 -n 8 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 28 -H 28 -c 256 -n 8 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 14 -H 14 -c 512 -n 8 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 7 -H 7 -c 512 -n 8 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 224 -H 224 -c 3 -n 16 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 112 -H 112 -c 64 -n 16 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 56 -H 56 -c 128 -n 16 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 28 -H 28 -c 256 -n 16 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 14 -H 14 -c 512 -n 16 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 7 -H 7 -c 512 -n 16 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 224 -H 224 -c 3 -n 16 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2
%D%\MIOpenDriver conv  -s 0 -V %V% -t 1 -F 1 -W 28 -H 28 -c 192 -n 16 -k 32 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1

