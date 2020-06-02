#!/usr/bin/env bash
ROCMINFO=$(which rocminfo)
SUDO=$(which sudo)
if [ -z "$ROCMINFO" ];
then
    ROCMINFO=/opt/rocm/bin/rocminfo;
fi

if [ -f $ROCMINFO ];
then
    echo "using rocminfo at $ROCMINFO"
else
    echo "rocminfo is not installed, please install the rocminfo package"
    return -1
fi
arches=$($ROCMINFO | grep -e ' gfx' -e 'Compute Unit:' | awk '/Name/{ arch= $2} /Compute Unit:/ {if(arch != "") { all_arches[(arch "-" $3)] }} END { for (a in all_arches) { print a}  }')

while IFS= read -r line ; 
do 
    if [ -f /etc/redhat-release ]; then
          echo sudo yum -y install "miopenkernels-$line"
          $SUDO yum -y install --nogpgcheck "miopenkernels-$line"
    elif [ -f /etc/lsb-release ]; then
          echo sudo apt install -y "miopenkernels-$line"
          $SUDO apt update
          $SUDO apt install -y "miopenkernels-$line"
    else
        echo "Unknown distribution"
        echo "Please install the miopenkernels-$line package using an appropriate package manager"
    fi
done <<< "$arches"
