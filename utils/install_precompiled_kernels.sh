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
    arch=$(echo $line | awk -F"-" '{print $1}')
    num_cu=$(echo $line | awk -F"-" '{print $2}')
    if [[ $arch = gfx1* ]]; then
        echo "using gfx1* architects ..."
        num_cu=$((num_cu/2))
    fi
    package=$arch-$num_cu
    if [ -f /etc/redhat-release ]; then
          echo sudo yum -y install "miopenkernels-${package}kdb"
          $SUDO yum -y install --nogpgcheck "miopenkernels-${package}kdb"
    elif [ -f /etc/lsb-release ]; then
          echo sudo apt install -y "miopenkernels-${package}kdb"
          $SUDO apt update
          $SUDO apt install -y "miopenkernels-${package}kdb"
    else
        echo "Unknown distribution"
        echo "Please install the miopenkernels-${package}kdb package using an appropriate package manager"
    fi
done <<< "$arches"
