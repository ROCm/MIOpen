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
backend="hip"

while IFS= read -r line ; 
do
    arch=$(echo $line | awk -F"-" '{print $1}')
    package=$arch
    if [ -f /etc/redhat-release ]; then
          echo sudo yum -y install "miopen-${backend}-${package}kdb"
          $SUDO yum -y install --nogpgcheck "miopen-${backend}-${package}kdb"
    elif [ -f /etc/lsb-release ]; then
          echo sudo apt install -y "miopen-${backend}-${package}kdb"
          $SUDO apt update
          $SUDO apt install -y "miopen-${backend}-${package}kdb"
    else
        echo "Unknown distribution"
        echo "Please install the miopen-${backend}-${package}kdb package using an appropriate package manager"
    fi
done <<< "$arches"
