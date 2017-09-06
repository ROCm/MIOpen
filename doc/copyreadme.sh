#!/bin/bash


cp ../README.md ./src/install.md
echo "Copying over ../README.md to docs folder as ./src/install.md."

sed -e '0,/MIOpen/ s/MIOpen/Build and Install Instructions/' -i ./src/install.md
echo "Replacing section title from install.md."

cp ../driver/README.md ./src/driver.md
echo "Copying over ../driver/README.md to docs folder as ./src/driver.md."

