#!/bin/bash
# Get all filenames in current dir
for name in *
do
    # chain two commands using ';'
    # one command to remove commas
    # and another to remove fullstops
    sed -i 's/,//g;s/\.//g' ${name}
done

