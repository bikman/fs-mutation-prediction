#!/bin/bash
echo "Running ALL 39 proteins"
for (( c=1; c<=39; c++ ))
do
        python3 run_full_flow.py -ep $c
        sleep 2
done
echo "DONE!"