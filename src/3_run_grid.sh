#!/bin/bash

source ../.venv/bin/activate 

# MUTPB_VALUES=(0.01 0.05 0.1)
# CXPB_VALUES=(0.3 0.5 0.7)
MUTPB_VALUES=(0.1 0.15 0.2 0.25)
CXPB_VALUES=(0.3 0.4 0.5)

for mutpb in "${MUTPB_VALUES[@]}"; do
    for cxpb in "${CXPB_VALUES[@]}"; do
        TEST_ITERATION="3/${mutpb}_${cxpb}"
        echo "Running with MUTPB=${mutpb}, CXPB=${cxpb}, TEST_ITERATION=${TEST_ITERATION}"
        
        TEST_ITERATION=$TEST_ITERATION CXPB=$cxpb MUTPB=$mutpb python main.py
    done
done