#!/bin/bash

source ../.venv/bin/activate 

K_VALUES=(2 3 4 5)

for k in "${K_VALUES[@]}"; do
    TEST_ITERATION="4/${k}"
    echo "Running with K_TOURNAMENT=${k}, TEST_ITERATION=${TEST_ITERATION}"
    
    TEST_ITERATION=$TEST_ITERATION K_TOURNAMENT=$k python main.py
done