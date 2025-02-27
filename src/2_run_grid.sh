#!/bin/bash

source ../.venv/bin/activate 

POP_VALUES=(50 75 125 150 200 400)

for pop in "${POP_VALUES[@]}"; do
    TEST_ITERATION="2/${pop}"
    echo "Running with POP=${pop}"
    
    TEST_ITERATION=$TEST_ITERATION POPULATION_SIZE=$pop python main.py
done