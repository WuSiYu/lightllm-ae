#!/bin/bash

echo "Gathering results from all clients..."
echo "Each log file will cossponding a datapoint in the fig7"
echo "---"
grep -a ghput client-log-results/* | tee /tmp/lightllm_all_result.txt

echo "---"
echo "Plotting the results..."
cat /tmp/lightllm_all_result.txt | python3 fig7.py
