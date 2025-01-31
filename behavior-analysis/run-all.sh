#!/bin/bash

echo "Running all schedulers for dist1-3"

echo -e "\n\n\n"
echo ">>> Scheduler: aggressive"
python ./scheduler_aggressive_dist1-3.py

echo -e "\n\n\n"
echo ">>> Scheduler: conservative"
python ./scheduler_conservative_dist1-3.py

echo -e "\n\n\n"
echo ">>> Scheduler: pastfuture"
python ./scheduler_pastfuture_dist1-3.py

echo -e "\n\n\n"
echo ">>> Scheduler: theoretical optimum"
python ./scheduler_theoretical_optimum_dist1-3.py
