#/bin/sh -x

port=28001
mode=agg99


for size in 100 80 60 40 30 20 10; do

python benchmark_serving_overload_o1.py --dataset ./ShareGPT-o1.json --tokenizer ./models/llama2-70b-chat --num-prompt 1900 --num-clients ${size} --sla-ttft 10 --sla-mtpot 1.5 --port ${port} --re-adding-failed-reqs 2>&1 | tee ./client-log-results/overload-7b-${mode}-o1-${size}-10-1.5.log
python benchmark_serving_overload_long3_2048-4096_32-4096_rand.py --tokenizer ./models/llama2-70b-chat --num-prompt 500 --num-clients ${size} --sla-ttft 10 --sla-mtpot 1.5 --port ${port} --re-adding-failed-reqs 2>&1 | tee ./client-log-results/overload-7b-${mode}-long3-${size}-10-1.5.log
python benchmark_serving_overload_long2_3000-5000_3000-5000_rand.py --tokenizer ./models/llama2-70b-chat --num-prompt 500 --num-clients ${size} --sla-ttft 10 --sla-mtpot 1.5 --port ${port} --re-adding-failed-reqs 2>&1 | tee ./client-log-results/overload-7b-${mode}-long2-${size}-10-1.5.log
python benchmark_serving_overload_32-4096_2048-4096_rand.py --tokenizer ./models/llama2-70b-chat --num-prompt 500 --num-clients ${size} --sla-ttft 10 --sla-mtpot 1.5 --port ${port} --re-adding-failed-reqs 2>&1 | tee ./client-log-results/overload-7b-${mode}-long1-${size}-10-1.5.log

done
