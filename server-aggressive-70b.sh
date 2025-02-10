#/bin/sh -x

# export CUDA_VISIBLE_DEVICES=4,5,6,7

port=28003
nccl_port=18003

cd lightllm-server-aggressive/
python -m lightllm.server.api_server --model_dir ../models/llama2-70b-chat/ --tp 4 --max_total_token_num 550000 --batch_max_tokens 18000 --mode triton_gqa_attention --tokenizer_mode auto --max_req_total_len 18000 --max_req_input_len 12000 --future_past_scheduler --nccl_port ${nccl_port} --port ${port} --running_max_req_size=2000 --router_sla_abort_ttft=10 --router_sla_abort_tpot=4.9

