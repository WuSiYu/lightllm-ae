#/bin/sh -x

# export CUDA_VISIBLE_DEVICES=1

port=28002
nccl_port=18002

cd lightllm-server-aggressive/
python -m lightllm.server.api_server --model_dir ../models/llama2-13b-chat/ --tp 1 --max_total_token_num 64000 --batch_max_tokens 18000 --tokenizer_mode auto --max_req_total_len 18000 --max_req_input_len 12000 --future_past_scheduler --nccl_port ${nccl_port} --port ${port}

