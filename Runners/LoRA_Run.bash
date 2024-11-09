# model_mode_set_maxstep_nep
CUDA_VISIBLE_DEVICES=0 nohup python ../Support_Utils/LoRA_Runner.py \
    --record_name "llama2-7b_4_wmt22test_1_5" \
    --base_model_name "llama2-7b-hf" \
    --mode "4" \
    --device_map "cuda" \
    --batch_size 4 \
    --cache_path "" \
    --output_dir "../LLM_PEFT" \
    --dataset_name "../Data/LoRA_train/lora_train.json" \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --save_num 1 \
    --epochs 1 \
    --save_frequency 0.2 \
    --neptune_noise 5 \
    >> /mnt/sgnfsdata/tolo-02-95/weicwang/data/local_logging/llama2-7b_4_wmt22test_1_5.out  &

CUDA_VISIBLE_DEVICES=1 nohup python ../Support_Utils/LoRA_Runner.py \
    --running_descrip "peft training on llama2 7b with 4 bit on wmt22 testset for 1 maxstep and netune with 5" \
    --record_name "llama2-7b_8_wmt22test_1_5" \
    --base_model_name "llama2-7b-hf" \
    --mode "8" \
    --device_map "cuda" \
    --batch_size 4 \
    --cache_path "" \
    --output_dir "../LLM_PEFT" \
    --save_name "llama2-7b_8_wmt22test_1_5" \
    --dataset_name "../Data/LoRA_train/lora_train.json" \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --save_num 1 \
    --epochs 1 \
    --save_frequency 0.2 \
    --neptune_noise 5 \
    >> /mnt/sgnfsdata/tolo-02-95/weicwang/data/local_logging/llama2-7b_8_wmt22test_1_5.out  &

CUDA_VISIBLE_DEVICES=2 nohup python Running.py \
    --running_descrip "peft training on llama2 7b with 4 bit on wmt22 testset for 1 maxstep and netune with 5" \
    --record_name "llama2-13b_4_wmt22test_1_5" \
    --base_model_name "/home/weichuanw/95server/model/llama2-13b-hf" \
    --mode "4" \
    --device_map "cuda" \
    --batch_size 2 \
    --cache_path "" \
    --output_dir "../LLM_PEFT" \
    --save_name "llama2-13b_4_wmt22test_1_5" \
    --dataset_name "../Data/LoRA_train/lora_train.json" \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --save_num 1 \
    --epochs 1 \
    --save_frequency 0.2 \
    --neptune_noise 5 \
    >> /mnt/sgnfsdata/tolo-02-95/weicwang/data/local_logging/llama2-13b_4_wmt22test_1_5.out  &

CUDA_VISIBLE_DEVICES=3 nohup python Running.py \
    --running_descrip "peft training on llama2 7b with 4 bit on wmt22 testset for 1 maxstep and netune with 5" \
    --record_name "llama2-13b_8_wmt22test_1_5" \
    --base_model_name "/home/weichuanw/95server/model/llama2-13b-hf" \
    --mode "8" \
    --device_map "cuda" \
    --batch_size 2 \
    --cache_path "" \
    --output_dir "../LLM_PEFT" \
    --save_name "llama2-13b_8_wmt22test_1_5" \
    --dataset_name "../Data/LoRA_train/lora_train.json" \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --save_num 1 \
    --neptune_noise 5 \
    >> /mnt/sgnfsdata/tolo-02-95/weicwang/data/local_logging/llama2-13b_8_wmt22test_1_5.out &

wait
sleep 60

CUDA_VISIBLE_DEVICES=0 nohup python Running.py \
    --running_descrip "peft training on llama2 7b with 4 bit on wmt22 testset for 1 maxstep and netune with 5" \
    --record_name "llama2-7b_4_wmt22test_2_5" \
    --base_model_name "llama2-7b-hf" \
    --mode "4" \
    --device_map "cuda" \
    --batch_size 4 \
    --cache_path "" \
    --output_dir "../LLM_PEFT" \
    --save_name "llama2-7b_4_wmt22test_2_5" \
    --dataset_name "../Data/LoRA_train/lora_train.json" \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --save_num 1 \
    --epochs 2 \
    --save_frequency 0.2 \
    --neptune_noise 5 \
    >> /mnt/sgnfsdata/tolo-02-95/weicwang/data/local_logging/llama2-7b_4_wmt22test_2_5.out  &

CUDA_VISIBLE_DEVICES=1 nohup python Running.py \
    --running_descrip "peft training on llama2 7b with 4 bit on wmt22 testset for 1 maxstep and netune with 5" \
    --record_name "llama2-7b_8_wmt22test_2_5" \
    --base_model_name "llama2-7b-hf" \
    --mode "8" \
    --device_map "cuda" \
    --batch_size 4 \
    --cache_path "" \
    --output_dir "../LLM_PEFT" \
    --save_name "llama2-7b_8_wmt22test_2_5" \
    --dataset_name "../Data/LoRA_train/lora_train.json" \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --save_num 1 \
    --epochs 2 \
    --save_frequency 0.2 \
    --neptune_noise 5 \
    >> /mnt/sgnfsdata/tolo-02-95/weicwang/data/local_logging/llama2-7b_8_wmt22test_2_5.out  &

CUDA_VISIBLE_DEVICES=2 nohup python Running.py \
    --running_descrip "peft training on llama2 7b with 4 bit on wmt22 testset for 1 maxstep and netune with 5" \
    --record_name "llama2-13b_4_wmt22test_2_5" \
    --base_model_name "/home/weichuanw/95server/model/llama2-13b-hf" \
    --mode "4" \
    --device_map "cuda" \
    --batch_size 2 \
    --cache_path "" \
    --output_dir "../LLM_PEFT" \
    --save_name "llama2-13b_4_wmt22test_2_5" \
    --dataset_name "../Data/LoRA_train/lora_train.json" \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --save_num 1 \
    --epochs 2 \
    --save_frequency 0.2 \
    --neptune_noise 5 \
    >> /mnt/sgnfsdata/tolo-02-95/weicwang/data/local_logging/llama2-13b_4_wmt22test_2_5.out  &

CUDA_VISIBLE_DEVICES=3 nohup python Running.py \
    --running_descrip "peft training on llama2 13b with 8 bit on wmt22 testset for 1 maxstep and netune with 5" \
    --record_name "llama2-13b_8_wmt22test_2_5" \
    --base_model_name "/home/weichuanw/95server/model/llama2-13b-hf" \
    --mode "8" \
    --device_map "cuda" \
    --batch_size 2 \
    --cache_path "" \
    --output_dir "../LLM_PEFT" \
    --save_name "llama2-13b_8_wmt22test_2_5" \
    --dataset_name "../Data/LoRA_train/lora_train.json" \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --save_num 1 \
    --epochs 2 \
    --save_frequency 0.2 \
    --neptune_noise 5 \
    >> /mnt/sgnfsdata/tolo-02-95/weicwang/data/local_logging/llama2-13b_8_wmt22test_2_5.out  &

wait  # Wait for the first command to finish
sleep 60  # Sleep for 60 seconds


