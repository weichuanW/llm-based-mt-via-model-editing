CUDA_VISIBLE_DEVICES=0 python ../Support_Utils/LoRA_Runner.py \
    --record_name "llama2-7b_4_wmt22test_1_5" \
    --base_model_name "/mnt/sgnfsdata/tolo-02-95/weicwang/model/llama2-7b-hf" \
    --mode "4" \
    --device_map "cuda" \
    --batch_size 4 \
    --cache_path "" \
    --output_dir "/mnt/sgnfsdata/tolo-02-95/weicwang/model/LLM_PEFT" \
    --dataset_name "/mnt/sgnfsdata/tolo-02-95/weicwang/data/LLM_MT/PEFT/wmt22-news-systems_zero0_train.json" \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --save_num 1 \
    --epochs 1 \
    --save_frequency 0.2 \
    --neptune_noise 5.0 \
    >> /mnt/sgnfsdata/tolo-02-95/weicwang/data/local_logging/llama2-7b_4_wmt22test_1_5.out  &