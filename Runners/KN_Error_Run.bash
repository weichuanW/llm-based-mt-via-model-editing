CUDA_VISIBLE_DEVICES=0 python ../Model_Editing_Adaptation/KN_Error_Extraction.py \
  --model_name "llama2-7b-hf" \
  --device "cuda" \
  --cache_dir "" \
  --detect_module "KN_hook_names" \
  --total_layers 32 \
  --batch_size 10 \
  --steps 20 \
  --instance_num 2000 \
  --file_root "../Data/One_Shot_Gene" \
  --file "detect_training_gene_one_shot_en-de.json" \
  --save_root "../Detect_Data/genes" \
  --save_name "kn_error_detection" \
  --lang "en-de" \
  >> KN_task_en_de.out  &