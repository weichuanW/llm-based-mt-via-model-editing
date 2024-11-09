CUDA_VISIBLE_DEVICES=0 python ../intervention_genes/kn_interventions.py \
    --one_shot_root '../Data/Source_Data/One_Shot_Cases/Test_set' \
    --zero_shot_root '../Data/Source_Data/Zero_Shot_Cases/Test_set' \
    --lang_pairs 'en-zh,en-de,de-en,zh-en' \
    --subset_root_save '../Detect_Data/genes' \
    --subset_eval_save '../Detect_Data/evals' \
    --model_name 'llama2-7b-hf' \
    --device 'cuda' \
    --batch_size 2 \
    --max_token 400 \
    --kn_type 'error_kn' \
    --filter_kn '../Data/KN_Error_overlap/top_100_overlap.json' \
    --operation 'suppress' \
    --file_type 'Test'

CUDA_VISIBLE_DEVICES=1 python ../intervention_genes/kn_interventions.py \
    --one_shot_root '../Data/Source_Data/One_Shot_Cases/Test_set' \
    --zero_shot_root '../Data/Source_Data/Zero_Shot_Cases/Test_set' \
    --lang_pairs 'en-zh,en-de,de-en,zh-en' \
    --subset_root_save '../Detect_Data/genes' \
    --subset_eval_save '../Detect_Data/evals' \
    --model_name 'llama2-7b-hf' \
    --device 'cuda' \
    --batch_size 2 \
    --max_token 400 \
    --kn_type 'task_kn' \
    --filter_kn '../Data/KN_Error_overlap/top_100_overlap.json' \
    --operation 'enhance' \
    --file_type 'Test'