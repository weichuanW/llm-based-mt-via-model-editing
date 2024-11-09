CUDA_VISIBLE_DEVICES=0 python ../intervention_genes/fv_mt_heads_intervention.py \
    --one_shot_root '../Data/Source_Data/One_Shot_Cases/Test_set' \
    --zero_shot_root '../Data/Source_Data/Zero_Shot_Cases/Test_set' \
    --lang_pairs 'en-zh' \
    --subset_root_save '../Detect_Data/genes' \
    --subset_eval_save '../Detect_Data/evals' \
    --model_name 'llama2-7b-hf' \
    --device 'cuda' \
    --batch_size 2 \
    --max_token 400 \
    --task_heads '../Data/FV_Task_overlap/overlapped_heads.json' \
    --heads_type 'fv-mt-heads' \
    --top_k 12 \
    --inter_type 'mt_heads' \
    --operation 'add' \
    --mean_head_root '../cache' \
    --file_type 'Test'

CUDA_VISIBLE_DEVICES=0 python ../intervention_genes/fv_mt_heads_intervention.py \
    --one_shot_root '../Data/Source_Data/One_Shot_Cases/Test_set' \
    --zero_shot_root '../Data/Source_Data/Zero_Shot_Cases/Test_set' \
    --lang_pairs 'en-zh' \
    --subset_root_save '../Detect_Data/genes' \
    --subset_eval_save '../Detect_Data/evals' \
    --model_name 'llama2-7b-hf' \
    --device 'cuda' \
    --batch_size 2 \
    --max_token 400 \
    --task_heads '../Data/FV_Task_overlap/overlapped_heads.json' \
    --heads_type 'fv' \
    --top_k 12 \
    --inter_type 'fv' \
    --operation 'add' \
    --mean_head_root '../cache' \
    --file_type 'Test'



