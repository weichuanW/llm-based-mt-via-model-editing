CUDA_VISIBLE_DEVICES=0 python ../intervention_genes/origin_generations.py \
    --one_shot_root '../Data/Source_Data/One_Shot_Cases/Test_set' \
    --zero_shot_root '../Data/Source_Data/Zero_Shot_Cases/Test_set' \
    --lang_pairs 'en-zh' \
    --subset_root_save '../Detect_Data/genes' \
    --subset_eval_save '../Detect_Data/evals' \
    --model_name 'llama2-7b-hf' \
    --device 'cuda' \
    --batch_size 2 \
    --max_token 400 \
    --heads_type 'llama2-7b-ori-genes' \
    --file_type 'Test'