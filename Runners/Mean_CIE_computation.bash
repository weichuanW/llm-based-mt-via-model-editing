CUDA_VISIBLE_DEVICES=0 python ../Model_Editing_Adaptation/fv_task_adaptation/src/compute_indirect_effect_adaptation_archived.py \
    --dataset_name 'mt_de-en,mt_en-de,mt_en-zh,mt_zh-en' \
    --model_name 'llama2-7b-hf'\
    --root_data_dir '../Model_Editing_Adaptation/fv_task_adaptation/dataset_files'\
    --save_path_root '../cache' \
    --seed 42 \
    --n_shots 10 \
    --n_trials_ama 1000 \
    --n_trials_cie 25 \
    --test_split 0.3 \
    --device cuda \
    --last_token_only True \
    --mean_activations_path '../cache' \
    --format 'qa'
    #--prefixes '{"input":"Q:", "output":"A:", "instructions":""}' \
    #--separators '{"input":"\n", "output":"\n", "instructions":""}' \
