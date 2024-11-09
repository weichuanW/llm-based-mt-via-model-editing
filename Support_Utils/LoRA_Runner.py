import argparse
import os
import sys


from LoRA_Trainer import PEFTTrainer

'''
D: arguments for running
-- detailed description:
    --running_descrip: recording event for wandb [str] to accelerate the comparison and ablation. For example, LLM_zero0_size_effects means the LLM-MT 
    + experiments ablation experiments on the size effects (model type, model bits, epoches). In this part, you can give an implicit one for initial test or
    + a detailed one for quick analysis
    
    --record_name: the name for wandb logging under the event and the local model name (after tuning) [str], which should include the detailed paramters for the running. For example, LLM_zero0_size_effects
    + llama2-7b_4_wmt22test_1_5, means the llama2-7b model with 4 bit on wmt22test dataset with 1 epoches and neptune noise 5 
    + !Notice that you should clarify this part by yourself
    
    --base_model_name: the name of the base large language model [str] (include the absolute local path and name) For example, 
    + /mnt/sgnfsdata/tolo-02-95/weicwang/data/LLM_MT/Models/llama2-7b-hf
    
    --mode: the mode of the peft tuning [str] (default is 8) For example, 8
    
    --device_map: the device type for training [str(default is cuda)] For example, cuda
    
    --batch_size: the batch size for training [int(default is 4)] For example, 4
    
    --gradient_accumulation_steps: the gradient accumulation steps for training [int(default is 4)] For example, 4
    
    --learning_rate: the learning rate for training [float(default is 2e-4)] For example, 2e-4
    
    --logging_steps: the logging steps for training [int(default is 20, we recommend the 1/50 of the overall training)] For example, 10
    
    --epochs: the epoches for training [int(default is 1)] For example, 1
    
    --save_num: the number of checkpoints to save [int(default is 1)] For example, 1
    
    --save_frequency: the frequency of saving checkpoints [float(default is 0.2)] For example, 0.2
    
    --neptune_noise: the neptune noise for training [int(default is 5)] For example, 5
    
    --cache_path: the cache path for the dataset [str(default is '')] For example, /mnt/sgnfsdata/tolo-02-95/weicwang/data/LLM_MT/Cache/
    
    --output_dir: the output dir for the peft model storage [str(default is '')] For example, /mnt/sgnfsdata/tolo-02-95/weicwang/data/LLM_MT/Models/
    
    --save_name: the save name for the specific training [str(default is '')] (include the abs path) For example, llama2-7b_4_wmt22test_1_5
    
    --dataset_name: the dataset name for training [str(default is '')] (include the abs path) For example, /mnt/sgnfsdata/tolo-02-95/weicwang/data/LLM_MT/Train/wmt22test_zero0_train.json    
'''
def parse_arguments():
    parser = argparse.ArgumentParser(description='Your script description here.')

    # Add positional arguments
    parser.add_argument('--running_descrip', type=str, help='Description of the running')
    parser.add_argument('--record_name', type=str, help='Name of the record')
    parser.add_argument('--base_model_name', type=str, help='Name of the base model')
    parser.add_argument('--mode', type=str, default='8', help='Mode information (default: 8)')

    # Set default value for device_map to "cuda"
    parser.add_argument('--device_map', type=str, default='cuda', help='Device map (default: cuda)')

    # Set default value for batch_size to 4
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')

    # Add optional arguments
    parser.add_argument('--cache_path', type=str, default='', help='Path for caching (default: '')')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory (default: '')')
    parser.add_argument('--save_name', type=str, default='', help='Save name (default: '')')
    parser.add_argument('--dataset_name', type=str, default='', help='Name of the dataset')

    # Additional optional arguments
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate (default: 2e-4)')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps (default: 10)')
    # Replace max_steps with epochs, save_num, and save_frequency
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs (default: 1)')
    parser.add_argument('--save_num', type=int, default=1, help='Number of checkpoints to save (default: 1)')
    parser.add_argument('--save_frequency', type=float, default=0.2,
                        help='Save a checkpoint every n epochs (default: 0.2)')


    parser.add_argument('--neptune_noise', type=int, default=5, help='Neptune noise (default: 5)')
    parser.add_argument('--type', type=str, default='peft', help='Max steps (default: 500)')

    # Parse the command-line arguments
    args = parser.parse_args()

    return args

# Example usage:
if __name__ == "__main__":

    args = parse_arguments()
    trainer = PEFTTrainer()
    trainer._trainer(args.base_model_name, args.mode, args.cache_path, args.output_dir, args.record_name, args.dataset_name, args.device_map, args.batch_size, args.gradient_accumulation_steps, args.learning_rate, args.logging_steps, args.save_num, args.epochs, args.save_frequency, args.neptune_noise, args.type)
