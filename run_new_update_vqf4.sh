#!/bin/bash 
#SBATCH --job-name=vqvae_ldm         #  
#SBATCH --output=logs_new/vqvae_ldm_%j.out  #  
#SBATCH --error=logs_new/vqvae_ldm_%j.err 
#SBATCH --nodes=1                #  
#SBATCH --ntasks=1               #  
#SBATCH --cpus-per-task=4        #  
#SBATCH --gres=gpu:1             #  
#SBATCH --time=6-23:59:59        #  
#SBATCH --partition=gpu-h100        #  
#SBATCH --mem=16G

echo "Job started on $(date)" 
echo "Running on node: $(hostname)"  

export CUBLAS_WORKSPACE_CONFIG=:4096:8
python main.py --base configs/vqvae/vq-f4_new_update.yaml -t True --gpus 0 -l logs_new -n vq_new_exp_decay_100000iter_0.1r

echo "Job finished on $(date)" 