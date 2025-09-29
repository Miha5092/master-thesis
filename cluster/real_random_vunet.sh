#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=gpgpu,gpgpuB,gpgpuM
#SBATCH --job-name=VUNet_corrupted_file
export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

srun python -m experiments.real_dataset_evaluation --model_types vunet --epochs 500 --experiment_name real_random_time_gaussian_full_0 --noise time_gaussian --full_noise --seed 0