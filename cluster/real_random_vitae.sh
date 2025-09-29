#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=gpgpu,gpgpuB
#SBATCH --job-name=Vitae_small

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

srun python -m experiments.real_dataset_evaluation --model_types vitae --epochs 500 --experiment_name small_unet_gaussian_0 --noise gaussian --full_noise --seed 0

srun python -m experiments.real_dataset_evaluation --model_types vitae --epochs 500 --experiment_name small_unet_gaussian_42 --noise gaussian --full_noise --seed 42

srun python -m experiments.real_dataset_evaluation --model_types vitae --epochs 500 --experiment_name small_unet_gaussian_100 --noise gaussian --full_noise --seed 100