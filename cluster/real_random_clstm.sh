#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=gpgpu,gpgpuB
#SBATCH --job-name=CLSTM_real_random_0

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

srun python -m experiments.real_dataset_evaluation --model_types clstm --epochs 500 --experiment_name real_random_0 --noise none --full_noise --seed 0