#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=gpgpu,gpgpuB
#SBATCH --job-name=CLSTM_fine_tuning

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

# srun python -m experiments.fine_tuning --model_types clstm --epochs 500 --fine_tuning_epochs 500 --experiment_name fine_tuned_no_pre_0 --noise none --full_noise --seed 0

# srun python -m experiments.fine_tuning --model_types clstm --epochs 500 --fine_tuning_epochs 500 --experiment_name fine_tuned_no_pre_42 --noise none --full_noise --seed 42

srun python -m experiments.fine_tuning --model_types clstm --epochs 500 --fine_tuning_epochs 500 --experiment_name fine_tuned_no_pre_100 --noise none --full_noise --seed 100