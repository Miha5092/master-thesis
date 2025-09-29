#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=gpgpu,gpgpuB
#SBATCH --job-name=Noise_Metrics

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

srun python -m add_extra_metrics