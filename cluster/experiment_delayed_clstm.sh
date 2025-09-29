#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=gpgpu,gpgpuB,gpgpuM
#SBATCH --job-name=CLSTM_delayed

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

python -m experiments.timestep_delayed --model_type clstm