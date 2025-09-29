#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=gpgpu,gpgpuB,gpgpuM
#SBATCH --job-name=ViT_t1

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

python -m src.usage.vitae_hyper -n 10 -es --model_type unet --epochs 400 --train_final_model --scaling_type min-max --sensor_type real --experiment_name t1