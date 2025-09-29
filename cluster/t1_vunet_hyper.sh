#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=gpgpu,gpgpuB,gpgpuM
#SBATCH --job-name=VUnet_t1

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

python -m src.usage.vcnn_hyper -n 10 -es --model_type tiny --epochs 400 --train_final_model --scaling_type min-max --sensor_type real --experiment_name t1