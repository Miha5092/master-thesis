#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mm1124
#SBATCH --partition=gpgpu,gpgpuB,gpgpuM
#SBATCH --job-name=ConvLSTM_tuning_unet_real_t6

export PATH=/vol/bitbucket/${USER}/master-thesis/.venv/bin/:$PATH

source activate

python -m src.usage.vclstm_hyper -n 20 -es --train_final_model --scaling_type min-max --epochs 200 --sensor_type real --timesteps 6 --experiment_name t6