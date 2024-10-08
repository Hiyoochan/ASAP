#!/bin/bash
#SBATCH -J joint       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH -c 6          # number of cpus, for multi-thread programs
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node04


train_gengen --config /home/guoqingzhang/jy_project/MAB/resources/train_tar_aux_unet.yaml
test_gengen --config /home/guoqingzhang/jy_project/MAB/resources/test_tar_aux_unet.yaml