#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8   # 8 processor core(s) per node 
#SBATCH --gres=gpu:1
#SBATCH --exclude=amp-1
#SBATCH --output="results/job-%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="results/job-%j.err" # job standard error file (%j replaced by job id)

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load miniconda3
module load cuda
source activate /home/benjis/benjis/weile-lab/envs/sadl_rl
cd /home/benjis/benjis/weile-lab/sadl/rl-baselines3-zoo
export PYTHONPATH=$PWD/..
python train.py --algo dqn --env PongNoFrameskip-v4 -tb results/zoo --log-folder results/zoo/logs --eval-freq 10000 --eval-episodes 10 --save-freq 100000

