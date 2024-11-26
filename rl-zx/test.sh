#!/bin/bash
#SBATCH --job-name=ibm_pretraining
#SBATCH --chdir=.
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mincpus=12

srun python3 ppo_async.py --exp-name ibm_pretraining --learning-rate 1e-3 --total-timesteps 20000000 --num-envs=12 --anneal-lr True --update-epochs 8 --max-grad-norm 0.5 --num-steps 512 --num-minibatches 32 --vf-coef 0.5 --ent-coef 0.01 --clip-vloss True --clip-coef 0.1 --gamma 0.995 --gae-lambda 0.95