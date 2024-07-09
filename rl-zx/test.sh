#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mincpus 4
srun --exclusive --cpus-per-task=4 python3 test_script.py --exp-name cquere-10q-SrH --learning-rate 1e-3 --total-timesteps 10000000 --num-envs 4 --anneal-lr True --update-epochs 8 --max-grad-norm 0.5 --num-steps 1024 --num-minibatches 32 --vf-coef 0.5 --ent-coef 0.01 --clip-vloss True --clip-coef 0.1 --gamma 0.995 --gae-lambda 0.95