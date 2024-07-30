#!/bin/bash
#SBATCH -J jacobi
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00


module purge
module load cuda/11.7

nvcc --version
nvidia-smi

time ./bin/poisson
#time ./bin/poisson_2d

nvidia-smi
