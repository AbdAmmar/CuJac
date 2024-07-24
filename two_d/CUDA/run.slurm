#!/bin/bash
#SBATCH -J jacobi
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH -p small 
#SBATCH --exclusive 


module purge
module load openmpi/nvidia/4.1.6-opt
module load arm-forge/22.1.4
#module list

#echo "=========================="
#echo "Compile poisson"
#make clean
#make
#echo


nvcc --version
nvidia-smi

#export CUDA_VISIBLE_DEVICES=0,1

nsys profile --trace=cuda,nvtx,mpi ./poisson
#time ./poisson
#mpirun -n ${SLURM_NTASKS} --map-by ppr:${SLURM_TASKS_PER_NODE}:node:PE=1 --bind-to core ./jacobi


