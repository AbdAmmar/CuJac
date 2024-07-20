#!/bin/bash
#SBATCH -J jacobi
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:1:00
#SBATCH -p small 
#SBATCH --exclusive 


echo
echo "=========================="
echo "Load environment"
module purge
module load openmpi/nvidia/4.1.6-opt
echo
module list
echo

echo "=========================="
echo "Compile poisson"
make clean
make
echo


echo "=========================="
echo "Launch application"
#export CUDA_VISIBLE_DEVICES=0,1

./poisson
#mpirun -n ${SLURM_NTASKS} --map-by ppr:${SLURM_TASKS_PER_NODE}:node:PE=1 --bind-to core ./jacobi

echo
echo "end job"

