# GPU-Accelerated Poisson's Equation Solver

This project provides a CUDA-accelerated implementation of the finite difference method to solve Poisson's equation in 2D. 
The code is optimized to run on NVIDIA GPUs, leveraging the parallel processing power of CUDA. Additionally, bindings in C 
and Fortran are provided to facilitate integration with various projects.

## Introduction

Poisson's equation is a partial differential equation describing the distribution of heat in a given region over time. 
This project uses the finite difference method to approximate the solution of Poisson's equation in two dimensions, with ongoing 
work to extend the implementation to three dimensions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
