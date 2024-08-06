# GPU-Accelerated 2D Poisson's Equation Solver

## Table of Contents
- [Introduction](#introduction)
- [usage](#usage)

## Introduction

This project addresses the 2D Poisson's equation, a common partial differential equation (PDE) used to describe heat distribution over a spatial domain. 
We focus on solving the problem using the finite difference method.

### Problem Statement

Consider the following Poisson problem:

$$
\frac{\partial^2 u(x,y)}{\partial x^2} + \frac{\partial^2 u(x,y)}{\partial y^2} = f(x,y) \quad \text{in } \Omega = [0,1]^2
$$

with boundary conditions:

$$
u(x,y) = 0 \quad \text{on } \partial \Omega.
$$

We take the source term $f(x,y)$ to be:

$$
f(x,y) = 2 \left(x^2 - x + y^2 - y\right)
$$

### Finite Difference Method

To solve this PDE, we apply the finite difference method such that the spatial domain is discretized with a grid spacing $h$.

The finite difference update formula for the solution $u_{i,j}$ at grid point $(i,j) \equiv (x_i=ih, y_i=ih)$ is given by:

$$
u_{i,j}^{(n+1)} = \frac{1}{4} \left(u_{i+1,j}^{(n)} + u_{i-1,j}^{(n)} + u_{i,j+1}^{(n)} + u_{i,j-1}^{(n)}\right) - \frac{h^2}{4} f_{i,j}
$$

where $h$ is the spatial discretization step, and $f_{i,j}$ is the value of the source term $f(x,y)$ at grid point $(i,j)$.

This algorithm is also known as the Jacobi method for updating the solution starting from a guess $u^0$.


## Usage

To run the code, follow these steps:

### 1. Source the Environment Setup Script

Before compiling, you need to source the environment setup script to configure your environment properly. 
Run the following command:

```bash
source CuJac/config/env.rc
```

### 2. **Navigate to the CUDA directory:**

```bash
cd CuJac/two_d/CUDA
```

### 3. **Compile the code to generate the binary:**

To compile the code and generate the binary executable, run the following command:

```bash
make poisson_2d
```

### 4. **Set the parameters in the configuration file:**

Before running the code, you need to configure the parameters in the `param_2d.txt` file. 
This file is located in the `CuJac/two_d/CUDA` directory. Open the file with a text editor of your choice and 
modify the parameters according to your requirements.

### 5. Run the Binary

After compiling the code, you can run the binary executable to start the simulation. Use the following command:

```bash
./bin/poisson_2d
```
