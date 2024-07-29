

NVCC = nvcc
NFLAGS = -O2 --compiler-options '-fPIC'
NLDFLAGS = --shared

CC = gcc
CFLAGS = -fPIC -O3#-O0 -g -Wall -ggdb

FC = gfortran
FFLAGS = -O3#-O0 -g -Wall -ggdb



# CUDA options
CUDA_LIBS = -lcudart -lstdc++
CUDA_LIBDIR = /usr/local/nvidia/nvhpc/Linux_aarch64/22.9/cuda/lib64
CUDA_INCDIR = /usr/local/nvidia/nvhpc/Linux_aarch64/22.9/cuda/include



# Directories
SRC_DIR = src
OBJ_DIR = build
$(shell mkdir -p $(OBJ_DIR))


KERNEL_SRC = $(SRC_DIR)/utils.cu $(SRC_DIR)/init_kernel.cu $(SRC_DIR)/comp_kernel.cu $(SRC_DIR)/comm_kernel.cu
KERNEL_OBJ = $(OBJ_DIR)/utils.o $(OBJ_DIR)/init_kernel.o $(OBJ_DIR)/comp_kernel.o $(OBJ_DIR)/comm_kernel.o

C_SRC = poisson_c.c
C_OBJ = $(OBJ_DIR)/poisson_c.o

LIBCU = $(OBJ_DIR)/libpoisson_cu.so
MY_LIBS = -lpoisson_cu

F_SRC = gpu_module.F90
F_OBJ = $(OBJ_DIR)/gpu_module.o

MAIN_SRC = poisson_f.f90
MAIN_OBJ = $(OBJ_DIR)/poisson_f.o



TARGET = poisson

$(TARGET): $(LIBCU) $(F_OBJ) $(MAIN_OBJ)
	$(FC) $(MAIN_OBJ) $(F_OBJ) -o $(TARGET) $(CUDA_LIBS) $(MY_LIBS) -L$(CUDA_LIBDIR) -L$(OBJ_DIR) -Wl,-rpath,$(OBJ_DIR)

$(LIBCU): $(KERNEL_OBJ) $(C_OBJ)
	$(NVCC) $(NFLAGS) $(NLDFLAGS) -o $(LIBCU) $(KERNEL_OBJ) $(C_OBJ)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NFLAGS) -c $< -o $@

$(C_OBJ): $(C_SRC)
	$(CC) $(CFLAGS) -I$(CUDA_INCDIR) -I$(SRC_DIR) -c $(C_SRC) -o $(C_OBJ)

$(F_OBJ): $(F_SRC)
	$(FC) $(FFLAGS) -c $(F_SRC) -o $(F_OBJ) -J$(OBJ_DIR)

$(MAIN_OBJ): $(MAIN_SRC)
	$(FC) $(FFLAGS) -c $(MAIN_SRC) -o $(MAIN_OBJ) -J$(OBJ_DIR)

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*.so $(OBJ_DIR)/*.mod $(TARGET)



