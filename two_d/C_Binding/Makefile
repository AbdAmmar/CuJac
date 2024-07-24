

NVCC = nvcc
NFLAG =

CC = gcc
CFLAG = -O3#-O0 -g -Wall -ggdb

# Directories
SRC_DIR = src
OBJ_DIR = obj

KERNEL_SRC = $(SRC_DIR)/utils.cu $(SRC_DIR)/init_kernel.cu $(SRC_DIR)/comp_kernel.cu $(SRC_DIR)/comm_kernel.cu
KERNEL_OBJ = $(OBJ_DIR)/utils.o $(OBJ_DIR)/init_kernel.o $(OBJ_DIR)/comp_kernel.o $(OBJ_DIR)/comm_kernel.o

MAIN_SRC = poisson.c
MAIN_OBJ = $(OBJ_DIR)/poisson.o

# Output
OUTPUT = poisson

# CUDA options
CUDA_LIBS = -lcudart
CUDA_LIBDIR = /usr/local/nvidia/nvhpc/Linux_aarch64/22.9/cuda/lib64
CUDA_INCDIR = /usr/local/nvidia/nvhpc/Linux_aarch64/22.9/cuda/include

# Create object directory if it does not exist
$(shell mkdir -p $(OBJ_DIR))

TARGET: $(OUTPUT)

$(OUTPUT): $(KERNEL_OBJ) $(MAIN_OBJ) 
	$(NVCC) $(MAIN_OBJ) $(KERNEL_OBJ) -o $(OUTPUT) $(CUDA_LIBS) -L$(CUDA_LIBDIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NFLAG) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC)
	$(CC) $(CFLAG) -I$(CUDA_INCDIR) -I$(SRC_DIR) -c $(MAIN_SRC) -o $(MAIN_OBJ)

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/*.o $(OUTPUT)



