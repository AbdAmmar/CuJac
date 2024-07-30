
NVCC = nvcc
NFLAGS = -O2 --compiler-options '-fPIC'
NLDFLAGS = --shared


CC = gcc
CFLAGS = -fPIC -O3#-O0 -g -Wall -ggdb

FC = gfortran
FFLAGS = -O3#-O0 -g -Wall -ggdb



PATH_CU = $(CuJac_PATH)/two_d

SRC_DIR = $(PATH_CU)/src_cu
INC_DIR = $(PATH_CU)/include_cu

SRC_DIR_LOCAL = src
INC_DIR_LOCAL = include
BIN_DIR_LOCAL = bin
BUILD_DIR_LOCAL = build
$(shell mkdir -p $(BIN_DIR_LOCAL))
$(shell mkdir -p $(BUILD_DIR_LOCAL))


KERNEL_SRC = $(wildcard $(SRC_DIR)/*.cu)
KERNEL_OBJ = $(KERNEL_SRC:$(SRC_DIR)/%.cu=$(BUILD_DIR_LOCAL)/%.o)

C_SRC_LOCAL = $(wildcard $(SRC_DIR_LOCAL)/*.c)
C_OBJ_LOCAL = $(C_SRC_LOCAL:$(SRC_DIR_LOCAL)/%.c=$(BUILD_DIR_LOCAL)/%.o)

LIBCU = $(BUILD_DIR_LOCAL)/libpoisson_cu.so
MY_LIBS = -lpoisson_cu

F_SRC_LOCAL = $(wildcard $(SRC_DIR_LOCAL)/*.f90)
F_OBJ_LOCAL = $(F_SRC_LOCAL:$(SRC_DIR_LOCAL)/%.f90=$(BUILD_DIR_LOCAL)/%.o)

MOD_SRC_LOCAL = $(wildcard $(SRC_DIR_LOCAL)/*.F90)
MOD_OBJ_LOCAL = $(MOD_SRC_LOCAL:$(SRC_DIR_LOCAL)/%.F90=$(BUILD_DIR_LOCAL)/%.o)



CUDA_LIBS = -lcudart

#CUDA_LIBDIR = /usr/local/nvidia/nvhpc/Linux_aarch64/22.9/cuda/lib64
#CUDA_INCDIR = /usr/local/nvidia/nvhpc/Linux_aarch64/22.9/cuda/include

CUDA_LIBDIR = /usr/local/nvidia_hpc_sdk/MAJSLURM/Linux_x86_64/22.7/cuda/11.7/lib64
CUDA_INCDIR = /usr/local/nvidia_hpc_sdk/MAJSLURM/Linux_x86_64/22.7/cuda/11.7/include


TARGET = $(BIN_DIR_LOCAL)/poisson

$(TARGET): $(LIBCU) $(MOD_OBJ_LOCAL) $(F_OBJ_LOCAL)
	#$(FC) $(FFLAGS) $(KERNEL_OBJ) $(C_OBJ_LOCAL) $(F_OBJ_LOCAL) -o $@ $(CUDA_LIBS) $(MY_LIBS) -L$(BUILD_DIR_LOCAL) -Wl,-rpath,$(BUILD_DIR_LOCAL)
	$(FC) $(FFLAGS) $(KERNEL_OBJ) $(C_OBJ_LOCAL) $(F_OBJ_LOCAL) -o $@ $(CUDA_LIBS) $(MY_LIBS) -L$(CUDA_LIBDIR) -L$(BUILD_DIR_LOCAL) -Wl,-rpath,$(BUILD_DIR_LOCAL)

$(LIBCU): $(KERNEL_OBJ) $(C_OBJ_LOCAL)
	$(NVCC) $(NFLAGS) $(NLDFLAGS) $< -o $@ 

$(BUILD_DIR_LOCAL)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NFLAGS) -c $< -o $@ -I$(INC_DIR)

$(BUILD_DIR_LOCAL)/%.o: $(SRC_DIR_LOCAL)/%.c
	#$(CC) $(CFLAGS) -c $< -o $@ -I$(INC_DIR) -I$(INC_DIR_LOCAL)
	$(CC) $(CFLAGS) -c $< -o $@ -I$(CUDA_INCDIR) -I$(INC_DIR) -I$(INC_DIR_LOCAL)

$(BUILD_DIR_LOCAL)/%.o: $(SRC_DIR_LOCAL)/%.F90
	$(FC) $(FFLAGS) -c $< -o $@ -J$(BUILD_DIR_LOCAL)

$(BUILD_DIR_LOCAL)/%.o: $(SRC_DIR_LOCAL)/%.f90
	$(FC) $(FFLAGS) -c $< -o $@ -J$(BUILD_DIR_LOCAL)



.PHONY: clean
clean:
	rm -f $(BUILD_DIR_LOCAL)/*.o $(TARGET)



