
NVCC = nvcc
NFLAGS = -O2#-O0 -g -Wall -ggdb

PATH_CU = $(CuJac_PATH)/two_d

SRC_DIR = $(PATH_CU)/src_cu
INC_DIR = $(PATH_CU)/include_cu

SRC_DIR_LOCAL = src
INC_DIR_LOCAL = include
BIN_DIR_LOCAL = bin
BUILD_DIR_LOCAL = build
$(shell mkdir -p $(BIN_DIR_LOCAL))
$(shell mkdir -p $(BUILD_DIR_LOCAL))


#KERNEL_SRC = $(wildcard $(SRC_DIR)/*.cu)
#KERNEL_OBJ = $(KERNEL_SRC:$(SRC_DIR)/%.cu=$(BUILD_DIR_LOCAL)/%.o)

#KERNEL_SRC_LOCAL = $(wildcard $(SRC_DIR_LOCAL)/*.cu)
#KERNEL_OBJ_LOCAL = $(KERNEL_SRC_LOCAL:$(SRC_DIR_LOCAL)/%.cu=$(BUILD_DIR_LOCAL)/%.o)

TARGET1 = poisson
TARGET2 = poisson_2d



all: $(TARGET1) $(TARGET2)



$(TARGET1): $(SRC_DIR)/utils.cu $(SRC_DIR)/comp_kernel.cu $(SRC_DIR)/init_kernel.cu $(SRC_DIR_LOCAL)/poisson.cu
	$(NVCC) $(NFLAGS) $^ -o $(BIN_DIR_LOCAL)/$(TARGET1) -I$(INC_DIR)
$(SRC_DIR)/comp_kernel.o: $(SRC_DIR)/comp_kernel.cu
	$(NVCC) $(NFLAGS) -c $< -o $@
$(BUILD_DIR_LOCAL)/poisson.o: $(SRC_DIR_LOCAL)/poisson.cu
	$(NVCC) $(NFLAGS) -c $< -o $@ -I$(INC_DIR) -I$(INC_DIR_LOCAL)
$(SRC_DIR)/init_kernel.o: $(SRC_DIR)/init_kernel.cu
	$(NVCC) $(NFLAGS) -c $< -o $@


$(TARGET2): $(SRC_DIR)/utils.cu $(SRC_DIR)/comp_kernel_2d.cu $(SRC_DIR)/init_kernel_2d.cu $(SRC_DIR_LOCAL)/poisson_2d.cu
	$(NVCC) $(NFLAGS) $^ -o $(BIN_DIR_LOCAL)/$(TARGET2) -I$(INC_DIR)
$(SRC_DIR)/comp_kernel_2d.o: $(SRC_DIR)/comp_kernel_2d.cu
	$(NVCC) $(NFLAGS) -c $< -o $@
$(BUILD_DIR_LOCAL)/poisson_2d.o: $(SRC_DIR_LOCAL)/poisson_2d.cu
	$(NVCC) $(NFLAGS) -c $< -o $@ -I$(INC_DIR) -I$(INC_DIR_LOCAL)
$(SRC_DIR)/init_kernel_2d.o: $(SRC_DIR)/init_kernel_2d.cu
	$(NVCC) $(NFLAGS) -c $< -o $@


$(SRC_DIR)/utils.o: $(SRC_DIR)/utils.cu
	$(NVCC) $(NFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -f $(BUILD_DIR_LOCAL)/*.o $(BIN_DIR_LOCAL)/*
