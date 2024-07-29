
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


KERNEL_SRC = $(wildcard $(SRC_DIR)/*.cu)
KERNEL_OBJ = $(KERNEL_SRC:$(SRC_DIR)/%.cu=$(BUILD_DIR_LOCAL)/%.o)


KERNEL_SRC_LOCAL = $(wildcard $(SRC_DIR_LOCAL)/*.cu)
KERNEL_OBJ_LOCAL = $(KERNEL_SRC_LOCAL:$(SRC_DIR_LOCAL)/%.cu=$(BUILD_DIR_LOCAL)/%.o)


TARGET = $(BIN_DIR_LOCAL)/poisson

$(TARGET): $(KERNEL_OBJ) $(KERNEL_OBJ_LOCAL)
	$(NVCC) $(NFLAGS) $^ -o $@

$(BUILD_DIR_LOCAL)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NFLAGS) -c $< -o $@ -I$(INC_DIR)

$(BUILD_DIR_LOCAL)/%.o: $(SRC_DIR_LOCAL)/%.cu
	$(NVCC) $(NFLAGS) -c $< -o $@ -I$(INC_DIR) -I$(INC_DIR_LOCAL)


.PHONY: clean
clean:
	rm $(BUILD_DIR_LOCAL)/*.o $(BIN_DIR_LOCAL)/poisson
