
NVCC = nvcc
NFLAGS = -O2#-O0 -g -Wall -ggdb

SRC_DIR = src
INC_DIR = include

BIN_DIR = bin
BUILD_DIR = build
$(shell mkdir -p $(BIN_DIR))
$(shell mkdir -p $(BUILD_DIR))

KERNEL_SRC = $(wildcard $(SRC_DIR)/*.cu)
KERNEL_OBJ = $(KERNEL_SRC:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)


TARGET = poisson

$(TARGET): $(KERNEL_OBJ)
	$(NVCC) $(NFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NFLAGS) -c $< -o $@ -I$(INC_DIR)



.PHONY: clean
clean:
	rm $(BUILD_DIR)/*.o poisson
