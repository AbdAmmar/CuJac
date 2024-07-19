

TARGET = poisson
CC = nvcc
CFLAG = #-O0 -g -Wall -ggdb
OBJ = utils.o init_kernel.o comp_kernel.o comm_kernel.o poisson.o #solution_kernel.o

$(TARGET): $(OBJ)
	$(CC) -o $(TARGET) $(OBJ)

%.o: %.cu
	$(CC) $(CFLAG) -c $< -o $@

.PHONY: clean
clean:
	rm *.o poisson
