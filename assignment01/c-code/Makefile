# Compiler and compilation options
CC = mpicc

# Compiler Flags
CFLAGS = -O2 -I/opt/homebrew/Cellar/lapack/3.12.1/include/

# Linker Flags
LDFLAGS = -L/opt/homebrew/Cellar/openblas/0.3.29/lib/ -lopenblas -llapack -lm

# Executable names
TARGET = tsqr
PERF_TARGET = performance_test

# Source files list for tsqr
SRCS = main.c matrix_utils.c tsqr_module.c
OBJS = $(SRCS:.c=.o)

# Source files list for performance test
PERF_SRCS = performance_test.c matrix_utils.c tsqr_module.c
PERF_OBJS = $(PERF_SRCS:.c=.o)

# Default target: Build the tsqr executable
all: $(TARGET)

# Link the tsqr executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Link the performance_test executable
$(PERF_TARGET): $(PERF_OBJS)
	$(CC) $(CFLAGS) -o $(PERF_TARGET) $^ $(LDFLAGS)

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $<

# Run tsqr: Execute using mpirun with 4 processes
run: $(TARGET)
	mpirun -np 4 ./$(TARGET)

# Run performance_test: Execute using mpirun with 4 processes
run_perf: $(PERF_TARGET)
	mpirun -np 4 ./$(PERF_TARGET)

# Clean up generated files
clean:
	rm -f $(TARGET) $(OBJS) $(PERF_TARGET) $(PERF_OBJS)