CC = mpicc

# Compiler Flags
CFLAGS = -O2 -I/opt/homebrew/Cellar/lapack/3.12.1/include/

# Linker Flags
LDFLAGS = -L/opt/homebrew/Cellar/openblas/0.3.29/lib/ -lopenblas -llapack -lm

TARGET = tsqr

all: $(TARGET)

$(TARGET): tsqr.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	mpirun -np 4 ./$(TARGET)

clean:
	rm -f $(TARGET)