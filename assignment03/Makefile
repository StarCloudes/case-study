CC = gcc
CFLAGS = -Wall -O2 -std=c99
TARGETS = cg_poisson cg_dense

all: $(TARGETS)

cg_poisson: cg_poisson.c
	$(CC) $(CFLAGS) -o cg_poisson cg_poisson.c -lm

cg_dense: cg_convergence_dense.c
	$(CC) $(CFLAGS) -o cg_dense cg_convergence_dense.c -lm

clean:
	rm -f $(TARGETS) *.o 