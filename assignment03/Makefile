CC = gcc
CFLAGS = -Wall -O2 -std=c99
TARGET = cg_poisson

all: $(TARGET)

$(TARGET): cg_poisson.c
	$(CC) $(CFLAGS) -o $(TARGET) cg_poisson.c -lm

clean:
	rm -f $(TARGET) *.o