CC      = gcc
CFLAGS  = -O2 -fopenmp -Wall -Wextra -Iinclude
LDFLAGS = -lm

SRC = src/main.c src/dataset.c src/triangular.c src/algorithms.c src/benchmark.c
OUT = vecino_mas_proximo

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	rm -f $(OUT) $(OUT).exe

run:
	./$(OUT) housing.csv 10 5000 first 0.1 4 64 12345