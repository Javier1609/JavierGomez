CC = gcc
CFLAGS = -O2 -fopenmp -Wall -Wextra -Iinclude
LDFLAGS = -lm

SRC = src/main.c src/dataset.c src/triangular.c src/algorithms.c src/benchmark.c
OUT = vecino_mas_proximo

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	del /Q $(OUT).exe 2>nul || rm -f $(OUT)

run:
	./$(OUT) housing.csv 10 5000 first 0.1 8 64 12345