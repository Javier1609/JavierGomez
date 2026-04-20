#include <stdio.h>
#include <stdlib.h>

#include "triangular.h"

size_t triangular_size(int n) {
    return (size_t)n * (size_t)(n - 1) / 2u;
}

size_t triangular_index(int i, int j, int n) {
    if (i == j) {
        fprintf(stderr, "Error: no existe índice triangular para la diagonal.\n");
        exit(EXIT_FAILURE);
    }

    if (i > j) {
        int tmp = i;
        i = j;
        j = tmp;
    }

    return (size_t)i * (size_t)(2 * n - i - 1) / 2u + (size_t)(j - i - 1);
}

double *allocate_triangular_matrix(int n) {
    size_t size = triangular_size(n);
    double *tri = (double *)malloc(size * sizeof(double));

    if (!tri) {
        fprintf(stderr, "Error: no se pudo reservar la matriz triangular.\n");
    }

    return tri;
}