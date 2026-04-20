#ifndef TRIANGULAR_H
#define TRIANGULAR_H

#include <stddef.h>

size_t triangular_size(int n);
size_t triangular_index(int i, int j, int n);
double *allocate_triangular_matrix(int n);

#endif