#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "config.h"

double euclidean_distance(const Sample *a, const Sample *b);

/* Construcción de la matriz triangular superior */
void compute_distances_sequential(const Sample *samples, int n, double *tri);
void compute_distances_parallel(const Sample *samples, int n, double *tri, SchedMode mode, int chunk);

/* Vecino más cercano usando la matriz triangular */
NeighborResult nearest_from_matrix_sequential(const double *tri, int n, int target);
NeighborResult nearest_from_matrix_parallel(const double *tri, int n, int target, SchedMode mode, int chunk);

/* Conteo de vecinos dentro de radio R */
void count_neighbors_radius_sequential(const double *tri, int n, double radius, int *counts);
void count_neighbors_radius_parallel(const double *tri, int n, double radius, int *counts, SchedMode mode, int chunk);

/* Versión alternativa: solo distancias a la muestra objetivo */
NeighborResult direct_target_only_sequential(const Sample *samples, int n, int target);
NeighborResult direct_target_only_parallel_critical(const Sample *samples, int n, int target, SchedMode mode, int chunk);
NeighborResult direct_target_only_parallel_reduction(const Sample *samples, int n, int target, SchedMode mode, int chunk);

#endif