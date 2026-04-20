#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#include "algorithms.h"
#include "triangular.h"

double euclidean_distance(const Sample *a, const Sample *b) {
    double dx = a->longitude - b->longitude;
    double dy = a->latitude - b->latitude;
    return sqrt(dx * dx + dy * dy);
}

/* ===================================================== */
/* 1) CONSTRUCCION DE LA MATRIZ TRIANGULAR SUPERIOR      */
/* ===================================================== */

void compute_distances_sequential(const Sample *samples, int n, double *tri) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            tri[triangular_index(i, j, n)] = euclidean_distance(&samples[i], &samples[j]);
        }
    }
}

void compute_distances_parallel(const Sample *samples, int n, double *tri, SchedMode mode, int chunk) {
    if (mode == SCHED_STATIC_MODE) {
        #pragma omp parallel for schedule(static, chunk)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                tri[triangular_index(i, j, n)] = euclidean_distance(&samples[i], &samples[j]);
            }
        }
    } else {
        #pragma omp parallel for schedule(dynamic, chunk)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                tri[triangular_index(i, j, n)] = euclidean_distance(&samples[i], &samples[j]);
            }
        }
    }
}

/* ===================================================== */
/* 2) BUSQUEDA DEL VECINO MAS PROXIMO DESDE LA MATRIZ    */
/* ===================================================== */

NeighborResult nearest_from_matrix_sequential(const double *tri, int n, int target) {
    NeighborResult result;
    result.neighbor_index = -1;
    result.distance = DBL_MAX;

    for (int j = 0; j < target; j++) {
        double d = tri[triangular_index(j, target, n)];
        if (d < result.distance) {
            result.distance = d;
            result.neighbor_index = j;
        }
    }

    for (int j = target + 1; j < n; j++) {
        double d = tri[triangular_index(target, j, n)];
        if (d < result.distance) {
            result.distance = d;
            result.neighbor_index = j;
        }
    }

    return result;
}

NeighborResult nearest_from_matrix_parallel(const double *tri, int n, int target, SchedMode mode, int chunk) {
    NeighborResult global_result;
    global_result.neighbor_index = -1;
    global_result.distance = DBL_MAX;

    #pragma omp parallel
    {
        NeighborResult local_result;
        local_result.neighbor_index = -1;
        local_result.distance = DBL_MAX;

        if (mode == SCHED_STATIC_MODE) {
            #pragma omp for schedule(static, chunk) nowait
            for (int j = 0; j < target; j++) {
                double d = tri[triangular_index(j, target, n)];
                if (d < local_result.distance) {
                    local_result.distance = d;
                    local_result.neighbor_index = j;
                }
            }

            #pragma omp for schedule(static, chunk) nowait
            for (int j = target + 1; j < n; j++) {
                double d = tri[triangular_index(target, j, n)];
                if (d < local_result.distance) {
                    local_result.distance = d;
                    local_result.neighbor_index = j;
                }
            }
        } else {
            #pragma omp for schedule(dynamic, chunk) nowait
            for (int j = 0; j < target; j++) {
                double d = tri[triangular_index(j, target, n)];
                if (d < local_result.distance) {
                    local_result.distance = d;
                    local_result.neighbor_index = j;
                }
            }

            #pragma omp for schedule(dynamic, chunk) nowait
            for (int j = target + 1; j < n; j++) {
                double d = tri[triangular_index(target, j, n)];
                if (d < local_result.distance) {
                    local_result.distance = d;
                    local_result.neighbor_index = j;
                }
            }
        }

        #pragma omp critical
        {
            if (local_result.distance < global_result.distance) {
                global_result = local_result;
            }
        }
    }

    return global_result;
}

/* ===================================================== */
/* 3) CONTEO DE VECINOS DENTRO DE UN RADIO R             */
/* ===================================================== */

void count_neighbors_radius_sequential(const double *tri, int n, double radius, int *counts) {
    for (int i = 0; i < n; i++) {
        counts[i] = 0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double d = tri[triangular_index(i, j, n)];
            if (d <= radius) {
                counts[i]++;
                counts[j]++;
            }
        }
    }
}

void count_neighbors_radius_parallel(const double *tri, int n, double radius, int *counts, SchedMode mode, int chunk) {
    for (int i = 0; i < n; i++) {
        counts[i] = 0;
    }

    #pragma omp parallel
    {
        int *local_counts = (int *)calloc((size_t)n, sizeof(int));
        if (!local_counts) {
            fprintf(stderr, "Error: sin memoria para contadores locales.\n");
            exit(EXIT_FAILURE);
        }

        if (mode == SCHED_STATIC_MODE) {
            #pragma omp for schedule(static, chunk)
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double d = tri[triangular_index(i, j, n)];
                    if (d <= radius) {
                        local_counts[i]++;
                        local_counts[j]++;
                    }
                }
            }
        } else {
            #pragma omp for schedule(dynamic, chunk)
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double d = tri[triangular_index(i, j, n)];
                    if (d <= radius) {
                        local_counts[i]++;
                        local_counts[j]++;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                counts[i] += local_counts[i];
            }
        }

        free(local_counts);
    }
}

/* ===================================================== */
/* 4) VERSION ALTERNATIVA: SOLO DISTANCIAS A TARGET      */
/* ===================================================== */

NeighborResult direct_target_only_sequential(const Sample *samples, int n, int target) {
    NeighborResult result;
    result.neighbor_index = -1;
    result.distance = DBL_MAX;

    for (int i = 0; i < n; i++) {
        if (i == target) continue;

        double d = euclidean_distance(&samples[target], &samples[i]);
        if (d < result.distance) {
            result.distance = d;
            result.neighbor_index = i;
        }
    }

    return result;
}

NeighborResult direct_target_only_parallel_critical(const Sample *samples, int n, int target, SchedMode mode, int chunk) {
    NeighborResult global_result;
    global_result.neighbor_index = -1;
    global_result.distance = DBL_MAX;

    #pragma omp parallel
    {
        NeighborResult local_result;
        local_result.neighbor_index = -1;
        local_result.distance = DBL_MAX;

        if (mode == SCHED_STATIC_MODE) {
            #pragma omp for schedule(static, chunk)
            for (int i = 0; i < n; i++) {
                if (i == target) continue;

                double d = euclidean_distance(&samples[target], &samples[i]);
                if (d < local_result.distance) {
                    local_result.distance = d;
                    local_result.neighbor_index = i;
                }
            }
        } else {
            #pragma omp for schedule(dynamic, chunk)
            for (int i = 0; i < n; i++) {
                if (i == target) continue;

                double d = euclidean_distance(&samples[target], &samples[i]);
                if (d < local_result.distance) {
                    local_result.distance = d;
                    local_result.neighbor_index = i;
                }
            }
        }

        #pragma omp critical
        {
            if (local_result.distance < global_result.distance) {
                global_result = local_result;
            }
        }
    }

    return global_result;
}

NeighborResult direct_target_only_parallel_reduction(const Sample *samples, int n, int target, SchedMode mode, int chunk) {
    double global_min = DBL_MAX;

    if (mode == SCHED_STATIC_MODE) {
        #pragma omp parallel for schedule(static, chunk) reduction(min:global_min)
        for (int i = 0; i < n; i++) {
            if (i == target) continue;
            double d = euclidean_distance(&samples[target], &samples[i]);
            if (d < global_min) {
                global_min = d;
            }
        }
    } else {
        #pragma omp parallel for schedule(dynamic, chunk) reduction(min:global_min)
        for (int i = 0; i < n; i++) {
            if (i == target) continue;
            double d = euclidean_distance(&samples[target], &samples[i]);
            if (d < global_min) {
                global_min = d;
            }
        }
    }

    NeighborResult result;
    result.neighbor_index = -1;
    result.distance = global_min;

    for (int i = 0; i < n; i++) {
        if (i == target) continue;
        double d = euclidean_distance(&samples[target], &samples[i]);
        if (fabs(d - global_min) < 1e-12) {
            result.neighbor_index = i;
            break;
        }
    }

    return result;
}