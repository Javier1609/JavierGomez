#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "benchmark.h"
#include "triangular.h"
#include "algorithms.h"

void print_usage(const char *prog) {
    fprintf(stderr,
        "Uso:\n"
        "  %s <csv> <indice_objetivo> <N> <modo_submuestreo> [radio] [hilos] [chunk] [seed]\n\n"
        "Argumentos:\n"
        "  <csv>               Ruta al fichero housing.csv\n"
        "  <indice_objetivo>   Índice dentro del subconjunto [0, N-1]\n"
        "  <N>                 Número de muestras a usar\n"
        "  <modo_submuestreo>  first | random\n"
        "  [radio]             Radio R (por defecto 0.1)\n"
        "  [hilos]             Número de hilos OpenMP\n"
        "  [chunk]             Tamaño del chunk para schedule\n"
        "  [seed]              Semilla para el modo random\n\n"
        "Ejemplo:\n"
        "  %s housing.csv 10 5000 random 0.1 8 64 12345\n",
        prog, prog);
}

int parse_mode(const char *text, SubsampleMode *mode) {
    if (strcmp(text, "first") == 0) {
        *mode = SUBSAMPLE_FIRST;
        return 1;
    }
    if (strcmp(text, "random") == 0) {
        *mode = SUBSAMPLE_RANDOM;
        return 1;
    }
    return 0;
}

const char *sched_name(SchedMode mode) {
    return (mode == SCHED_STATIC_MODE) ? "static" : "dynamic";
}

static void print_sample(const Sample *samples, int idx, const char *label) {
    printf("%s idx=%d -> (longitude=%.6f, latitude=%.6f)\n",
           label, idx, samples[idx].longitude, samples[idx].latitude);
}

void print_config(const Config *cfg, int total_samples) {
    printf("========================================\n");
    printf("CONFIGURACION DEL EXPERIMENTO\n");
    printf("========================================\n");
    printf("CSV: %s\n", cfg->csv_path);
    printf("Total muestras en el CSV: %d\n", total_samples);
    printf("N usado: %d\n", cfg->N);
    printf("Indice objetivo: %d\n", cfg->target_index);
    printf("Submuestreo: %s\n", cfg->subsample_mode == SUBSAMPLE_FIRST ? "first" : "random");
    printf("Radio R: %.6f\n", cfg->radius);
    printf("Hilos OpenMP: %d\n", cfg->threads);
    printf("Chunk: %d\n", cfg->chunk);
    printf("========================================\n\n");
}

void benchmark_all(const Sample *samples, const Config *cfg) {
    double *tri = allocate_triangular_matrix(cfg->N);
    int *counts_seq = (int *)malloc((size_t)cfg->N * sizeof(int));
    int *counts_par = (int *)malloc((size_t)cfg->N * sizeof(int));

    if (!tri || !counts_seq || !counts_par) {
        fprintf(stderr, "Error: no se pudo reservar memoria de trabajo.\n");
        free(tri);
        free(counts_seq);
        free(counts_par);
        exit(EXIT_FAILURE);
    }

    double t0, t1;

    printf("===== 1) SOLUCION SECUENCIAL =====\n");
    t0 = omp_get_wtime();
    compute_distances_sequential(samples, cfg->N, tri);
    NeighborResult seq_nn = nearest_from_matrix_sequential(tri, cfg->N, cfg->target_index);
    count_neighbors_radius_sequential(tri, cfg->N, cfg->radius, counts_seq);
    t1 = omp_get_wtime();

    printf("Tiempo total secuencial: %.6f s\n", t1 - t0);
    print_sample(samples, cfg->target_index, "Muestra objetivo");
    print_sample(samples, seq_nn.neighbor_index, "Vecino mas cercano (secuencial)");
    printf("Distancia minima: %.12f\n", seq_nn.distance);
    printf("Vecinos dentro de R para la muestra objetivo: %d\n\n", counts_seq[cfg->target_index]);

    printf("===== 2) 3) 4) VERSIONES PARALELAS =====\n");
    for (int s = 0; s < 2; s++) {
        SchedMode mode = (s == 0) ? SCHED_STATIC_MODE : SCHED_DYNAMIC_MODE;

        printf("---- schedule(%s, %d) ----\n", sched_name(mode), cfg->chunk);

        t0 = omp_get_wtime();
        compute_distances_parallel(samples, cfg->N, tri, mode, cfg->chunk);
        t1 = omp_get_wtime();
        printf("Tiempo calculo matriz distancias: %.6f s\n", t1 - t0);

        t0 = omp_get_wtime();
        NeighborResult par_nn = nearest_from_matrix_parallel(tri, cfg->N, cfg->target_index, mode, cfg->chunk);
        t1 = omp_get_wtime();
        printf("Tiempo busqueda vecino mas cercano: %.6f s\n", t1 - t0);
        print_sample(samples, par_nn.neighbor_index, "Vecino mas cercano (paralelo)");
        printf("Distancia minima: %.12f\n", par_nn.distance);

        t0 = omp_get_wtime();
        count_neighbors_radius_parallel(tri, cfg->N, cfg->radius, counts_par, mode, cfg->chunk);
        t1 = omp_get_wtime();
        printf("Tiempo conteo vecinos dentro de R: %.6f s\n", t1 - t0);
        printf("Vecinos dentro de R para la muestra objetivo: %d\n\n", counts_par[cfg->target_index]);
    }

    printf("===== 5) VERSION ALTERNATIVA: SOLO DISTANCIAS A LA MUESTRA OBJETIVO =====\n");

    t0 = omp_get_wtime();
    NeighborResult alt_seq = direct_target_only_sequential(samples, cfg->N, cfg->target_index);
    t1 = omp_get_wtime();
    printf("Secuencial solo objetivo: %.6f s\n", t1 - t0);
    print_sample(samples, alt_seq.neighbor_index, "Vecino (solo objetivo secuencial)");
    printf("Distancia minima: %.12f\n\n", alt_seq.distance);

    for (int s = 0; s < 2; s++) {
        SchedMode mode = (s == 0) ? SCHED_STATIC_MODE : SCHED_DYNAMIC_MODE;

        t0 = omp_get_wtime();
        NeighborResult crit_res = direct_target_only_parallel_critical(samples, cfg->N, cfg->target_index, mode, cfg->chunk);
        t1 = omp_get_wtime();
        printf("Paralelo critical - schedule(%s): %.6f s\n", sched_name(mode), t1 - t0);
        print_sample(samples, crit_res.neighbor_index, "Vecino (critical)");
        printf("Distancia minima: %.12f\n", crit_res.distance);

        t0 = omp_get_wtime();
        NeighborResult red_res = direct_target_only_parallel_reduction(samples, cfg->N, cfg->target_index, mode, cfg->chunk);
        t1 = omp_get_wtime();
        printf("Paralelo reduction - schedule(%s): %.6f s\n", sched_name(mode), t1 - t0);
        print_sample(samples, red_res.neighbor_index, "Vecino (reduction)");
        printf("Distancia minima: %.12f\n\n", red_res.distance);
    }

    free(tri);
    free(counts_seq);
    free(counts_par);
}