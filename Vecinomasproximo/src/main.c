#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "config.h"
#include "dataset.h"
#include "benchmark.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    Config cfg;
    cfg.csv_path = argv[1];
    cfg.target_index = atoi(argv[2]);
    cfg.N = atoi(argv[3]);
    cfg.radius = (argc > 5) ? atof(argv[5]) : DEFAULT_RADIUS;
    cfg.threads = (argc > 6) ? atoi(argv[6]) : omp_get_max_threads();
    cfg.chunk = (argc > 7) ? atoi(argv[7]) : DEFAULT_CHUNK;
    cfg.seed = (argc > 8) ? (unsigned int)strtoul(argv[8], NULL, 10) : DEFAULT_SEED;

    if (!parse_mode(argv[4], &cfg.subsample_mode)) {
        fprintf(stderr, "Error: modo de submuestreo no valido.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (cfg.N <= 1) {
        fprintf(stderr, "Error: N debe ser mayor que 1.\n");
        return EXIT_FAILURE;
    }

    if (cfg.target_index < 0) {
        fprintf(stderr, "Error: el indice objetivo debe ser >= 0.\n");
        return EXIT_FAILURE;
    }

    if (cfg.threads <= 0) cfg.threads = 1;
    if (cfg.chunk <= 0) cfg.chunk = DEFAULT_CHUNK;

    omp_set_num_threads(cfg.threads);

    int total = 0;
    Sample *all_samples = load_all_samples(cfg.csv_path, &total);
    if (!all_samples) {
        return EXIT_FAILURE;
    }

    if (total < 2) {
        fprintf(stderr, "Error: el CSV no tiene suficientes muestras.\n");
        free(all_samples);
        return EXIT_FAILURE;
    }

    if (cfg.N > total) cfg.N = total;

    if (cfg.target_index >= cfg.N) {
        fprintf(stderr, "Error: el indice objetivo debe estar entre 0 y %d.\n", cfg.N - 1);
        free(all_samples);
        return EXIT_FAILURE;
    }

    Sample *samples = subsample_samples(all_samples, total, cfg.N, cfg.subsample_mode, cfg.seed);
    free(all_samples);

    if (!samples) {
        fprintf(stderr, "Error: no se pudo crear el subconjunto de muestras.\n");
        return EXIT_FAILURE;
    }

    print_config(&cfg, total);
    benchmark_all(samples, &cfg);

    free(samples);
    return EXIT_SUCCESS;
}