#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    // 1. Lectura de argumentos: N, umbral e iteraciones
    if (argc < 4) {
        printf("Uso: %s <N> <umbral> <max_iter>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    float umbral = atof(argv[2]);
    int max_iter = atoi(argv[3]);

    // 2. Alojamiento de memoria para matrices N x N
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));

    // 3. Inicialización: bordes a 100, interior a 0
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
                A[i][j] = B[i][j] = 100.0;
            else
                A[i][j] = B[i][j] = 0.0;
        }
    }

    double t_inicio = omp_get_wtime(); // Medir tiempo con OpenMP

    omp_set_num_threads(4);

    int iter = 0;
    float diff_max;

    do {
        diff_max = 0;
        // 4. Paralelización del bucle con OpenMP
        // Se usa reduction para evitar condiciones de carrera en diff_max
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                B[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]) / 4.0;

                float cambio = fabsf(B[i][j] - A[i][j]);
                if (cambio > diff_max) diff_max = cambio;
            }
        }

        // Actualizar matriz A (copia de B a A)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                A[i][j] = B[i][j];
            }
        }
        iter++;
    } while (iter < max_iter && diff_max > umbral); // Condiciones de parada

    double t_fin = omp_get_wtime();
    printf("--- Resultados Calor ---\n");
    printf("N: %d, Iteraciones: %d, Tiempo: %f s, Diff: %f\n", N, iter, t_fin - t_inicio, diff_max);

    free(A); free(B);
    return 0;
}
