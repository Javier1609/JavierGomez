#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "dataset.h"

static int equals_ignore_case(const char *a, const char *b) {
    while (*a && *b) {
        char ca = *a;
        char cb = *b;

        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char)(cb - 'A' + 'a');

        if (ca != cb) return 0;
        a++;
        b++;
    }
    return (*a == '\0' && *b == '\0');
}

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;

    if (*s == '\0') return s;

    char *end = s + strlen(s) - 1;
    while (end > s && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) {
        *end = '\0';
        end--;
    }

    return s;
}

static int parse_header_indices(char *header, int *longitude_col, int *latitude_col) {
    *longitude_col = -1;
    *latitude_col = -1;

    int col = 0;
    char *token = strtok(header, ",");

    while (token != NULL) {
        char *field = trim(token);

        if (equals_ignore_case(field, "longitude")) *longitude_col = col;
        if (equals_ignore_case(field, "latitude"))  *latitude_col = col;

        col++;
        token = strtok(NULL, ",");
    }

    return (*longitude_col >= 0 && *latitude_col >= 0);
}

static int parse_sample_line(char *line, int longitude_col, int latitude_col, Sample *sample) {
    int col = 0;
    int got_lon = 0;
    int got_lat = 0;

    char *token = strtok(line, ",");

    while (token != NULL) {
        char *field = trim(token);

        if (col == longitude_col) {
            sample->longitude = strtod(field, NULL);
            got_lon = 1;
        }

        if (col == latitude_col) {
            sample->latitude = strtod(field, NULL);
            got_lat = 1;
        }

        col++;
        token = strtok(NULL, ",");
    }

    return got_lon && got_lat;
}

Sample *load_all_samples(const char *path, int *out_count) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("No se pudo abrir el CSV");
        return NULL;
    }

    char line[MAX_LINE];

    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        fprintf(stderr, "Error: el CSV está vacío.\n");
        return NULL;
    }

    int longitude_col, latitude_col;
    if (!parse_header_indices(line, &longitude_col, &latitude_col)) {
        fclose(f);
        fprintf(stderr, "Error: no se encontraron las columnas longitude y latitude.\n");
        return NULL;
    }

    int capacity = 1024;
    int count = 0;

    Sample *samples = (Sample *)malloc((size_t)capacity * sizeof(Sample));
    if (!samples) {
        fclose(f);
        fprintf(stderr, "Error: sin memoria para las muestras.\n");
        return NULL;
    }

    while (fgets(line, sizeof(line), f)) {
        char tmp[MAX_LINE];
        strncpy(tmp, line, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';

        Sample s;
        if (!parse_sample_line(tmp, longitude_col, latitude_col, &s)) {
            continue;
        }

        if (count == capacity) {
            capacity *= 2;
            Sample *new_samples = (Sample *)realloc(samples, (size_t)capacity * sizeof(Sample));
            if (!new_samples) {
                free(samples);
                fclose(f);
                fprintf(stderr, "Error: sin memoria al ampliar el array de muestras.\n");
                return NULL;
            }
            samples = new_samples;
        }

        samples[count++] = s;
    }

    fclose(f);
    *out_count = count;
    return samples;
}

static void shuffle_ints(int *arr, int n, unsigned int seed) {
    srand(seed);
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

Sample *subsample_samples(const Sample *all, int total, int n, SubsampleMode mode, unsigned int seed) {
    if (n > total) n = total;

    Sample *out = (Sample *)malloc((size_t)n * sizeof(Sample));
    if (!out) {
        return NULL;
    }

    if (mode == SUBSAMPLE_FIRST) {
        memcpy(out, all, (size_t)n * sizeof(Sample));
        return out;
    }

    int *indices = (int *)malloc((size_t)total * sizeof(int));
    if (!indices) {
        free(out);
        return NULL;
    }

    for (int i = 0; i < total; i++) {
        indices[i] = i;
    }

    shuffle_ints(indices, total, seed);

    for (int i = 0; i < n; i++) {
        out[i] = all[indices[i]];
    }

    free(indices);
    return out;
}