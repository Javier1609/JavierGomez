#ifndef CONFIG_H
#define CONFIG_H

#define MAX_LINE 4096
#define DEFAULT_RADIUS 0.1
#define DEFAULT_CHUNK 64
#define DEFAULT_SEED 12345u

typedef enum {
    SUBSAMPLE_FIRST = 0,
    SUBSAMPLE_RANDOM = 1
} SubsampleMode;

typedef enum {
    SCHED_STATIC_MODE = 0,
    SCHED_DYNAMIC_MODE = 1
} SchedMode;

typedef struct {
    double longitude;
    double latitude;
} Sample;

typedef struct {
    int neighbor_index;
    double distance;
} NeighborResult;

typedef struct {
    const char *csv_path;
    int target_index;
    int N;
    double radius;
    int threads;
    int chunk;
    unsigned int seed;
    SubsampleMode subsample_mode;
} Config;

#endif