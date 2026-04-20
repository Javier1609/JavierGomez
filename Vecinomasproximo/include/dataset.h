#ifndef DATASET_H
#define DATASET_H

#include "config.h"

Sample *load_all_samples(const char *path, int *out_count);
Sample *subsample_samples(const Sample *all, int total, int n, SubsampleMode mode, unsigned int seed);

#endif