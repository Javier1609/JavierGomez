#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "config.h"

void print_usage(const char *prog);
int parse_mode(const char *text, SubsampleMode *mode);
const char *sched_name(SchedMode mode);
void print_config(const Config *cfg, int total_samples);
void benchmark_all(const Sample *samples, const Config *cfg);

#endif