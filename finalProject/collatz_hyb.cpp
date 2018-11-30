/*
Collatz code for CS 4380 / CS 5351

Copyright (c) 2018, Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <mpi.h>
#include <sys/time.h>

void GPU_Init();
void GPU_Exec(const long start, const long stop);
int GPU_Fini();

static int collatz(const long start, const long stop)
{
  int maxlen = 0;

  // todo: OpenMP code with 19 threads, default(none), a reduction, and a block-cyclic schedule with a block size of 100

  return maxlen;
}

int main(int argc, char *argv[])
{
  int comm_sz, my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    printf("Collatz v1.0\n");
  }

  // check command line
  if (argc != 3) {fprintf(stderr, "usage: %s range cpu_percentage\n", argv[0]); exit(-1);}
  const long range = atol(argv[1]);
  if (range < 1) {fprintf(stderr, "error: range must be at least 1\n"); exit(-1);}
  const double percentage = atof(argv[2]);
  if ((percentage < 0) || (percentage > 100.0)) {fprintf(stderr, "error: cpu_percentage must be between 0 and 100\n"); exit(-1);}

  const long cpu_start = my_rank * range / comm_sz;
  const long gpu_stop = (my_rank + 1) * range / comm_sz;
  const long my_range = gpu_stop - cpu_start;
  const long cpu_stop = cpu_start + my_range * percentage / 100.0;
  const long gpu_start = cpu_stop;

  if (my_rank == 0) {
    printf("range: 1, ..., %ld\n", range);
    printf("CPU percentage: %.1f\n", percentage);
    printf("MPI tasks: %d\n", comm_sz);
  }

  GPU_Init();

  // start time
  timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD);  // for better timing
  gettimeofday(&start, NULL);

  GPU_Exec(gpu_start, gpu_stop);
  const int cpu_maxlen = collatz(cpu_start, cpu_stop);
  const int gpu_maxlen = GPU_Fini();
  int loc_maxlen = std::max(cpu_maxlen, gpu_maxlen);

  int maxlen;
  // todo: MPI_Reduce(...);

  if (my_rank == 0) {
    gettimeofday(&end, NULL);
    const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("compute time: %.3f s\n", runtime);

    printf("longest sequence: %d elements\n", maxlen);
  }

  MPI_Finalize();
  return 0;
}
