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
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static int* maxlen_d;

static __global__ void CollatzKernel(const long start, const long stop, int* maxlen_d)
{
  // todo: process from start (exclusive) to stop (inclusive) with one thread per value (based on code from previous project)
  // compute sequence lengths

  const long x = stop - start;

  const long idx = threadIdx.x + blockIdx.x * (long)blockDim.x;
  if(idx <= x){
    long val = idx + 1;
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val = val / 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }

    //thread updating maxlen using atomicMax
    if (*maxlen < len) atomicMax(maxlen, len);
  }
}

void GPU_Init()
{
  int temp = 0;
  if (cudaSuccess != cudaMalloc((void **)&maxlen_d, sizeof(int))) {fprintf(stderr, "could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMemcpy(maxlen_d, &temp, sizeof(int), cudaMemcpyHostToDevice)) {fprintf(stderr, "copying to device failed\n"); exit(-1);}
}

void GPU_Exec(const long start, const long stop)
{
  if (start != stop) {
    CollatzKernel<<<(stop - start + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(start, stop, maxlen_d);
  }
}

int GPU_Fini()
{
  int ret;
  if (cudaSuccess != cudaMemcpy(&ret, maxlen_d, sizeof(int), cudaMemcpyDeviceToHost)) {fprintf(stderr, "copying from device failed\n"); exit(-1);}
  cudaFree(maxlen_d);
  return ret;
}
