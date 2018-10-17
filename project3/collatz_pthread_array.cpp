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
#include <pthread.h>
#include <sys/time.h>

// shared variables
static long threads; 
static long range;
         
//global dynamic array
static long *solution;
        
static void* collatz(void* arg)
{            
  // determine work for each thread
  const long my_rank = (long)arg;
  const long beg = my_rank * range / threads;
  const long end = (my_rank + 1) * range / threads;
  int maxlen = 0;

  // compute sequence lengths
  int ml = 0;
  for (long i = beg + 1; i <= end; i++) {
    long val = i; 
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val = val / 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }
    if (ml < len) ml = len;
  }

  // reduction
  if (maxlen < ml) {
    if (maxlen < ml) {
      maxlen = ml;
    }

  }

 for (int j = 0; j < threads ; j++){
   solution[j] = maxlen;

}
  return NULL;
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.0\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "usage: %s range num_threads\n", argv[0]); exit(-1);}
  range = atol(argv[1]);
  if (range < 1) {fprintf(stderr, "error: range must be at least 1\n"); exit(-1);}
  printf("range: 1, ..., %ld\n", range);
  threads = atoi(argv[2]);
  if (threads < 1) {fprintf(stderr, "error: num_threads must be at least 1\n"); exit(-1);}
  printf("threads: %ld\n", threads);

  // initialize pthread variables
  pthread_t* const handle = new pthread_t[threads - 1];

  //array for local solutions per thread
  solution = new long [threads];

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  //local variable maxlen
  int  maxlen = 0;

  // launch threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_create(&handle[thread], NULL, collatz, (void *)thread);
  }

  // work for master
  collatz((void*)(threads - 1));

  // join threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_join(handle[thread], NULL);
  }

  // master thread searching for max solution
  maxlen = solution [0];
  for (long thread = 0; thread < threads; thread++){

   if( solution[thread] > maxlen){
        maxlen = solution[thread];
  }

}


  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.3f s\n", runtime);

  // print result
  printf("longest sequence: %d elements\n\n", maxlen);

  // clean up
  delete [] handle;
  delete [] solution;
  return 0;
}
