#include <cstdlib>
#include <cstdio>
#include <pthread.h>
#include <sys/time.h>

// shared variables
static long threads;
static long range;
static int maxlen;
static pthread_mutex_t mutex;

static void* collatz(void* arg)
{

  maxlen = 0;
  // determine work for each thread
  const long my_rank = (long)arg;
  const long beg = my_rank * range / threads;
  const long end = (my_rank + 1) * range / threads;

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

  // reduction <<<<<<<<<<<<<
  if (maxlen < ml) {
    pthread_mutex_lock(&mutex);
    if (maxlen < ml) {
      maxlen = ml;
    }
    pthread_mutex_unlock(&mutex);
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

  // initialize pthread variables <<<<<<<<<<<<<<
  pthread_mutex_init(&mutex, NULL);
  pthread_t* const handle = new pthread_t[threads - 1];
  //A dynamic array for all threads.
  pthread_t* const result = new pthread_t[threads-1];

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);
	
  maxlen = 0; 

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

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.3f s\n", runtime);


  // print result edit this for maxlen
  for(long thread = 0; thread < threads - 1; thread++){
  	 printf("longest sequence: %d elements\n\n", result[thread]); 
  }
 

  // clean up
  pthread_mutex_destroy(&mutex);
  delete [] handle;
  delete [] result;
  return 0;
}