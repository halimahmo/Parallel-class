/*
Fractal code for CS 4380 / CS 5351

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
#include <cmath>
#include <sys/time.h>
#include "cs43805351.h"
#include <mpi.h>

static const double Delta = 0.004;
static const double xMid =  0.2389;
static const double yMid =  0.55267;

unsigned char* GPU_Init(const int gpu_frames, const int width);
void GPU_Exec(const int start_frame, const int gpu_frames, const int width, unsigned char* pic_d);
void GPU_Fini(const int gpu_frames, const int width, unsigned char* pic, unsigned char* pic_d);

static void fractal(const int start_frame, const int cpu_frames, const int width, unsigned char* pic)
{
  const int totalframes = start_frame + cpu_frames;
  // todo: use OpenMP to compute the frames with 19 threads, default(none), and a cyclic schedule
  #pragma omp parallel for num_threads(19) default(none) shared(pic, width) schedule(static, 1)
  for (int frame = start_frame; frame < totalframes; frame++) {
    const double delta = Delta * pow(0.98, frame);
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    
    for (int row = 0; row < width; row++) {
      const double cy = yMin + row * dw;
      for (int col = 0; col < width; col++) {
        const double cx = xMin + col * dw;
        double x = cx;
        double y = cy;
        int depth = 256;
        double x2, y2;
        do {
          x2 = x * x;
          y2 = y * y;
          y = 2 * x * y + cy;
          x = x2 - y2 + cx;
          depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));
        pic[(frame - start_frame) * width * width + row * width + col] = (unsigned char)depth;
      }
    }
  }
}

int main(int argc, char *argv[])
{
  int comm_sz
  int my_rank;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {printf("Fractal v1.7\n");}

  // check command line
  if (argc != 4) {fprintf(stderr, "usage: %s frame_width cpu_frames gpu_frames\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: frame_width must be at least 10\n"); exit(-1);}
  int cpu_frames = atoi(argv[2]);
  if (cpu_frames < 0) {fprintf(stderr, "error: cpu_frames must be at least 0\n"); exit(-1);}
  int gpu_frames = atoi(argv[3]);
  if (gpu_frames < 0) {fprintf(stderr, "error: gpu_frames must be at least 0\n"); exit(-1);}
  int frames = cpu_frames + gpu_frames;
  if (frames < 1) {fprintf(stderr, "error: total number of frames must be at least 1\n"); exit(-1);}
  if (my_rank == 0) {
    printf("computing %d frames of %d by %d fractal (%d CPU frames and %d GPU frames)\n", frames, width, width, cpu_frames, gpu_frames);
    printf("The number of MPI Processes is: %d\n",comm_sz);    
    }

  gpu_frames = gpu_frames / comm_sz;
  cpu_frames = cpu_frames / comm_sz;
  
  frames = frames / comm_sz;
  const int cpu_start_frame = my_rank * frames;
  const int gpu_start_frame = cpu_start_frame + cpu_frames;
  

  // allocate picture arrays
  unsigned char* pic ;
  unsigned char* my_pic = new unsigned char[frames * width *width];
  unsigned char* pic_d = GPU_Init(gpu_frames, width);

if (my_rank == 0){
  pic = new unsigned char[(frames * comm_sz) * width * width];
}
  // start time
  timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&start, NULL);

  // asynchronously compute the requested frames on the GPU
  GPU_Exec(gpu_start_frame, gpu_frames, width, pic_d);

  // compute the remaining frames on the CPU
  fractal(cpu_start_frame, cpu_frames, width, my_pic);
  
  // copy the GPU's result into the appropriate location of the CPU's pic array
  GPU_Fini(gpu_frames, width, &my_pic[cpu_frames * width * width], pic_d);
  MPI_Gather(my_pic, frames * width * width, MPI_UNSIGNED_CHAR, pic, frames * width * width, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  
  if(my_rank == 0) {
      printf("compute time: %.3f s\n", runtime);

  // verify result by writing frames to BMP files
  if ((width <= 256) && ((frames * comm_sz) <= 100)){
    for (int frame = 0; frame < (frames * comm_sz); frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, width, &pic[frame * width * width], name);
        }
    }
  }
  if(my_rank == 0) {
  delete [] pic;
  }
  delete [] my_pic;
  MPI_Finalize();
  return 0;
}
