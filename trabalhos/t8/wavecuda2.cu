/*
Adapted from fractal code for CS 4380 / CS 5351

Copyright (c) 2018, Texas State University. All rights reserved.

Redistribution and usage in source and binary form, with or without
modification, is only permitted for educational use.

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
Revision history:
20190610   andreainfufsm   Replaced the function to calculate the color of each pixel
20190615   lucasroges      CUDA parallel version
*/

#include <cstdlib>
#include <sys/time.h>
#include <math.h>
#include "wave.h"

// Kernel function to calculate the value of each pixel
__global__
void calculatePixels(int width, int frames, unsigned char* pic)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int frame = 0; frame < frames; frame++) {
    for (int row = index; row < width; row += offset) {
      for (int col = 0; col < width; col++) {
        float fx = col - 1024/2;
        float fy = row - 1024/2;
        float d = sqrtf( fx * fx + fy * fy );
        unsigned char color = (unsigned char) (160.0f + 127.0f *
                                          cos(d/10.0f - frame/7.0f) /
                                          (d/50.0f + 1.0f));
        pic[frame * width * width + row * width + col] = (unsigned char) color;
      }
    }
  }
}

int main(int argc, char *argv[])
{

  // check command line
  if (argc != 4) {fprintf(stderr, "usage: %s frame_width num_frames num_threads\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 100) {fprintf(stderr, "error: frame_width must be at least 100\n"); exit(-1);}
  int frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  int threads = atoi(argv[3]);
  if (threads < 1) {fprintf(stderr, "error: num_threads must be at least 1\n"); exit(-1);};
  printf("computing %d frames of %d by %d picture with %d blocks of %d threads\n", frames, width, width, (width + threads - 1) / threads, threads);

  // allocate picture array
  unsigned char* pic;
  cudaMallocManaged(&pic, frames*width*width*sizeof(unsigned char));
    
  // start time
  timeval start, end;
  gettimeofday(&start, NULL);
    
  int blockSize = threads;
  int numBlocks = (width + blockSize - 1) / blockSize;

  calculatePixels<<<numBlocks, blockSize>>>(width, frames, pic);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  // verify result by writing frames to BMP files
  if ((width <= 256) && (frames <= 100)) {
    for (int frame = 0; frame < frames; frame++) {
      char name[32];
      sprintf(name, "wave%d.bmp", frame + 1000);
      writeBMP(width, width, &pic[frame * width * width], name);
    }
  }

  //delete [] pic;
  cudaFree(pic);
  return 0;
}
