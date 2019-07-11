#include <complex>
#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <thrust/complex.h>
#include <math.h>

using namespace std;
typedef thrust::complex<float> complex_f;

__global__
void mandelbrot_CUDA(char *mat, int max_row, int max_column, int max_n)
{

        int pos = (blockDim.x * threadIdx.x) + threadIdx.y;
        pos = pos + (blockDim.x*blockDim.x) * blockIdx.x;

        int r = float(pos) / float(max_column);
        int c = pos - (max_column * r);

        if(pos < max_row * max_column){
            complex_f z;
            int n = 0;
            while(abs(z) < 2 && ++n < max_n)
                z = z*z + complex_f(
                    (float)c * 2 / max_column - 1.5,
                    (float)r * 2 / max_row - 1
                );
            mat[pos]=(n == max_n ? '#' : '.');
        }
}


int main(){

	int max_row, max_column, max_n;
	cin >> max_row;
	cin >> max_column;
	cin >> max_n;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    // start time
    timeval start, end;
    gettimeofday(&start, NULL);

    char *mat;
    cudaMallocManaged(&mat, max_row * max_column * sizeof(char));


    if(max_row * max_column <= devProp.maxThreadsDim[0]){
        dim3 thr_per_block(max_row, max_column);
        mandelbrot_CUDA<<<1, thr_per_block>>> (mat, max_row, max_column, max_n);
    }else{
        float Blocks = (float(max_row) * float(max_column)) / float(devProp.maxThreadsDim[1]);
        float threads = sqrt(devProp.maxThreadsDim[1]);
        int iblocks, ithreads;
        iblocks = ceil(Blocks);
        ithreads = round(threads);
        dim3 thr_per_block(ithreads, ithreads);

        mandelbrot_CUDA<<<iblocks, thr_per_block>>> (mat, max_row, max_column, max_n);
    }


    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // end time
    gettimeofday(&end, NULL);
    double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
    printf("%.4f\n", runtime);

	// for(int r = 0; r < max_row; ++r){
	// 	for(int c = 0; c < max_column; ++c)
	// 		std::cout << mat[r * max_column + c];
	// 	cout << '\n';
	// }
}
