#include <iostream>
#include <math.h>
#include <cstdlib>
#include <sys/time.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
// #include <stdio.h>
// #include <math.h>
// #include <stdlib.h>
// #include <time.h>
#include <string.h>
#include <stdbool.h>

int nNodes;
short int* graph;


void write(FILE *fl){
  int i, j;

  // printf("RESULTADO: \n");

  for( i=0; i<nNodes; i++ ){
    for( j=0; j<nNodes; j++ )
      fprintf( fl, "%d ", graph[i * nNodes + j] );

    fprintf( fl, "\n");
  }
}

void read (){

  char line[50];
  char* token;
  int size = 50;

  int l;
  int c;

  fgets(line,size,stdin);

  while(!feof(stdin)){

    token = strtok(line," "); // split using space as divider
    if(*token == 'p') {

      token = strtok(NULL," "); // sp

      token = strtok(NULL," "); // no. of vertices

      // nNodes é o número de NÓS no grafo.
      nNodes = atoi(token);

      token = strtok(NULL," "); // no. of directed edges

      // printf("N_NODES: %d\n", nNodes);

      // Aloca um vetor do tamanho NODES por NODES
      // graph = (short int*) malloc(nNodes * nNodes * sizeof (short int));
      cudaMallocManaged(&graph, nNodes * nNodes * sizeof (short int));
      if (graph == NULL) {
        printf( "Error in graph allocation: NULL!\n");
        exit( EXIT_FAILURE);
      }

      // Zera a matriz
      for(int i = 0; i < nNodes;i++){
        for(int j = 0; j < nNodes;j++){
          graph[i*nNodes+j] = 0;
        }
      }

    }
    // Coloca o valor do caminho do nó de saída para o de chegada.
     else if(*token == 'a'){
      token = strtok(NULL," ");
      // l -> partida
      l = atoi(token)-1;

      token = strtok(NULL," ");
      // c -> chegada
      c = atoi(token)-1;

      token = strtok(NULL," ");
      graph[l*nNodes+c] = 1;

    }

    fgets(line,size,stdin);
  }

}



__global__
void warshall_CUDA(short int* graph, int nNodes, int offset)
{

    if(offset == -1){
        int k = threadIdx.x + blockIdx.x * blockDim.x;

        for (int i = 0; i < nNodes; i++){
            for (int j = 0; j < nNodes; j++){
                if(graph[i * nNodes + k] + graph[k * nNodes + j] < graph[i * nNodes + j])
                  graph[i * nNodes + j] = 1;
            }
        }
    }
    else{
        int i = (threadIdx.x + blockIdx.x * blockDim.x) + (offset * blockDim.x);

        if(i < nNodes * nNodes){
            int k = float(i) / float(blockDim.x);

            for (int j = 0; j < nNodes; j++){
                if(graph[i * nNodes + k] + graph[k * nNodes + j] < graph[i * nNodes + j])
                  graph[i * nNodes + j] = 1;
            }
        }
    }

}

int main(int argc, char *argv[])
{

    // start time
    timeval start, end;
    gettimeofday(&start, NULL);

    read();

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);


    if(nNodes <= devProp.maxThreadsDim[0]){
        warshall_CUDA<<<1, nNodes>>> (graph, nNodes, -1);
    }else{

        float Blocks = float(nNodes * nNodes) / float(devProp.maxThreadsDim[0]);
        float Offset = Blocks / float(devProp.maxThreadsDim[0]);
        int iblocks = ceil(Blocks);
        int iOffset = ceil(Offset);

        for(int i=0; i < iOffset; i++){
            warshall_CUDA<<< devProp.maxThreadsDim[0], devProp.maxThreadsDim[0] >>> (graph, nNodes, i);
        }

    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // end time
    gettimeofday(&end, NULL);
    double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
    printf("%.4f\n", runtime);

    // write(stdout);

    // Free memory
    cudaFree(graph);

    return 0;
}
