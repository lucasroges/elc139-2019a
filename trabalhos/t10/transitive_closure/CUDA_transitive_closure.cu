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


// Para cada nó do grafo a funcão calcula
// se existe um caminho com distância 1 até MAX
// (MAX = núm de nós no grafo) até outro nó do grafo.
void warshall(){
  // #pragma omp parallel for collapse(3)

// k e i percorrem cada posição da 'matriz'
  for (int k = 0; k < nNodes; k++){
      for (int i = 0; i < nNodes; i++){
          for (int j = 0; j < nNodes; j++){
            // j é a distância testada
            // if graph[i][k] + graph[k][j] < graph[i][j]
            //      graph[i][j] = 1;
            if(graph[i * nNodes + k] + graph[k * nNodes + j] < graph[i * nNodes + j])
                graph[i * nNodes + j] = 1;
            }
        }
    }

}


__global__
void warshall_CUDA(short int* graph, int nNodes, int resto)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    for(int offset = 0; offset < resto; offset++){
        
    }

    int nNodes2 = nNodes * nNodes;
    if(i < nNodes2 && k < nNodes2){
        for (int j = 0; j < nNodes; j++){
            if(graph[i * nNodes + k] + graph[k * nNodes + j] < graph[i * nNodes + j])
                graph[i * nNodes + j] = 1;
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


    if(nNodes <= devProp.maxThreadsDim[0] && nNodes <= devProp.maxThreadsDim[1]){
        dim3 thr_per_block(nNodes, nNodes);
        warshall_CUDA<<<1, thr_per_block>>> (graph, nNodes, 1);
    }else{
        float resto = (nNodes * nNodes) / devProp.maxThreadsDim[0] * devProp.maxThreadsPerBlock;

        warshall_CUDA<<<ceil(Blocks), thr_per_block>>> (graph, nNodes, resto);
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
