# T10: Maratona de Programação Paralela

### Nomes: Bruno Alves e Lucas Araujo

# Parte 1 -> Transitive Closure em CUDA:

### Kernel:

**Código: [Transitive.cu](transitive_closure/CUDA_transitive_closure.cu)**

```c
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
```

### Main:

```c
// Caso o número de nós seja menor que o valor máximo de threads
if(nNodes <= devProp.maxThreadsDim[0]){
    warshall_CUDA<<<1, nNodes>>> (graph, nNodes, -1);
}
// Caso geral:
else{

    // Calcula o número de blocos que seriam necessários
    float Blocks = float(nNodes * nNodes) / float(devProp.maxThreadsDim[0]);
    // Como o número de blocos é maior do que o limite, calcula um offset
    float Offset = Blocks / float(devProp.maxThreadsDim[0]);
    int iblocks = ceil(Blocks);
    int iOffset = ceil(Offset);

    // Executa <<<1024, 1024>>> Offset vezes:
    for(int i=0; i < iOffset; i++){
        warshall_CUDA<<< devProp.maxThreadsDim[0], devProp.maxThreadsDim[0] >>> (graph, nNodes, i);
    }

}

```

### Speedup:

Para o cálculo do speedup foi feita uma média de 10 execuções do programa serial e do programa paralelo.

**Speedup obtido = 208,060**

Os testes foram realizados utilizando o google colab, aqui o [link](https://colab.research.google.com/drive/18oIRloO4_nLnZfdyjpsM5YIK1eGQC3rz#scrollTo=QJNYmWp08gYS) para o notebook.

-----------------------

# Parte 2 -> Mandelbrot em CUDA:

**Código: [mandelbrot.cu](mandelbrot/CUDA_mandelbrot.cu)**

### Kernel:

```c
__global__
void mandelbrot_CUDA(char *mat, int max_row, int max_column, int max_n)
{
        // Calcula a pos do vetor com base nos índices x e y
        int pos = (blockDim.x * threadIdx.x) + threadIdx.y;
        // Aqui calcula o offset do bloco
        pos = pos + (blockDim.x*blockDim.x) * blockIdx.x;

        // Com a posicao, calcula a linha e a coluna da matriz
        int r = float(pos) / float(max_column);
        int c = pos - (max_column * r);

        // Mesmo teste realizado no código serial
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
```

### Main:

```c
// Faz a alocação de um vetor com o tamanho total da matriz
char *mat;
cudaMallocManaged(&mat, max_row * max_column * sizeof(char));

// Caso a matriz caiba em somente um bloco de threads
if(max_row * max_column <= devProp.maxThreadsDim[0]){
    dim3 thr_per_block(max_row, max_column);
    mandelbrot_CUDA<<<1, thr_per_block>>> (mat, max_row, max_column, max_n);
}
// Caso geral:
else{
    // Calcula o número de blocos necessários:
    float Blocks = (float(max_row) * float(max_column)) / float(devProp.maxThreadsDim[1]);
    // Calcula o número de threads -> sqrt(max_threads)
    float threads = sqrt(devProp.maxThreadsDim[1]);
    int iblocks, ithreads;
    iblocks = ceil(Blocks);
    ithreads = round(threads);
    dim3 thr_per_block(ithreads, ithreads);

    // Ex.: Caso a entrada seja: 40, 40, 4
    // mandelbrot_CUDA<<<2, (32, 32)>>>  
    mandelbrot_CUDA<<<iblocks, thr_per_block>>> (mat, max_row, max_column, max_n);
}

```

### Speedup:

Para o cálculo do speedup foi feita uma média de 10 execuções do programa serial e do programa paralelo.

**Speedup obtido = 485.393**

Os testes foram realizados utilizando o google colab, aqui o [link](https://colab.research.google.com/drive/18oIRloO4_nLnZfdyjpsM5YIK1eGQC3rz#scrollTo=QJNYmWp08gYS) para o notebook.

OBS: Quando o max_n for um valor muito grande algumas posições da matriz de saída podem variar. Acreditamos que isso possa ocorrer devido a biblioteca <thrust/complex.h> não ser tão precisa quanto a biblioteca utilizada no código serial.

--------------------------

# Parte 3 -> Kmeans em MPI e OpenMP:

Código: [MPI](kmeans/kmeans_mpi.c).
Código: [OpenMP](kmeans/kmeans_omp.c).
