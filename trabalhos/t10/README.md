# T10: Maratona de Programação Paralela

### Nomes: Bruno Alves e Lucas Araujo

# Parte 1 -> Transitive Closure em CUDA:

### Kernel:

```c
__global__
void warshall_CUDA(short int* graph, int nNodes)
{   
    // k é a linha da matriz
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    // i é a coluna
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    int nNodes2 = nNodes * nNodes;
    // Testa se não passa dos limites da matriz
    // Para os casos em que a divisão não é perfeita.
    if(i < nNodes2 && k < nNodes2){
        for (int j = 0; j < nNodes; j++){
            if(graph[i * nNodes + k] + graph[k * nNodes + j] < graph[i * nNodes + j])
                graph[i * nNodes + j] = 1;
        }
    }

}
```

### Main:

```c
// Caso o número de nós seja menor que o valor máximo de threads
if(nNodes <= devProp.maxThreadsDim[0] && nNodes <= devProp.maxThreadsDim[1]){
    dim3 thr_per_block(nNodes, nNodes);
    // Cria somente 1 bloco com nNodes x nNodes threads
    warshall_CUDA<<<1, thr_per_block>>> (graph, nNodes);
}else{
    // Caso o número de threads for maior do que o valor máximo permitido:

    // Divide nNodes * nNodes pelo valor máximo permitido para a Dim0 e a Dim1
    float Blocks = (nNodes * nNodes) / (devProp.maxThreadsDim[0] * devProp.maxThreadsDim[1]);
    dim3 thr_per_block(devProp.maxThreadsDim[0], devProp.maxThreadsDim[1]);

    // Ex.: nNodes = 5120
    // Blocks = (5120 * 5120) / (1024 * 1024)
    // Blocks = 25
    // dim3 thr_per_block(1024, 1024);
    // warshall_CUDA<<<25, (1024, 1024)>>> (graph, nNodes);

    warshall_CUDA<<<ceil(Blocks), thr_per_block>>> (graph, nNodes);
}

```

### Speedup:

Para o cálculo do speedup foi feita uma média de 10 execuções do programa serial e do programa paralelo.

**Speedup obtido = 257,063**

Os testes foram realizados utilizando o google colab, aqui o [link](https://colab.research.google.com/drive/18oIRloO4_nLnZfdyjpsM5YIK1eGQC3rz#scrollTo=QJNYmWp08gYS) para o notebook.

-----------------------

# Parte 2 -> Mandelbrot em CUDA:

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
