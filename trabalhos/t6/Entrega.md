[Programação Paralela](https://github.com/lucasroges/elc139-2019a) > T6

# Comunicação coletiva em MPI

- Nome: Lucas Roges de Araujo
- Disciplina: Programação Paralela

## Parte 1
O programa solicitado encontra-se em [parte1.c](parte1.c).

As alterações realizadas no código visaram a substituição de métodos como `MPI_Send` e `MPI_Recv` por chamadas coletivas, como `MPI_Bcast` e `MPI_Scatter`. A junção dos resultados já utilizava `MPI_Gatter`, mas nesse caso ainda houve pequenas mudanças nos parâmetros da função.

O primeiro conjunto de `MPI_Send` e `MPI_Recv` substituídos foi o seguinte:

```
if (myrank == 0){
    for (int i = 1; i < nproc; ++i){
      ...
      // Broadcast B to other process
      MPI_Send(B, SIZE*SIZE, MPI_INT, i, tag_A, MPI_COMM_WORLD);
      ...
    }
  } else {
    MPI_Recv(B, SIZE*SIZE, MPI_INT, MPI_ANY_SOURCE, tag_A, MPI_COMM_WORLD, &status);
    ...
  }
```

Essas duas chamadas, como o próprio comentário acima do *send* diz, realizam a tarefa de um *broadcast* ao enviarem a matriz B inteira para todos os outros processos. A divisão de trabalho na matriz B é através das colunas, mas como não é possível enviar colunas separadamente para cada processo, então a matriz inteira deve ser enviada. Dessa forma, as chamadas acima, que eram condicionais (*send* para o processo 0 e *recv* para todos os outros) foram substituídas por uma única chamada para todos os processos:

```
// Broadcast B from process 0 to other processes
MPI_Bcast(B, SIZE*SIZE, MPI_INT, 0, MPI_COMM_WORLD);
```

O segundo conjunto de `MPI_Send` e `MPI_Recv` substituídos foi o seguinte:

```
if (myrank == 0){
    for (int i = 1; i < nproc; ++i){
      ...
      // Send "Total of lines" / "Number of process" lines to other process
      MPI_Send(A[lFrom], (lTo - lFrom) * SIZE, MPI_INT, i, tag_B, MPI_COMM_WORLD);
    }
  } else {
  	...
    MPI_Recv(A[from], (to - from)*SIZE, MPI_INT, MPI_ANY_SOURCE, tag_B, MPI_COMM_WORLD, &status);   
  }
```

Essas duas chamadas, enviam `rows/nproc` linhas da matriz A para cada um dos outros processos. A divisão de trabalho na matriz A é através das linhas e, sendo assim, é possível mandar apenas parte da matriz para os outros processos. Uma chamada coletiva que visa esse tipo de operação é a `MPI_Scatter`. Dessa forma, houve a substituição do par *send* e *recv* pela operação abaixo, que assim como o *broadcast* não é condicional ao número do processo e acaba sendo invocada por todos eles. 

```
// Scatter "SIZE/nproc" rows to each process
MPI_Scatter(A, SIZE*SIZE/nproc, MPI_INT, A, SIZE*SIZE/nproc, MPI_INT, 0, MPI_COMM_WORLD);
```

Uma pequena diferença entre o *send* e *recv* e o *scatter* é que com as duas chamadas anteriores, as linhas da matriz eram posicionadas com um deslocamento de linhas, que dependia do número do processo. Já na chamada coletiva, as linhas enviadas serão posicionadas no começo da matriz, sem nenhum deslocamento. Esse novo posicionamento exigiu uma pequena alteração nas condições iniciais do primeiro *for* dos laços aninhados para cálculo.

```
printf("computing slice %d (from row %d to %d)\n", myrank, from, to-1);
for (i=from; i<to; i++) {
...
```

A alteração foi remover o deslocamento que era controlado por duas variáveis: *from* e *to*.
No código alterado, o iterador inicia em 0 e vai até a linha onde ainda há conteúdo a ser computado por esse processo.

```
printf("computing slice %d (from row %d to %d)\n", myrank, from, to-1);
for (i=0; i<SIZE/nproc; i++) {
...
```

Por fim, o código anterior já utilizava `MPI_Gatter` para unir os resultados de cada processo na matriz C do processo 0. Nesse caso, como havia deslocamento na matriz A, o resultado também era calculado em linhas deslocadas na matriz C. Os resultados de cada processo eram colocados a partir da linha `C[from]` até a linha `C[to]` e, dessa forma, essas eram as linhas que se uniriam na matriz C do processo 0, através da chamada abaixo.

```
MPI_Gather(C[from], SIZE*SIZE/nproc, MPI_INT, C, SIZE*SIZE/nproc, MPI_INT, 0, MPI_COMM_WORLD);
```

Agora, como não há mais deslocamento na matriz A e nem na matriz C, então os resultados são computados a partir da linha C[0], ou simplesmente a partir de C, como é representado na chamada.

```
MPI_Gather(C, SIZE*SIZE/nproc, MPI_INT, C, SIZE*SIZE/nproc, MPI_INT, 0, MPI_COMM_WORLD);
```

## Parte 2
<!--TODO-->

## Referências

- [Message Passing Interface (MPI)](https://computing.llnl.gov/tutorials/mpi/)  
  Tutorial do Lawrence Livermore National Laboratory (LLNL) sobre MPI.

- [Open MPI Documentation](https://www.open-mpi.org/doc/)  
  Documentação da implementação Open MPI.

- [MPI: A Message-Passing Interface Standart](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf)

- [MPI Scatter, Gather and Allgather Tutorial](http://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/)
