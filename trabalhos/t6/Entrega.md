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
O texto inicia falando um pouco do histórico da programação paralela e da predominância e popularidade no MPI nesse âmbito. Além disso, o autor faz referência ao texto publicado por Dijkstra, em 1968, em relação a utilização do *goto*. Dado esses dois aspectos iniciais, ele tenta fazer uma comparação de práticas prejudiciais entre o *goto*, para a programação sequencial, e o *send-receive*, para a programação paralela.

Na seção 2, o autor esclarece os parâmetros para realizar a comparação entre o modelo *send-receive* e o coletivo. Os parâmetros escolhidos foram: simplicidade, programabilidade, expressividade, desempenho e previsibilidade.

Entre as seções 3 e 7, cada um desses pontos é discutidos. As seções são abertas com um mito que defende o modelo *send-receive* e ao fim, após uma pequena discussão acerca daquele tópico, é apresentado o que os autores realmente concluem acerca daquilo que foi apresentado. Essa conclusão de cada seção é classificada como a realidade, em alusão ao mito inicialmente citado.

* **Simplicidade.** O autor inicia o tópico com o mito de que programar comunicação com as primitivas *send-receive* é a maneira mais simples de fazê-la. Para derrubar esse argumento, ele fala da quantidade de combinações possíveis para a operação de *send-receive* no MPI, onde nota-se que apesar da abrangência de possibilidades para comunicação com essa operação, a simplicidade realmente fica de lado. Por outro lado, as operações coletivas são bastante objetivas e, apesar de não apresentarem grande abrangência, são de fato mais simples de se implementar a partir do momento que o funcionamento delas é compreendido. Outra questão é a quantidade de linhas de código ocupadas por operação de *send-receive* que pode ser substituída por uma operação coletiva, onde essa substituição normalmente ocasionará numa redução da quantidade de linhas necessárias para fazer a comunicação. Essa questão foi vista na própria parte 1 do trabalho.

* **Programabilidade.** Essa seção é bem focada nas operações coletivas, apenas. O foco da seção é mostrar as possíveis transformações que podem ser efetuadas em caso de utilização de operações coletivas, o que pode resultar em menor tempo de execução, e também aumenta as possibilidades na maneira como podemos desenvolver o código para resolver um mesmo problema. Em relação as transformações, as regras de transformação são apresentadas pelo autor e incluem a validação para todas as implementações de operações coletivas e serem independentes de arquitetura. Além da fusão de operações coletivas, elas podem ser decompostas em operações menores. Na seção 7, de previsibilidade, esses conceitos de transformações também se mostram úteis para esse fator de avaliação.

* **Expressividade.** Partindo do mito que as operações coletivas não são propensas a serem aplicadas em diversas aplicações importantes, o autor traz exemplos de aplicações e subrotinas que são implementadas utilizando, exclusivamente, operações coletivas para comunicação. Muitas daquelas subrotinas são utilizadas em diversas aplicações, o que amplifica a quantidade de aplicações onde se utilizam as operações coletivas e, consequentemente, amplifica sua expressividade. Em relação a isso, apesar do pouco contato que tive com aplicações MPI de grande porte, essas poucas aplicações eram implementadas, majoritariamente, utilizando comunicação coletiva e incluindo algumas subrotinas apresentadas no artigo.

* **Desempenho.** Nesse tópico, se partirmos do mito apresentado pelo autor, onde as operações coletivas utilizam *send-receive* em nível mais baixo, faria sentido elas apresentarem pior desempenho. Porém, após a apresentação dos argumentos, fica claro que já existem diversas implementações diferentes para as operações coletivas, muitas delas baseadas nas diferentes arquiteturas existentes, levando em conta a própria rede por onde ocorrerá a comunicação. Essas diferentes implementações podem tirar melhor proveito de determinadas arquiteturas, em diferentes aplicações, apresentando melhor desempenho para determinados casos. Se a operação coletiva utiliza *send-receive* em um nível mais baixo, é previsível que tenha desempenho menor. Caso contrário, o desempenho não é previsível e dependerá das variáveis já citadas: a própria implementação e a arquitetura onde a aplicação será executada. Para reforçar seu ponto de vista, o autor traz citações de algumas publicações acerca do desempenho do *send-receive* em seus trabalhos, evidenciando pontos fracos dessa operação.

* **Previsibilidade.** Apesar do autor admitir que a previsibilidade de desempenho é mais difícil de atingir do que o desempenho absoluto, ele afirma que as operações coletivas e as transformações aplicadas sobre essas operações, apresentadas na seção 2, podem ter seu impacto calculado e contribuir para uma predição mais correta. Para reforçar, há uma tabela apresentando os cálculos com as condições de melhoria para cada transformação, onde essas estimativas foram validadas por um trabalho anterior.

Após a última dessas sessões, a seção 8 apresenta a conclusão geral em torno desses tópicos abordados e do assunto. Nela, o autor reafirma seu objetivo com a escrita do artigo, que é aconselhar a utilização de operações coletivas e evitar a utilização de operações *send-receive*. Além disso, ele reforça que sua escrita é baseada em publicações recentes e, por fim, parafraseia Dijkstra (em sua carta em relação ao *goto*) adaptando seu discurso para o MPI e destacando o caráter primitivo do *send-receive*.

## Referências

- [Message Passing Interface (MPI)](https://computing.llnl.gov/tutorials/mpi/)  
  Tutorial do Lawrence Livermore National Laboratory (LLNL) sobre MPI.

- [Open MPI Documentation](https://www.open-mpi.org/doc/)  
  Documentação da implementação Open MPI.

- [MPI: A Message-Passing Interface Standart](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf)

- [MPI Scatter, Gather and Allgather Tutorial](http://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/)

- [Send-Receive Considered Harmful: Myths and Realities of Message Passing](https://dl.acm.org/citation.cfm?id=963780)
