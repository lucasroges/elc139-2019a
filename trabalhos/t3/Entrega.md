[Programação Paralela](https://github.com/lucasroges/elc139-2019a) > T3

# Scheduling com OpenMP

- Nome: Lucas Roges de Araujo
- Disciplina: Programação Paralela

## Código

- Código desenvolvido disponível em [OpenMPDemoABC.cpp](OpenMPDemoABC.cpp).

## Casos de teste

- O número de threads permaneceu em 3 e o tamanho de chunk escolhido foi de `tamanho total / 10`, para todos os casos abaixo.

- Além disso, foram realizadas execuções abrangendo todos os tipos de escalonamento solicitados na [descrição do trabalho](README.md#trabalho).

- O tamanho total do vetor também foi variado entre os valores 20, 200 e 2000.

- As saídas para os casos testados encontram-se na pasta [outputs](outputs).

### Observações

- `static`: como a divisão é prévia a execução do problema, cada thread já terá os índices que deverá acessar definidos, com ou sem a especificação de *chunk*. A diferença é que com *chunks* as *threads* intercalarão os índices.

- `dynamic`: tipo de escalonamento onde existem as maiores variações em como o vetor é preenchido. A utilização de *chunks* faz com que, se escalonada, a *thread* receba vários índices ao invés de um único. Diferentemente do escalonamento estático, pode ser que uma *thread* receba `n` *chunks* na sequência, preenchedo `n * chunkSize` índices adjacentes do vetor. 

- `guided`: tipo de escalonamento semelhante ao dinâmico, porém leva em consideração o quanto de trabalho restante ainda há para ser realizado, além da quantidade de *threads* disponíveis. Normalmente escalonará *chunks*, porém a tendência é de que o tamanho dos *chunks* diminuia. Se houver uma especificação de tamanho de *chunk*, o tamanho dos *chunks* finais será no mínimo esse, exceto no caso da quantidade de índices restantes ser menor que o tamanho de *chunk* definido.

- `runtime`: definido por variável de ambiente `OMP_SCHEDULE` ou pela rotina `omp_set_schedule(omp_sched_t,chunkSize)`. Nos [*outputs*](outputs/) obtidos, não havia utilização de nenhuma dessas técnicas, portanto não houve controle sobre qual dos outros escalonamentos possíveis foi utilizado por `runtime`. Observando os [*outputs*](outputs/), a técnica semelhante ao comportamento visto é a de escalonamento dinâmico.

- `auto`: por padrão (compilador e/ou SO), é definido como `static` e segue o comportamento do mesmo.

## Referências

- [OpenMP](https://computing.llnl.gov/tutorials/openMP/)  
  Tutorial do Lawrence Livermore National Laboratory (LLNL) sobre OpenMP.

- [Jaka's Corner - OpenMP: For & Scheduling](http://jakascorner.com/blog/2016/06/omp-for-scheduling.html)  
  Material de apoio sobre a cláusula ```schedule``` do OpenMP.

- [OpenMP 4.0 API C/C++ Syntax Quick Reference Card](https://www.openmp.org/wp-content/uploads/OpenMP-4.0-C.pdf)  
  Cartão de referência para diretivas, rotinas, variáveis de ambiente e cláusulas do OpenMP (versão 4.0+). 
