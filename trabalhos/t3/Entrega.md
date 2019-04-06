[Programação Paralela](https://github.com/lucasroges/elc139-2019a) > T3

# Scheduling com OpenMP

- Nome: Lucas Roges de Araujo
- Disciplina: Programação Paralela

## Código

- Código desenvolvido disponível em [OpenMPDemoABC.cpp](OpenMPDemoABD.cpp).

## Casos de teste

- O número de threads permaneceu em 3 e o tamanho de chunk escolhido foi de `tamanho total / 10`, para todos os casos abaixo.

- Além disso, foram realizadas execuções abrangendo todos os tipos de escalonamento solicitados na [descrição do trabalho](README.md).

#### Comportamentos gerais

- Ocorrem independentemente do tamanho do vetor:
	- `static`: como a divisão é prévia a execução do problema, cada thread já terá os índices que deverá acessar definidos, com ou sem a especificação de `chunk`. A diferença é que com `chunks` as threads intercalarão os índices.

	- `auto`: por padrão, é definido como `static` e segue o comportamento do mesmo.


#### Caso 1: Tamanho do vetor = 20


#### Caso 2: Tamanho do vetor = 200


#### Caso 3: Tamanho do vetor = 2000


## Referências

- [OpenMP](https://computing.llnl.gov/tutorials/openMP/)  
  Tutorial do Lawrence Livermore National Laboratory (LLNL) sobre OpenMP.

- [Jaka's Corner - OpenMP: For & Scheduling](http://jakascorner.com/blog/2016/06/omp-for-scheduling.html)  
  Material de apoio sobre a cláusula ```schedule``` do OpenMP.
