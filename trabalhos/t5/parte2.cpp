#include <iostream>
#include <ctime>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char** argv) {
    srand(time(0));
    int nrank, nproc, tag = 0;
    int number = rand() % 10000;
    MPI_Status stat;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (nrank == 0) {
        std::cout << "Numero inicial: " << number << std::endl;
        number++;
        MPI_Send(&number, 1, MPI_INT, nrank + 1, tag, MPI_COMM_WORLD);
        std::cout << "Processo " << nrank << " enviou " << number << " para processo " << nrank + 1 << ".\n";
    } else if (nrank == nproc - 1) {
        MPI_Recv(&number, 1, MPI_INT, nrank - 1, tag, MPI_COMM_WORLD, &stat);
        std::cout << "Processo " << nrank << " recebeu " << number << " do processo " << nrank - 1 << ".\n";
    } else {
        MPI_Recv(&number, 1, MPI_INT, nrank - 1, tag, MPI_COMM_WORLD, &stat);
        std::cout << "Processo " << nrank << " recebeu " << number << " do processo " << nrank - 1 << ".\n";
        number++;
        MPI_Send(&number, 1, MPI_INT, nrank + 1, tag, MPI_COMM_WORLD);
        std::cout << "Processo " << nrank << " enviou " << number << " para processo " << nrank + 1 << ".\n";
    }
    MPI_Finalize();
}
