#include <iostream>
#include <mpi.h>
#include <ctime>
#include <sys/time.h>

long wtime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000000 + t.tv_usec;
}

void fill (double *a, double value, double wsize, double nproc) {
    for (int i = 0; i < wsize * nproc; i++) {
        a[i] = value;
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Uso: " << argv[0] << " <worksize> <repetitions>\n";
        exit(EXIT_FAILURE);
    }

    long start_time, end_time;
    int wsize = atoi(argv[1]);
    int repeat = atoi(argv[2]);

    MPI_Init(&argc, &argv);

    int nrank, nproc;
    double* a;
    double* b;
    double localSum, globalSum;
    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    a = new double[wsize * nproc];
    fill(a, 0.01, wsize, nproc);
    b = new double[wsize * nproc];
    fill(b, 1, wsize, nproc);

    if (nrank == 0) {
        start_time = wtime();
    }

    for (int i = 0; i < repeat; i++) {
        localSum = 0.0;
        for (int j = nrank * wsize; j < (nrank + 1) * wsize; j++) {
            localSum += (a[j] * b[j]);
        }
    }

    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (nrank == 0) {
        end_time = wtime();
        std::cout << globalSum << std::endl;
        std::cout << nproc << " processo(s), " << (long) (end_time - start_time) << "usec\n";
    }

    delete a;
    delete b;
    MPI_Finalize();
    return 0;
}
