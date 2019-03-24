#include <iostream>
#include <omp.h>
#include <ctime>
#include <sys/time.h>

class DotData{
public:
    double *a;
    double *b;
    double c;
    int wsize;
    int repeat;
    int nthreads;
};

DotData dotdata;

void dotProd_funct() {
    double mySum;
    for (int i = 0; i < dotdata.repeat; i++) {
        mySum = 0.0;
#pragma omp parallel for reduction(+: mySum) schedule(static)
        for (int j = 0; j < dotdata.wsize * dotdata.nthreads; j++) {
            mySum += (dotdata.a[j] * dotdata.b[j]);
        }
    }
    dotdata.c += mySum;
}

long wtime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec*1000000 + t.tv_usec;
}

void fill (double *a, double value) {
    for (int i = 0; i < dotdata.wsize * dotdata.nthreads; i++) {
        a[i] = value;
    }
}

int main(int argc, char **argv) {
    long start_time, end_time;

    if ((argc != 4)) {
        std::cout << "Uso: " << argv[0] << " <nthreads> <worksize> <repetitions>\n";
        exit(EXIT_FAILURE);
    }

    dotdata.nthreads = atoi(argv[1]);
    dotdata.wsize = atoi(argv[2]);
    dotdata.repeat = atoi(argv[3]);
    omp_set_num_threads(dotdata.nthreads);

    dotdata.a = new double[dotdata.wsize*dotdata.nthreads];
    fill(dotdata.a, 0.01);
    dotdata.b = new double[dotdata.wsize*dotdata.nthreads];
    fill(dotdata.b, 1.0);

    start_time = wtime();
    dotProd_funct();
    end_time = wtime();

    // Mostra resultado e estatisticas da execucao
    std::cout << dotdata.c << std::endl;
    std::cout << dotdata.nthreads << " thread(s), " << (long) (end_time - start_time) << "usec\n";
    //fflush(stdout);

    delete dotdata.a;
    delete dotdata.b;

    return 0;
}