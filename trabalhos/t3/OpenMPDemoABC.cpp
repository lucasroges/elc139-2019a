#include <iostream>
#include <omp.h>

/*!
 * Scheduling class: executes all types of scheduling requested
 *
 */
class Scheduling{
private:
    char *array;
    int nThreads;
    int arraySize;
    int chunkSize;
public:
    Scheduling(int nThreads, int arraySize) : arraySize(arraySize), nThreads(nThreads), chunkSize(arraySize/10) {
        array = new char[arraySize];
        omp_set_num_threads(nThreads);
    }

    ~Scheduling() {
        delete array;
    }

    void dashFilling() {
        std::fill(array, array + arraySize, '-');
    }

    void printResult() {
        std::cout << "Result: " << std::string(array, arraySize) << std::endl;
        dashFilling();
    }

    void execs(int _case) {
        dashFilling();
        switch (_case) {
            case 1:
                std::cout << "Case 1: static scheduling (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(static)
                for (int i = 0; i < arraySize; i++) {
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 2:
                std::cout << "Case 2: static scheduling with chunks (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(static,chunkSize)
                for (int i = 0; i < arraySize; i++) {
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 3:
                std::cout << "Case 3: dynamic scheduling (with critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(dynamic)
                for (int i = 0; i < arraySize; i++) {
                    #pragma omp critical
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 4:
                std::cout << "Case 4: dynamic scheduling (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(dynamic)
                for (int i = 0; i < arraySize; i++) {
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 5:
                std::cout << "Case 5: dynamic scheduling with chunks (with critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(dynamic,chunkSize)
                for (int i = 0; i < arraySize; i++) {
                    #pragma omp critical
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 6:
                std::cout << "Case 6: dynamic scheduling with chunks (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(dynamic,chunkSize)
                for (int i = 0; i < arraySize; i++) {
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 7:
                std::cout << "Case 7: guided scheduling (with critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(guided)
                for (int i = 0; i < arraySize; i++) {
                    #pragma omp critical
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 8:
                std::cout << "Case 8: guided scheduling (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(guided)
                for (int i = 0; i < arraySize; i++) {
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 9:
                std::cout << "Case 9: guided scheduling with chunks (with critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(guided,chunkSize)
                for (int i = 0; i < arraySize; i++) {
                    #pragma omp critical
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 10:
                std::cout << "Case 10: guided scheduling with chunks (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(guided,chunkSize)
                for (int i = 0; i < arraySize; i++) {
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 11:
                std::cout << "Case 11: runtime scheduling (with critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(runtime)
                for (int i = 0; i < arraySize; i++) {
                    #pragma omp critical
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 12:
                std::cout << "Case 12: runtime scheduling (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(runtime)
                for (int i = 0; i < arraySize; i++) {
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            case 13:
                std::cout << "Case 13: auto scheduling (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array) schedule(auto)
                for (int i = 0; i < arraySize; i++) {
                    array[i] = 'A' + omp_get_thread_num();
                }
                break;
            default:
                break;
        }
    }
};

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Uso: ./OpenMPDemoABC <número de threads> <tamanho do array>" << std::endl;
        exit(-1);
    }
    Scheduling scheduling(atoi(argv[1]), atoi(argv[2]));
    for (int i = 1; i <= 13; i++) {
        scheduling.execs(i);
        scheduling.printResult();
    }
    return 0;
}

/*! omp_set_num_threads(atoi(argv[1]));
    int size = atoi(argv[2]);
    char *array = new char[size];
    std::fill(array, array + size, '-');
    int i;
#pragma omp parallel for shared(i,array) schedule()
    for (i = 0; i < size; i++) {
        array[i] = 'A' + omp_get_thread_num();
    }
    std::cout << std::string(array, size) << std::endl; */
