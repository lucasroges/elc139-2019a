#include <iostream>
#include <omp.h>

/*!
 * Scheduling class: executes all types of scheduling requested
 */
class Scheduling{
private:
    char *array;
    int nThreads;
    int arraySize;
    int chunkSize;
    int index;
public:
    Scheduling(int nThreads, int arraySize) : arraySize(arraySize), nThreads(nThreads), chunkSize(arraySize/10) {
        array = new char[arraySize];
        omp_set_num_threads(nThreads);
        index = 0;
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
        index = 0;
    }

    void spendSomeTime() {
        for (int i = 0; i < 10000; ++i) {
            for (int j = 0; j < 100; ++j) {

            }
        }
    }

    void execs(int _case) {
        dashFilling();
        switch (_case) {
            case 1:
                std::cout << "Case 1: static scheduling (with critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(static)
                for (int i = 0; i < arraySize; ++i) {
                    #pragma omp critical
                    {
                        array[index] = 'A' + omp_get_thread_num();
                        spendSomeTime();
                    index++;
                    }
                }
                break;
            case 2:
                std::cout << "Case 2: static scheduling (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(static)
                for (int i = 0; i < arraySize; ++i) {
                    array[index] = 'A' + omp_get_thread_num();
                    spendSomeTime();
                    index++;
                }
                break;
            case 3:
                std::cout << "Case 3: static scheduling with chunks (with critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(static,chunkSize)
                for (int i = 0; i < arraySize; ++i) {
                    #pragma omp critical
                    {
                        array[index] = 'A' + omp_get_thread_num();
                        spendSomeTime();
                    index++;
                    }
                }
                break;
            case 4:
                std::cout << "Case 4: static scheduling with chunks (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(static,chunkSize)
                for (int i = 0; i < arraySize; ++i) {
                    array[index] = 'A' + omp_get_thread_num();
                    spendSomeTime();
                    index++;
                }
                break;
            case 5:
                std::cout << "Case 5: dynamic scheduling (with critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(dynamic)
                for (int i = 0; i < arraySize; ++i) {
                    #pragma omp critical
                    {
                        array[index] = 'A' + omp_get_thread_num();
                        spendSomeTime();
                    index++;
                    }
                }
                break;
            case 6:
                std::cout << "Case 6: dynamic scheduling (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(dynamic)
                for (int i = 0; i < arraySize; ++i) {
                    array[index] = 'A' + omp_get_thread_num();
                    spendSomeTime();
                    index++;
                }
                break;
            case 7:
                std::cout << "Case 7: dynamic scheduling with chunks (with critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(dynamic,chunkSize)
                for (int i = 0; i < arraySize; ++i) {
                    #pragma omp critical
                    {
                        array[index] = 'A' + omp_get_thread_num();
                        spendSomeTime();
                    index++;
                    }
                }
                break;
            case 8:
                std::cout << "Case 8: dynamic scheduling with chunks (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(dynamic,chunkSize)
                for (int i = 0; i < arraySize; ++i) {
                    array[index] = 'A' + omp_get_thread_num();
                    spendSomeTime();
                    index++;
                }
                break;
            case 9:
                std::cout << "Case 9: guided scheduling (with critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(guided)
                for (int i = 0; i < arraySize; ++i) {
                    #pragma omp critical
                    {
                        array[index] = 'A' + omp_get_thread_num();
                        spendSomeTime();
                    index++;
                    }
                }
                break;
            case 10:
                std::cout << "Case 10: guided scheduling (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(guided)
                for (int i = 0; i < arraySize; ++i) {
                    array[index] = 'A' + omp_get_thread_num();
                    spendSomeTime();
                    index++;
                }
                break;
            case 11:
                std::cout << "Case 11: guided scheduling with chunks (with critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(guided,chunkSize)
                for (int i = 0; i < arraySize; ++i) {
                    #pragma omp critical
                    {
                        array[index] = 'A' + omp_get_thread_num();
                        spendSomeTime();
                    index++;
                    }
                }
                break;
            case 12:
                std::cout << "Case 12: guided scheduling with chunks (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(guided,chunkSize)
                for (int i = 0; i < arraySize; ++i) {
                    array[index] = 'A' + omp_get_thread_num();
                    spendSomeTime();
                    index++;
                }
                break;
            case 13:
                std::cout << "Case 13: runtime scheduling (with critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(runtime)
                for (int i = 0; i < arraySize; ++i) {
                    #pragma omp critical
                    {
                        array[index] = 'A' + omp_get_thread_num();
                        spendSomeTime();
                    index++;
                    }
                }
                break;
            case 14:
                std::cout << "Case 14: runtime scheduling (w/o critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(runtime)
                for (int i = 0; i < arraySize; ++i) {
                    array[index] = 'A' + omp_get_thread_num();
                    spendSomeTime();
                    index++;
                }
                break;
            case 15:
                std::cout << "Case 15: auto scheduling (with critical)" << std::endl;
                #pragma omp parallel for shared(array,index) schedule(auto)
                for (int i = 0; i < arraySize; ++i) {
                    #pragma omp critical
                    {
                        array[index] = 'A' + omp_get_thread_num();
                        spendSomeTime();
                        index++;
                    }
                }
                break;
            case 16:
                std::cout << "Case 16: auto scheduling (w/o critical)" << std::endl;
#pragma omp parallel for shared(array,index) schedule(auto)
                for (int i = 0; i < arraySize; ++i) {
                    array[index] = 'A' + omp_get_thread_num();
                    spendSomeTime();
                    index++;
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
    for (int i = 1; i <= 16; i++) {
        scheduling.execs(i);
        scheduling.printResult();
    }
    return 0;
}