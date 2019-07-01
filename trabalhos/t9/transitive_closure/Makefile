FLAGS=-O3 -fopenmp -std=c99

all: transitive_closure

transitive_closure: transitive_closure.c
	$(CC) $(FLAGS) transitive_closure.c -o transitive_closure

clean:
	rm -f transitive_closure
