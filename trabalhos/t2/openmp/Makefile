# GNU Makefile

CC = g++ 
CCFLAGS = -Wall -std=c++11
LDFLAGS = -fopenmp
TARGET = openmp_dotprod

%.o: %.c
	$(CC) $(CCFLAGS) -c $<

%: %.o
	$(CC) $(LDFLAGS) $^ -o $@ 

all: $(TARGET)

# Dependencias

openmp_dotprod: openmp_dotprod.c

clean:
	rm -f *.o *~ $(TARGET)
