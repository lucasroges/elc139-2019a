# GNU Makefile

CC = g++
CCFLAGS = -std=c++11
LDFLAGS = -fopenmp
TARGET = OpenMPDemoABC

%.o: %.c
	$(CC) $(CCFLAGS) -c $<

%: %.o
	$(CC) $(LDFLAGS) $^ -o $@ 

all: $(TARGET)

# Dependencias

OpenMPDemoABC: OpenMPDemoABC.cpp

clean:
	rm -f *.o *~ $(TARGET)
