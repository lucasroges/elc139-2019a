#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
	int DIM = 3;
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int i, j, k, n, c;
	double dmin, dx;
	double *x, *mean, *sum;
	int *cluster, *count, color;
	// condições de parada
	int gFlips, lFlips;
	if (rank == 0) {
		scanf("%d", &k);
		scanf("%d", &n);
	}
	MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// posição dos pontos
	x = (double *)malloc(sizeof(double)*DIM*n);
	// posição dos centróides (ponto médio do cluster)
	mean = (double *)malloc(sizeof(double)*DIM*k);
	// soma das coordenadas dos pontos de um cluster (usado para calcular ponto médio do cluster)
	sum= (double *)malloc(sizeof(double)*DIM*k);
	// cluster anterior
	cluster = (int *)malloc(sizeof(int)*n);
	// quantidade de pontos em um cluster
	count = (int *)malloc(sizeof(int)*k);
	for (i = 0; i<n; i++) 
		cluster[i] = 0;
	if (rank == 0) {
		for (i = 0; i<k; i++)
			scanf("%lf %lf %lf", mean+i*DIM, mean+i*DIM+1, mean+i*DIM+2);
		for (i = 0; i<n; i++)
			scanf("%lf %lf %lf", x+i*DIM, x+i*DIM+1, x+i*DIM+2);
	}
	MPI_Bcast(mean, 3*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(x, 3*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// calcula posição inicial e final sobre as quais cada processo itera
	int limit, offset = rank * n / size;
	if (rank == size - 1) {
		limit = n;
	} else {
		limit = (rank + 1) * n / size;
	}
	gFlips = n;
	while (gFlips>0) {
		lFlips = 0;
		gFlips = 0;

		for (j = 0; j < k; j++) {
			count[j] = 0; 
			for (i = 0; i < DIM; i++) 
				sum[j*DIM+i] = 0.0;
		}

		for (i = offset; i < limit; i++) {
			dmin = -1; color = cluster[i];
			for (c = 0; c < k; c++) {
				dx = 0.0;
				for (j = 0; j < DIM; j++) 
					dx +=  (x[i*DIM+j] - mean[c*DIM+j])*(x[i*DIM+j] - mean[c*DIM+j]);
				if (dx < dmin || dmin == -1) {
					color = c;
					dmin = dx;
				}
			}
			if (cluster[i] != color) {
				lFlips++;
				cluster[i] = color;
	      	}
		}

		MPI_Allgather(cluster+rank*n/size, n / size, MPI_INT, cluster, n / size, MPI_INT, MPI_COMM_WORLD);
		
		for (i = 0; i < n; i++) {
			count[cluster[i]]++;
			for (j = 0; j < DIM; j++) 
				sum[cluster[i]*DIM+j] += x[i*DIM+j];
		}
		for (i = 0; i < k; i++) {
			for (j = 0; j < DIM; j++) {
				mean[i*DIM+j] = sum[i*DIM+j]/count[i];
	  		}
		}

		MPI_Allreduce(&lFlips, &gFlips, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	}

	if (rank == 0) {
		for (i = 0; i < k; i++) {
			for (j = 0; j < DIM; j++)
				printf("%5.2f ", mean[i*DIM+j]);
			printf("\n");
		}
		#ifdef DEBUG
		for (i = 0; i < n; i++) {
			for (j = 0; j < DIM; j++)
				printf("%5.2f ", x[i*DIM+j]);
			printf("%d\n", cluster[i]);
		}
		#endif
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	
	return(0);
}
