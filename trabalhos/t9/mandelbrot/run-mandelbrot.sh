#!/bin/bash
#SBATCH -J mandelbrot           # job name
#SBATCH -o mandelbrot%j.out     # output file name (%j expands to jobID), this file captures standered output from the shell
#SBATCH -e mandelbrot%j.err     # error file name (%j expands to jobID), this file captures standered errors genereted from the program
#SBATCH --nodes 1        	# total number of nodes requested
#SBATCH --ntasks-per-node 36 	# total number cores requested per node. Using this option and --node option above you could fine tune resource requests.
#SBATCH -p qCDER            	# partition --qCDER (to find out available partitions please run 'sinfo' command)
#SBATCH -t 01:00:00       	# run time (hh:mm:ss) - 1 hour

module load Compilers/gcc-4.9.2

make clean
make
./mandelbrot < mandelbrot.in

exit
