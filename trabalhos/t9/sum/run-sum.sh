#!/bin/bash
#SBATCH -J sum          	# job name
#SBATCH -o sum%j.out	       	# output file name (%j expands to jobID), this file captures standered output from the shell
#SBATCH -e sum%j.err      	# error file name (%j expands to jobID), this file captures standered errors genereted from the program
#SBATCH --nodes 1        	# total number of nodes requested
#SBATCH --ntasks-per-node 36 	# total number cores requested per node. Using this option and --node option above you could fine tune resource requests.
#SBATCH -p qCDER            	# partition --qCDER (to find out available partitions please run 'sinfo' command)
#SBATCH -t 01:00:00       	# run time (hh:mm:ss) - 1 hour

# depende de algum módulo com OpenMP 4.5
# as duas versões do gcc não suportam e as versões do compilador da intel também resultaram em erro

# module load 

make clean
make
./sum < sum.in

exit
