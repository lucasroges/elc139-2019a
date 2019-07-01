#!/bin/bash
#SBATCH -J transitive_closure 		# job name
#SBATCH -o transitive_closure%j.out     # output file name (%j expands to jobID), this file captures standered output from the shell
#SBATCH -e transitive_closure%j.err    	# error file name (%j expands to jobID), this file captures standered errors genereted from the program
#SBATCH --nodes 1        		# total number of nodes requested
#SBATCH --ntasks-per-node 18 		# total number cores requested per node. Using this option and --node option above you could fine tune resource requests.
#SBATCH -p qCDER            		# partition --qCDER (to find out available partitions please run 'sinfo' command)
#SBATCH -t 01:00:00       		# run time (hh:mm:ss) - 1 hour

make clean
make
./transitive_closure < transitive_closure.in

exit
