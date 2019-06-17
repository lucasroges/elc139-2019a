#!/bin/bash
#SBATCH -J t8p2             # job name
#SBATCH -o t8p2%j.out       # output file name (%j expands to jobID), this file captures standered output from the shell
#SBATCH -e t8p2%j.err       # error file name (%j expands to jobID), this file captures standered errors genereted from the program
#SBATCH --nodes 1           # total number of nodes requested
#SBATCH --ntasks-per-node 1 # total number cores requested per node. Using this option and --node option above you could fine tune resource requests.
#SBATCH -p qCDER            # partition --qCDER (to find out available partitions please run 'sinfo' command)
#SBATCH --gres=gpu:1        # request gpus for the job,if needed, otherwise remove this line
#SBATCH -t 08:00:00         # run time (hh:mm:ss) - 8 hours

# Setar projeto de execuções
PROJETO=t8p2exp.csv

# Ler o projeto experimental, e para cada experimento
tail -n +2 $PROJETO |
while IFS=, read -r name runnoinstdorder runno runnostdrp width frames threads Blocks
do
    # Limpar valores
    export name=$(echo $name | sed "s/\"//g")
    export width=$(echo $width | sed "s/\"//g")
    export frames=$(echo $frames | sed "s/\"//g")
    export threads=$(echo $threads | sed "s/\"//g")

    # Definir uma chave única
    KEY="$name-$width-$frames-$threads"

    compile="nvcc wavecuda2.cu -o wave"
    nvprof="nvprof --log-file logs/nvprof-$KEY"

    # Compilar o programa
    echo "Compiling >> $compile <<"
    eval "$compile"
    ls -l wave
    #ldd wave
    sync
    
    echo $KEY

    # Prepara comando de execução
    runline=""
    runline+="$nvprof ./wave $width $frames $threads "

    runline+="> logs/${KEY}.log"

    # Executar o experimento
    echo "Running >> $runline <<"
    eval "$runline < /dev/null"
    echo "Done!"
done

exit
