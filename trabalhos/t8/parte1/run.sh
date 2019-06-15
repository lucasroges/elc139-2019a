#!/bin/bash
#SBATCH -J t8p1             # job name
#SBATCH -o t8p1%j.out       # output file name (%j expands to jobID), this file captures standered output from the shell
#SBATCH -e t8p1%j.err       # error file name (%j expands to jobID), this file captures standered errors genereted from the program
#SBATCH --nodes 1           # total number of nodes requested
#SBATCH --ntasks-per-node 1 # total number cores requested per node. Using this option and --node option above you could fine tune resource requests.
#SBATCH -p qCDER            # partition --qCDER (to find out available partitions please run 'sinfo' command)
#SBATCH --gres=gpu:1        # request gpus for the job,if needed, otherwise remove this line
#SBATCH -t 08:00:00         # run time (hh:mm:ss) - 8 hours

# Setar projeto de execuções
PROJETO=colab-t8p1.csv

# Ler o projeto experimental, e para cada experimento
tail -n +2 $PROJETO |
while IFS=, read -r name runnoinstdorder runno runnostdrp code width frames Blocks
do
    # Limpar valores
    export name=$(echo $name | sed "s/\"//g")
    export code=$(echo $code | sed "s/\"//g")
    export width=$(echo $width | sed "s/\"//g")
    export frames=$(echo $frames | sed "s/\"//g")

    # Definir uma chave única
    KEY="$name-$code-$width-$frames"

    compile=""
    nvprof=""

    # Configurações de compilação
    if [ "$code" = "wave" ]; then
        compile+="g++ wave.cpp -o wave"       
    else
        compile+="nvcc ${code}.cu -o wave"
        nvprof+="nvprof --log-file logs/nvprof-$KEY"
    fi

    # Compilar o programa com a versão de código apropriada
    echo "Compiling >> $compile <<"
    eval "$compile"
    ls -l wave
    #ldd wave
    sync
    
    echo $KEY

    # Prepara comando de execução
    runline=""
    runline+="$nvprof ./wave $width $frames "

    runline+="> logs/${KEY}.log"

    # Executar o experimento
    echo "Running >> $runline <<"
    eval "$runline < /dev/null"
    echo "Done!"
done

exit
