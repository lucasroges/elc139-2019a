#!/bin/bash
#SBATCH --job-name=expFractalPar
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --ntasks=8
#SBATCH --partition=qCDER

# Diretório de base
export BASE=/home/users/lroges571rs

# Ingressar no diretório de base
pushd $BASE

# Criar um diretório para conter todos os resultados
rm -rf $SLURM_JOB_NAME
mkdir -p $SLURM_JOB_NAME
pushd $SLURM_JOB_NAME

# Verificar se projeto experimental foi fornecido
PROJETO=$BASE/experimentos-t4.csv
if [[ -f $PROJETO ]]; then
  echo "O projeto experimental é o seguinte:"
  cat $PROJETO | sed -e "s/^/PROJETO|/"
  # Salva o projeto no diretório corrente (da saída)
  cp $PROJETO .
else
  echo "Arquivo $PROJETO está faltando."
  exit
fi

# Verificar se os programas foram fornecidos
for i in `seq 1 3`
do
  PROGRAMA=$BASE/fractalpar$i.cpp
  if [[ -f $PROGRAMA ]]; then
    echo "O programa $i é o seguinte:"
    cat $PROGRAMA | sed -e "s/^/PROGRAMA$i|/"
    # Salva o programa no diretório corrente (da saída)
    cp $PROGRAMA .
  else
    echo "Arquivo $PROGRAMA está faltando."
    exit
  fi
done

# Verificar se o header foi fornecido
HEADER=$BASE/fractal.h
if [[ -f $HEADER ]]; then
  echo "O cabeçalho é o seguinte:"
  cat $HEADER | sed -e "s/^/HEADER|/"
  # Salva o projeto no diretório corrente (da saída)
  cp $HEADER .
else
  echo "Arquivo $HEADER está faltando."
  exit
fi

# Ler o projeto experimental, e para cada experimento
tail -n +2 $PROJETO |
while IFS=, read -r name runnoinstdorder runno runnostdrp codeVersion width frames threads Blocks
do
    # Limpar valores
    export name=$(echo $name | sed "s/\"//g")
    export codeVersion=$(echo $codeVersion | sed "s/\"//g")
    export width=$(echo $width | sed "s/\"//g")
    export frames=$(echo $frames | sed "s/\"//g")
    export threads=$(echo $threads | sed "s/\"//g")

    # Definir uma chave única
    KEY="$codeVersion-$width-$frames-$threads-$name"

    # Configurações de compilação
    CC=g++
    PROGRAMA=fractalpar${codeVersion}.cpp
    FLAGS=-fopenmp

    # Compilar o programa com a versão de código apropriada
    $CC $PROGRAMA -o fractal $FLAGS
    ls -l fractal
    ldd fractal
    sync
    
    echo $KEY

    # Prepara comando de execução
    runline=""
    runline+="./fractal $width $frames $threads "

    runline+="> ${KEY}.log"

    # Executar o experimento
    echo "Running >> $runline <<"
    eval "$runline < /dev/null"
    echo "Done!"
done

cp $BASE/slurm-$SLURM_JOB_ID.out .

exit