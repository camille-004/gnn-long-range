for MODEL in gcn gat gin
do
  for DATASET in pubmed cora citeseer
    do
    for N_HIDDEN in 1 2 5 10 20 50 100 200
    do
      echo "Running $MODEL with $N_HIDDEN hidden layers on $DATASET"
      python run.py node $MODEL \
      --dataset $DATASET \
      --n_hidden $N_HIDDEN \
      --plot_energy \
      --plot_influence
    done
  done
done
