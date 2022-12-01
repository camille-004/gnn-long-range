rm -r reports/figures/*
truncate -s 0 logs/results.csv
echo "model, dataset, num_hidden, val_loss, val_acc, test_loss, test_acc, activation, num_heads" > logs/results.csv
for MODEL in gin gat gcn
do
  for DATASET in pubmed citeseer cora
    do
    for N_HIDDEN in 1 2 5 10
    do
      echo "Running $MODEL with $N_HIDDEN hidden layers on $DATASET"
      if [ $MODEL = gat ]; then
        for N_HEADS in 1 2 8 16
        do
          python run.py node $MODEL \
          -d $DATASET \
          -nh $N_HIDDEN \
          -a "elu" \
          --n_heads $N_HEADS \
          --plot_energy \
          --plot_rayleigh \
          --plot_influence
        done
      else
        python run.py node $MODEL \
        -d $DATASET \
        -nh $N_HIDDEN \
        -a "relu" \
        --plot_energy \
        --plot_rayleigh \
        --plot_influence
      fi
    done
  done
done
