rm -r reports/figures/*
truncate -s 0 logs/results.csv
echo "model,dataset,num_hidden,add_edges_thres,val_loss,val_acc,test_loss,test_acc,activation,num_heads" > logs/results.csv
for THRES in 0.0 0.2 0.4 0.6
do
  for MODEL in gin gat gcn
  do
    for DATASET in pubmed citeseer cora
      do
      for N_HIDDEN in 0 1 2 5 10
      do
        echo "Running $MODEL with $N_HIDDEN hidden layers (i.e., convolution layers with hidden dimensions (dim_h, dim_h) on $DATASET"
        if [ $MODEL = gat ]; then
          for N_HEADS in 1 2 8 16
          do
            python run.py $MODEL \
            -d $DATASET \
            -nh $N_HIDDEN \
            -t $THRES \
            -a "elu" \
            --n_heads $N_HEADS \
            --plot_energy \
            --plot_rayleigh \
            --plot_influence
          done
        else
          # Use tanh, differentiable
          python run.py $MODEL \
          -d $DATASET \
          -nh $N_HIDDEN \
          -t $THRES \
          -a "tanh" \
          --plot_energy \
          --plot_rayleigh \
          --plot_influence
        fi
      done
    done
  done
done
