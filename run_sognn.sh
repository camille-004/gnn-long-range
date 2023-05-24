touch logs/results_sognn.csv
truncate -s 0 logs/results_sognn.csv
echo "model,dataset,num_hidden,val_loss,val_acc,test_loss,test_acc,activation,distance" > logs/results_sognn.csv


for r in 3 4 5 6 8 10
do
  for N_HIDDEN in 1 2 3 5 10
  do
    for activation in "relu" "elu" "tanh"
    do
      echo "Running sognn with $N_HIDDEN hidden layers, r=$r and activation function $activation on wisconsin..."
      python -u run.py sognn \
      -d wisconsin \
      -nh $N_HIDDEN \
      -a $activation \
      -r $r \
      --plot_energy \
      --plot_rayleigh \
      --plot_influence
    done
  done
done
