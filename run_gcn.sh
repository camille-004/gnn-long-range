touch logs/results_gcn.csv
truncate -s 0 logs/results_gcn.csv
echo "model,dataset,num_hidden,val_loss,val_acc,test_loss,test_acc,activation,distance" > logs/results_gcn.csv


for N_HIDDEN in 1 2 3 4 5 6 7 8 9 10
do
  for activation in "relu" "elu" "tanh"
  do
    echo "Running gcn with $N_HIDDEN hidden layers,and activation function $activation on Actor..."
    python -u run.py gcn \
    -d Actor \
    -nh $N_HIDDEN \
    -a $activation \
    --plot_energy \
    --plot_rayleigh \
    --plot_influence \
    -pn Actor_gcn
  done
done

