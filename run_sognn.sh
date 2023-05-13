rm -r figures/actor_results/neighbor_influences/node_*
touch logs/results_sognn.csv
truncate -s 0 logs/results_sognn.csv
echo "model,dataset,num_hidden,add_edges_thres,val_loss,val_acc,test_loss,test_acc,activation,distance" > logs/results_sognn.csv

for r in 3 4 5 6
do
  for N_HIDDEN in 1 2 3 5 10
  do
    echo "Running sognn with $N_HIDDEN hidden layers and r=$r on Pubmed..."
    python run.py sognn \
    -d Actor \
    -nh $N_HIDDEN \
    -a "elu" \
    -r $r \
    --plot_energy \
    --plot_rayleigh \
    --plot_influence
  done
done

