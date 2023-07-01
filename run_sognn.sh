touch logs/results_sognn.csv
truncate -s 0 logs/results_sognn.csv
echo "model,dataset,num_hidden,val_loss,val_acc,test_loss,test_acc,activation,distance" > logs/results_sognn.csv

for lr in 0.01 0.005 0.001
do
  for wd in 0.005 0.0005 0.000005
  do
    for dr in 0 0.2 0.3 0.5
    do
      for r in 3 5 8 10
      do
        for N_HIDDEN in 1 3 5 10
        do
          for activation in "relu"
          do
            CUDA_VISIBLE_DEVICES=3 python -u run.py sognn \
            -d cornell \
            -nh $N_HIDDEN \
            -a $activation \
            -r $r \
            -lr $lr \
            -wd $wd \
            -dr $dr \
            --tud \
            --ordered \
            -pn cornell_ordered
          done
        done
      done
    done
  done
done