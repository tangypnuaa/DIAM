#!/bin/bash

GPU_ID=${1}
MODEL_NUM=${2}
export CUDA_VISIBLE_DEVICES=$GPU_ID
DATASET_ARR=('mnist' 'kmnist')
METHOD_ARR=(entropy margin least_conf coreset DIAM CAL random)
BATCH_SIZE=1500

echo "Running with ${MODEL_NUM} target models on GPU ${GPU_ID}..."
echo "Settings: ${DATASET_ARR[*]} ${METHOD_ARR[*]}"

for NET in $(seq 0 1 $(($MODEL_NUM-1)))
do
    for DATASET in ${DATASET_ARR[*]}
    do
      echo "Training ${DATASET} initial model"
      python train_net.py --al_iter 0 --net_id $NET --dataset $DATASET
      if [[ $METHOD == DIAM || $METHOD == CAL ]] ; then
        python train_net.py --al_iter 0 --net_id $NET --dataset $DATASET --mDIS --DIS_frac 0.5
      fi
    done
done

for DATASET in ${DATASET_ARR[*]}
do
	for METHOD in ${METHOD_ARR[*]}
	do
      echo "Running ${DATASET} ${METHOD} active select"
      python al_select.py --method $METHOD --dataset $DATASET --al_iter 0 --batch_size $BATCH_SIZE --model_num $MODEL_NUM
  done
done

for ITER in $(seq 1 1 10);
do
  for METHOD in ${METHOD_ARR[*]}
  do

    for DATASET in ${DATASET_ARR[*]}
      do

      for NET in $(seq 0 1 $(($MODEL_NUM-1)))
      do
          echo "Running ${DATASET} ${METHOD} ${ITER}"
          python train_net.py --net_id $NET --dataset $DATASET --al_iter $ITER --method $METHOD
          if [[ $METHOD == DIAM || $METHOD == CAL ]] ; then
            echo "enable mDIS option because of ${METHOD} method"
            python train_net.py --net_id $NET --dataset $DATASET --al_iter $ITER --method $METHOD --mDIS --DIS_frac 0.5
          fi
      done

      echo "Running ${DATASET} ${METHOD} ${ITER} active select"
      python al_select.py --method $METHOD --dataset $DATASET --al_iter $ITER --batch_size $BATCH_SIZE --model_num $MODEL_NUM
    done
  done
done

echo "test phase..."
for ITER in $(seq 0 1 10);
do
  for DATASET in ${DATASET_ARR[*]}
  do
    for METHOD in ${METHOD_ARR[*]}
    do
      echo "testing ${DATASET} ${METHOD} ${ITER}"
      python test_multi_models.py --dataset $DATASET --method $METHOD --al_iter $ITER --model_num $MODEL_NUM
    done
  done
done

