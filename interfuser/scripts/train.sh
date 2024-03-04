GPU_NUM=1
DATASET_ROOT='/home/InterFuser/dataset/'
MODEL=$1

./distributed_train.sh $GPU_NUM $DATASET_ROOT  --dataset carla --train-towns 1 2 3 4 6 7 10  --val-towns 5 \
    --train-weathers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20  --val-weathers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20\
    --model $MODEL --sched cosine --epochs 10 --warmup-epochs 1 --lr 0.005 --batch-size 16 -j 16 --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --loss_kind dec \
    --experiment test_dec\
    --pretrained
