
MODEL_NAME="sf_vit_encB_decB"

PATH_ARGS="
    --data_path /media/dataset1/kitti_raw_jpg
    --log_dir /media/dataset1/jinlovespho/log/${MODEL_NAME}
"

DATA_ARGS="
    --dataset kitti
    --split eigen_zhou
    --height 192
    --width 640
    --min_depth 0.1
    --max_depth 100.0
"

TRAINING_ARGS="
    --model_name ${MODEL_NAME}
    --disparity_smoothness 1e-3
"

OPTIMIZATION_ARGS="
    --batch_size 8
    --learning_rate 1e-4
    --num_epochs 20
    --scheduler_step_size 15
    --num_workers 4
"


LOGGING_ARGS="
    --wandb
    --wandb_proj_name 20240719_mf_depth
    --wandb_exp_name pho_server5_gpu1_kitti_${MODEL_NAME}_coloraug
    --log_frequency 250
    --save_frequency 5
"


CUDA_VISIBLE_DEVICES=1 python train.py  ${PATH_ARGS} \
                                        ${DATA_ARGS} \
                                        ${TRAINING_ARGS} \
                                        ${OPTIMIZATION_ARGS} \
                                        ${LOGGING_ARGS}