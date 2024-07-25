
CUDA=1
MODEL_NAME="monodepth2_baseline"
WANDB_EXP_NAME="pho_server5_gpu${CUDA}_kitti_${MODEL_NAME}_coloraug"
MEMO="my_colorJitter_implemented_schedulerstep_outside_loop"


PATH_ARGS="
    --data_path /media/dataset1/kitti_raw_jpg
    --log_dir /media/dataset1/jinlovespho/log/${WANDB_EXP_NAME}
"

TRAINING_ARGS="
    --model_name ${MODEL_NAME}
    --split eigen_zhou
    --num_layers 18
    --dataset kitti
    --height 192
    --width 640
    --disparity_smoothness 1e-3
    --min_depth 0.1
    --max_depth 100.0
"

OPTIMIZATION_ARGS="
    --batch_size 12
    --learning_rate 1e-4
    --num_epochs 20
    --scheduler_step_size 15
    --num_workers 4
"

# LOADING_ARGS="
#     --load_weights_folder 
#     --models_to_load
# "

LOGGING_ARGS="
    --wandb
    --wandb_proj_name 20240719_mf_depth
    --wandb_exp_name ${WANDB_EXP_NAME}
    --log_frequency 250
    --save_frequency 5
    --memo ${MEMO}
"


CUDA_VISIBLE_DEVICES=${CUDA} python train.py  ${PATH_ARGS} ${TRAINING_ARGS} ${OPTIMIZATION_ARGS} ${LOADING_ARGS} ${LOGGING_ARGS}