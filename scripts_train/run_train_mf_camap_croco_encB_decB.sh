
CUDA=0
MODEL_NAME="mf_camap_croco_encB_decB"
WANDB_EXP_NAME="pho_server5_gpu${CUDA}_kitti_${MODEL_NAME}_color"
MEMO="enc_and_dec_lr1e-5"


PATH_ARGS="
    --data_path /media/dataset1/kitti_raw_jpg
    --log_dir /media/dataset1/jinlovespho/log/${WANDB_EXP_NAME}
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
    --pretrained_weight ./pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth
    --disparity_smoothness 1e-3
"

OPTIMIZATION_ARGS="
    --batch_size 1
    --learning_rate 1e-4
    --num_epochs 20
    --scheduler_step_size 15
    --num_workers 4
"


LOGGING_ARGS="
    
    --wandb_proj_name 20240719_mf_depth
    --wandb_exp_name ${WANDB_EXP_NAME}
    --log_frequency 250
    --save_frequency 5
    --memo ${MEMO}
"


CUDA_VISIBLE_DEVICES=${CUDA} python train.py    ${PATH_ARGS} \
                                                ${DATA_ARGS} \
                                                ${TRAINING_ARGS} \
                                                ${OPTIMIZATION_ARGS} \
                                                ${LOGGING_ARGS}