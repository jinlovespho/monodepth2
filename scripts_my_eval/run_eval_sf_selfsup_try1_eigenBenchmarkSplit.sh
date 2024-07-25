
PATH_ARGS="
    --data_path /media/dataset1/kitti_raw_jpg
    --log_dir /media/dataset1/jinlovespho/log/monodepth2

"

TRAINING_ARGS="
    --model_name sf_selfsup_try1
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

LOGGING_ARGS="
    
    --wandb_proj_name 20240719_mf_depth
    --wandb_exp_name monodepth2_baseline
    --log_frequency 250
    --save_frequency 5
"

LOADING_ARGS="
    --load_weights_folder /media/dataset1/jinlovespho/log/monodepth2/monodepth2_baseline/models/weights_19
"

EVALUATION_ARGS="
    --eval_mono
    --eval_split eigen_benchmark
    --save_pred_disps
    --eval_out_dir /media/dataset1/jinlovespho/log/monodepth2
"


CUDA_VISIBLE_DEVICES=3 python evaluate_depth.py     ${PATH_ARGS} \
                                                    ${TRAINING_ARGS} \
                                                    ${OPTIMIZATION_ARGS} \
                                                    ${LOADING_ARGS} \
                                                    ${LOGGING_ARGS} \
                                                    ${EVALUATION_ARGS}


# -> Saving predicted disparities to  /media/dataset1/jinlovespho/log/monodepth2/save_pred_disps/sf_selfsup_try1_eigen_benchmark.npy
# -> Evaluating
#    Mono evaluation - using median scaling
#  Scaling ratios | med: 26.532 | std: 0.188

#    abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
# &   0.078  &   0.477  &   3.622  &   0.127  &   0.934  &   0.979  &   0.991  \\
