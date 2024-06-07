
PATHS="
--data_path /media/data1/KITTI
--log_dir /media/data1/jinlovespho/log/monodepth2
"

TRAINING_OPTIONS="
--model_name mono_model 
--split eigen_zhou 
--num_layers 18 
--dataset kitti
--height 192 
--width 640 
--disparity_smoothness 1e-3
--scales 0 1 2 3
--min_depth 0.1
--max_depth 80.0
"

OPTIMIZATION_OPTIONS="
--batch_size 16
--learning_rate 1e-4
--num_epochs 20 
--scheduler_step_size 15
"

SYSTEM_OPTIONS="
--num_workers 4
"

LOG_OPTIONS="
--save_frequency 5
"


JINLOVESPHO_OPTIONS="

--wdb_proj_name 20240606_MultiFrame_SelfSup_Depth
--wdb_exp_name server05_gpu1_kitti_bs16_reproduce_monodepth2
--wdb_log_path /media/data1/jinlovespho/log/monodepth2

--seed 41
"


CUDA_VISIBLE_DEVICES=1 python ../train.py   ${PATHS} \
                                            ${TRAINING_OPTIONS} \
                                            ${OPTIMIZATION_OPTIONS} \
                                            ${SYSTEM_OPTIONS} \
                                            ${LOG_OPTIONS} \
                                            ${JINLOVESPHO_OPTIONS}