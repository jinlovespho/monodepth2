# extract gt_depths from eigen (velodyne)
python export_gt_depth.py --data_path /media/dataset1/kitti_raw_jpg --split eigen

# extact gt_depths from eigen_benchmark (proj_depth/groundtruth)
python export_gt_depth.py --data_path /media/dataset1/kitti_raw_jpg/kitti_depth --split eigen_benchmark