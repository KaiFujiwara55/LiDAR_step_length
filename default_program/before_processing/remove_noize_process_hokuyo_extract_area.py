import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

from default_program.class_method import get_pcd_information
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import plot
from default_program.class_method import create_gif

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()

def extract_points(cloud, x_min, x_max, y_min, y_max):
    points = def_method.get_points(cloud)
    
    new_points = points[(points[:, 0]>x_min) & (points[:, 0]<x_max) & (points[:, 1]>y_min) & (points[:, 1]<y_max)]
    new_cloud = def_method.get_cloud(new_points)
    
    return new_cloud, new_points

# ごみ取りを行う
sec_list = ["0025"]

for sec in sec_list:
    active_folder_list = glob.glob(f"/Users/kai/大学/小川研/LiDAR_step_length/20241204/pcd_{sec}s/2d/*")
    for active_folder in active_folder_list:
        print(active_folder)

        if "cose_6" in active_folder:
            x_min = 300
            x_max = 7300
            y_min = -500
            y_max = 500
        elif "cose_7" in active_folder or "cose_8" in active_folder:
            x_min = 300
            x_max = 10800
            y_min = -750
            y_max = 750
        else:
            x_min = 300
            x_max = 7300
            y_min = -750
            y_max = 750

        pcd_info_list = get_pcd_information.get_pcd_information()
        pcd_info_list.load_pcd_dir(active_folder)

        time_area_points_list = []
        time_area_center_point_list = []
        for cloud in tqdm(pcd_info_list.cloud_list):
            new_cloud, new_points = extract_points(cloud, x_min, x_max, y_min, y_max)

            if new_cloud.size == 0:
                time_area_points_list.append([])
                time_area_center_point_list.append([])
            else:
                new_cloud = def_method.get_cloud(new_points)
                # new_cloud = def_method.statistical_outlier_removal(new_cloud)
                area_points_list = [new_points]
                area_center_point_list = [np.mean(new_points, axis=0)]
                # area_points_list, area_center_point_list = ori_method.get_neighborhood_points(new_cloud, radius=250, count_threshold=3)
                
                time_area_points_list.append(area_points_list)
                time_area_center_point_list.append(area_center_point_list)

        # 処理結果を保存
        area_path = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/2d/{sec}s/{pcd_info_list.dir_name}"
        center_path = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/2d/{sec}s/{pcd_info_list.dir_name}"
        ori_method.save_original_data(time_area_points_list, time_area_center_point_list, area_path, center_path)
