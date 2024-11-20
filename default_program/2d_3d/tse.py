import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

sys.path.append("/Users/kai/大学/小川研/Lidar_step_length")
from default_program.class_method import get_pcd_information
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import plot
from default_program.class_method import create_gif

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()
set_ax = plot.set_plot()

corect_x = -20
corect_y = -200

sec_list = ["01"]
for sec in sec_list:
    # データの読み込み
    dir_list = glob.glob(f"/Users/kai/大学/小川研/Lidar_step_length/20241120/pcd_{sec}s/2d/*")
    for dir in dir_list:
        if "nothing" in dir or "mapping" in dir:
            continue
        if "far" in dir:
            continue

        print(dir)
        pcd_info_2d = get_pcd_information.get_pcd_information()
        pcd_info_2d.load_pcd_dir(dir)
        pcd_info_3d = get_pcd_information.get_pcd_information()
        pcd_info_3d.load_pcd_dir(dir.replace("2d", "3d"))
        set_ax.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_3d.get_all_min()[0], pcd_info_3d.get_all_max()[0]), ylim=(pcd_info_3d.get_all_min()[1], pcd_info_3d.get_all_max()[1]), zlim=(pcd_info_3d.get_all_min()[2], pcd_info_3d.get_all_max()[2]), azim=150)

        area_path_2d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/2d/{sec}s/{pcd_info_2d.dir_name}"
        center_path_2d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/2d/{sec}s/{pcd_info_2d.dir_name}"
        time_area_points_list_2d, time_area_center_point_list_2d = ori_method.load_original_data(area_path_2d, center_path_2d)

        area_path_3d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/3d/{sec}s/{pcd_info_3d.dir_name}"
        center_path_3d = f"/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/3d/{sec}s/{pcd_info_3d.dir_name}"
        time_area_points_list_3d, time_area_center_point_list_3d = ori_method.load_original_data(area_path_3d, center_path_3d)
        integraded_area_points_list_3d, integraded_area_center_point_list_3d = ori_method.grouping_points_list(time_area_points_list_3d, time_area_center_point_list_3d, integrade_threshold=5)
        sec_2 = 0.1
        cloud_folder_path = "/Users/kai/大学/小川研/Lidar_step_length/20241120/pcd_"+str(sec_2).replace(".", "")+"s/3d/"+pcd_info_3d.dir_name
        integraded_area_points_list, integraded_area_center_point_list_3d = ori_method.grouping_points_list_2(integraded_area_points_list_3d, integraded_area_center_point_list_3d, cloud_folder_path, sec=sec_2, is_incline=False)

        top_percent = 0.1
        
        height_list = []
        for group_idx in range(len(integraded_area_points_list)):
            all_points = None

            for time_idx in range(len(integraded_area_points_list[group_idx])):
                if all_points is None:
                    all_points = integraded_area_points_list[group_idx][time_idx]
                else:
                    all_points = np.vstack([all_points, integraded_area_points_list[group_idx][time_idx]])

            z = all_points[:, 2]
            z = np.sort(z)[::-1]
            z = z[:int(len(z)*top_percent/100)]

            height_list.append(np.mean(z))
        
        print(height_list)
