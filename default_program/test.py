import os
import sys
import numpy as np
import pcl
import glob
import time
import matplotlib.pyplot as plt

sys.path.append("/Users/kai/大学/小川研/Lidar_step_length")
from default_program.class_method import default_method
from default_program.class_method import original_method
from default_program.class_method import get_pcd_information
from default_program.class_method import plot
from default_program.class_method import create_gif

# ノイズ除去のクラスをインスタンス化
def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()

target_dir_list = glob.glob("/Users/kai/大学/小川研/LiDAR_step_length/20241204/pcd_01s/3d/cose_7*")
sec_1 = 0.1
sec_2 = 0.1

for target_dir in target_dir_list:
    print(target_dir)

    pcd_info_list = get_pcd_information.get_pcd_information()
    pcd_info_list.load_pcd_dir(target_dir)

    set_ax = plot.set_plot()
    set_ax.set_ax_info(pcd_info_list.dir_name, xlabel="X(mm)", ylabel="Y(mm)", zlabel="Z(mm)", xlim=[pcd_info_list.get_all_min()[0], pcd_info_list.get_all_max()[0]], ylim=[pcd_info_list.get_all_min()[1], pcd_info_list.get_all_max()[1]], zlim=[pcd_info_list.get_all_min()[2], pcd_info_list.get_all_max()[2]])

    time_area_points_path = "/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_points_list/3d/01s/"+pcd_info_list.dir_name
    time_area_center_path = "/Users/kai/大学/小川研/LiDAR_step_length/remove_noize_data/time_area_center_point_list/3d/01s/"+pcd_info_list.dir_name
    time_area_points_list, time_area_center_point_list = ori_method.load_original_data(time_area_points_path, time_area_center_path)
    integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list(time_area_points_list, time_area_center_point_list, integrade_threshold=5, distance_threshold=500)
    print(len(integraded_area_points_list))
    cloud_folder_path = target_dir.replace(str(sec_1).replace(".", "")+"s", str(sec_2).replace(".", "")+"s")
    integraded_area_points_list, integraded_area_center_point_list = ori_method.grouping_points_list_2(integraded_area_points_list, integraded_area_center_point_list, cloud_folder_path, sec=sec_2, is_incline=False)
    print(len(integraded_area_points_list))

    for time_idx in range(len(time_area_points_list)):
        fig = plt.figure()
        ax1 = fig.add_subplot(211, projection='3d')
        ax1 = set_ax.set_ax(ax1, title="time_area_points")
        ax2 = fig.add_subplot(212, projection='3d')
        ax2 = set_ax.set_ax(ax2, title="integraded_area_points")
        
        for group_idx in range(len(time_area_points_list[time_idx])):
            points = time_area_points_list[time_idx][group_idx]
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="b")
        
        for inte_group_idx in range(len(integraded_area_points_list)):
            try:
                points = integraded_area_points_list[inte_group_idx][time_idx]
                ax2.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="b")
            except:
                pass
            
        plt.suptitle(pcd_info_list.dir_name+"_"+str(time_idx))
        plt.show()
        plt.close()
