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

from default_program.lidar_3d import get_step_length_chilt_func
from default_program.lidar_2d import get_step_length_half_func
from default_program.lidar_2d import get_step_length_test

def_method = default_method.cloud_method()
ori_method = original_method.cloud_method()

set_ax_2d = plot.set_plot()
set_ax_3d = plot.set_plot()

corect_x = -20
corect_y = -200

sec_2d = 0.025
sec_3d_1 = 0.1
sec_3d_2 = 0.1

# データの読み込み
dir_list = glob.glob("/Users/kai/大学/小川研/Lidar_step_length/20241120/pcd_"+str(sec_3d_1).replace(".", "")+"s/3d/*")
for dir in dir_list:
    if "nothing" in dir or "mapping" in dir:
        continue
    if "far" in dir:
        continue
    print(dir)

    dir_2d = dir.replace("3d", "2d")
    dir_3d = dir.replace("2d", "3d")

    pcd_info_2d = get_pcd_information.get_pcd_information()
    pcd_info_2d.load_pcd_dir(dir_2d)
    pcd_info_3d = get_pcd_information.get_pcd_information()
    pcd_info_3d.load_pcd_dir(dir_3d)

    set_ax_2d.set_ax_info(title="title", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", xlim=(pcd_info_2d.get_all_min()[0], pcd_info_2d.get_all_max()[0]), ylim=(pcd_info_2d.get_all_min()[1], pcd_info_2d.get_all_max()[1]), zlim=(pcd_info_2d.get_all_min()[2], pcd_info_2d.get_all_max()[2]), azim=150)
    set_ax_3d.set_ax_info(title="title", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", xlim=(pcd_info_3d.get_all_min()[0], pcd_info_3d.get_all_max()[0]), ylim=(pcd_info_3d.get_all_min()[1], pcd_info_3d.get_all_max()[1]), zlim=(pcd_info_3d.get_all_min()[2], pcd_info_3d.get_all_max()[2]), azim=150)

    left_peak_list, right_peak_list, step_length_list_2d = get_step_length_half_func.get_step(0.025, dir_2d)
    peak_list, step_length_list_3d = get_step_length_chilt_func.get_step(sec_3d_1, sec_3d_2, dir_3d)
    cross_points = get_step_length_test.get_step(0.025, dir_2d)

    # 2d
    peak_time_idx_2d = np.sort(np.array([x[0] for x in left_peak_list]+[x[0] for x in right_peak_list]))
    peak_points_2d = np.array([x[1] for x in left_peak_list]+[x[1] for x in right_peak_list])[np.argsort(np.array([x[0] for x in left_peak_list]+[x[0] for x in right_peak_list]))]
    cofficient = np.polyfit(peak_points_2d[:, 0], peak_points_2d[:, 1], 1)
    collect_theta_z = np.arctan(cofficient[0])
    rotated_peak_points_2d = def_method.rotate_points(peak_points_2d, theta_z=-collect_theta_z)
    
    
    # 3d
    peak_time_idx_3d = {}
    peak_points_3d = {}
    for group_idx, group_peak in enumerate(peak_list):
        peak_time_idx_3d[group_idx] = np.sort(np.array([x[0] for x in group_peak]))
        peak_points_3d[group_idx] = np.array([x[1] for x in group_peak])[np.argsort(np.array([x[0] for x in group_peak]))]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = set_ax_2d.set_ax(ax, title=pcd_info_2d.dir_name, xlim=[0, 15000])
    # ax.scatter(peak_points_2d[:, 0], peak_points_2d[:, 1], s=5, c="b", label="2d")
    # ax.scatter(rotated_peak_points_2d[:, 0], rotated_peak_points_2d[:, 1], s=5, c="g", label="rotated 2d")
    ax.scatter(cross_points[:, 1], np.array([0 for _ in range(len(cross_points))]), s=5, c="b", label="2d")


    for group_idx in peak_points_3d.keys():
        ax.scatter(peak_points_3d[group_idx][:, 0], peak_points_3d[group_idx][:, 1], s=5, c="r", label="3d")
    plt.legend()
    plt.show()
    plt.close()

    if False:
        # ヒストグラム
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        ax1.hist(step_length_list_2d, bins=55, range=(0, 1100))
        ax2.hist(step_length_list_3d[0], bins=55, range=(0, 1100))
        
        ax1.set_title("2d")
        ax2.set_title("3d")
        
        plt.suptitle(pcd_info_2d.dir_name)
        plt.tight_layout()
        plt.show()
        plt.close()

        # 平均値の比較
        print("2d", np.mean(step_length_list_2d))
        print("3d", np.mean(step_length_list_3d[0]))
