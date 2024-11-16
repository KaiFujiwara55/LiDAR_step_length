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

sec_list = ["01"]
for sec in sec_list:
    # データの読み込み
    dir_list = glob.glob(f"/Users/kai/大学/小川研/Lidar_step_length/20241113/pcd_{sec}s/2d/*")
    for dir in dir_list:
        pcd_info_2d = get_pcd_information.get_pcd_information()
        pcd_info_2d.load_pcd_dir(dir)
        pcd_info_3d = get_pcd_information.get_pcd_information()
        pcd_info_3d.load_pcd_dir(dir.replace("2d", "3d"))
        set_ax.set_ax_info(title="title", xlabel="X", ylabel="Y", zlabel="Z", xlim=(pcd_info_3d.get_all_min()[0], pcd_info_3d.get_all_max()[0]), ylim=(pcd_info_3d.get_all_min()[1], pcd_info_3d.get_all_max()[1]), zlim=(pcd_info_3d.get_all_min()[2], pcd_info_3d.get_all_max()[2]), azim=150)


        for time_idx in range(min(len(pcd_info_2d.cloud_list), len(pcd_info_3d.cloud_list))):
            cloud_2d = pcd_info_2d.cloud_list[time_idx]
            points_2d = np.array(cloud_2d)
            cloud_3d = pcd_info_3d.cloud_list[time_idx]
            points_3d = np.array(cloud_3d)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points_2d[:, 0], points_2d[:, 1], 0, c="b", s=1)
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c="r", s=1)
            ax = set_ax.set_ax(ax)
            plt.show()
            plt.close()
